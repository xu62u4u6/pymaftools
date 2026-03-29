"""
TCGA data readers — convert GDC-downloaded files into PivotTable objects.

Each reader handles a specific data type downloaded via gdc-client and
produces the corresponding PivotTable subclass.

Common patterns (directory scanning, manifest parsing, UUID mapping) are
extracted into shared helpers at the top of this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from ..core.CopyNumberVariationTable import CopyNumberVariationTable
from ..core.ExpressionTable import ExpressionTable
from ..core.MAF import MAF
from ..core.PivotTable import PivotTable


# ------------------------------------------------------------------ #
#  Shared helpers (used by all readers)
# ------------------------------------------------------------------ #


def read_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Read a GDC manifest TSV and return a DataFrame.

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest file (id, filename, md5, size, state columns).

    Returns
    -------
    pd.DataFrame
        Manifest with ``file_id`` as index.
    """
    df = pd.read_csv(manifest_path, sep="	", dtype=str)
    df = df.rename(columns={"id": "file_id"})
    return df.set_index("file_id")



def build_uuid_to_case_mapping(manifest_path: str | Path) -> dict[str, str]:
    """
    Build a mapping from file_uuid to case_id using a manifest + GDC API.

    The manifest only has file_id and filename. We query GDC to get the
    associated case submitter_id for each file.

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest TSV.

    Returns
    -------
    dict
        ``{file_uuid: case_submitter_id}``.
    """
    manifest = read_manifest(manifest_path)
    file_ids = manifest.index.tolist()

    mapping = {}
    batch_size = 200
    for i in range(0, len(file_ids), batch_size):
        batch = file_ids[i : i + batch_size]
        filters = {
            "op": "in",
            "content": {"field": "file_id", "value": batch},
        }
        payload = {
            "filters": json.dumps(filters),
            "fields": "file_id,cases.submitter_id",
            "size": len(batch),
            "format": "json",
        }
        response = requests.post("https://api.gdc.cancer.gov/files", json=payload)
        response.raise_for_status()
        for hit in response.json()["data"]["hits"]:
            file_id = hit["file_id"]
            case_id = hit["cases"][0]["submitter_id"] if hit.get("cases") else None
            if case_id:
                mapping[file_id] = case_id

    return mapping



def scan_gdc_directory(data_dir: str | Path, pattern: str) -> dict[str, Path]:
    """
    Scan a GDC download directory and find files matching a pattern.

    Supports two directory layouts:

    1. **gdc-client layout**: ``{data_dir}/{file_uuid}/{filename}``
       UUID is extracted from the parent directory name.
    2. **flat layout**: ``{data_dir}/{filename}``
       UUID is extracted from the manifest by matching filenames.

    Parameters
    ----------
    data_dir : str or Path
        Base directory (e.g., ``data/raw/expression``).
    pattern : str
        Glob pattern for the data files.

    Returns
    -------
    dict
        ``{file_uuid: file_path}``.
    """
    data_dir = Path(data_dir)
    result = {}
    for filepath in data_dir.rglob(pattern):
        parent = filepath.parent.name
        if parent == "logs":
            continue
        if parent == data_dir.name:
            # Flat layout: use filename stem as key (will be resolved via manifest)
            result[filepath.name] = filepath
        else:
            # gdc-client layout: parent is UUID
            result[parent] = filepath
    return result



def resolve_files_to_cases(
    data_dir: str | Path,
    pattern: str,
    manifest_path: str | Path,
) -> list[tuple[str, Path]]:
    """
    Scan directory and resolve each file to its case_id.

    Combines ``scan_gdc_directory`` and ``build_uuid_to_case_mapping``
    to produce a list of (case_id, file_path) pairs. Handles deduplication
    when multiple files map to the same case (keeps the first found).

    Parameters
    ----------
    data_dir : str or Path
        gdc-client download directory.
    pattern : str
        Glob pattern for data files.
    manifest_path : str or Path
        Path to manifest TSV.

    Returns
    -------
    list of (case_id, Path)
        Deduplicated list sorted by case_id.
    """
    key_to_path = scan_gdc_directory(data_dir, pattern)
    uuid_to_case = build_uuid_to_case_mapping(manifest_path)

    # Build filename→uuid lookup for flat layout
    manifest = read_manifest(manifest_path)
    fname_to_uuid = {}
    for file_id, row in manifest.iterrows():
        fname_to_uuid[row["filename"]] = file_id

    seen_cases = {}
    for key, filepath in key_to_path.items():
        # key is either a UUID (gdc-client layout) or filename (flat layout)
        uuid = key if key in uuid_to_case else fname_to_uuid.get(key)
        if uuid:
            case_id = uuid_to_case.get(uuid)
            if case_id and case_id not in seen_cases:
                seen_cases[case_id] = filepath

    return sorted(seen_cases.items(), key=lambda x: x[0])


# ------------------------------------------------------------------ #
#  Reader 1: Expression
# ------------------------------------------------------------------ #


def read_star_counts(
    data_dir: str | Path,
    manifest_path: str | Path,
    value_column: Literal[
        "unstranded",
        "stranded_first",
        "stranded_second",
        "tpm_unstranded",
        "fpkm_unstranded",
        "fpkm_uq_unstranded",
    ] = "unstranded",
    strip_gene_version: bool = True,
) -> ExpressionTable:
    """
    Read STAR gene counts from a gdc-client download directory.

    Scans for ``*.augmented_star_gene_counts.tsv`` files, merges them
    into a gene × sample ExpressionTable.
    """
    case_files = resolve_files_to_cases(
        data_dir, "*.augmented_star_gene_counts.tsv", manifest_path
    )

    if not case_files:
        raise FileNotFoundError(
            f"No STAR count files found in {data_dir}. "
            "Expected *.augmented_star_gene_counts.tsv inside uuid subdirectories."
        )

    qc_prefixes = {"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"}
    gene_info = None
    series_dict = {}

    for case_id, filepath in case_files:
        df = pd.read_csv(filepath, sep="	", comment="#")
        df = df[~df["gene_id"].isin(qc_prefixes)]

        if gene_info is None:
            gene_info = df[["gene_id", "gene_name", "gene_type"]].copy()

        gene_ids = df["gene_id"]
        if strip_gene_version:
            gene_ids = gene_ids.str.replace(r"\.\d+$", "", regex=True)

        series_dict[case_id] = pd.Series(
            df[value_column].values, index=gene_ids.values, name=case_id
        )

    matrix = pd.DataFrame(series_dict)

    if strip_gene_version:
        gene_info["gene_id"] = gene_info["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    feature_meta = gene_info.set_index("gene_id")[["gene_name", "gene_type"]]
    feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]

    sample_meta = pd.DataFrame({"case_id": list(series_dict.keys())}, index=list(series_dict.keys()))

    table = ExpressionTable(matrix)
    table.feature_metadata = feature_meta.reindex(matrix.index)
    table.sample_metadata = sample_meta

    print(
        f"[read_star_counts] Loaded {table.shape[0]} genes × "
        f"{table.shape[1]} samples (column: {value_column})"
    )
    return table


# ------------------------------------------------------------------ #
#  Reader 2: CNV segments
# ------------------------------------------------------------------ #


def read_seg_files(
    data_dir: str | Path,
    manifest_path: str | Path,
) -> pd.DataFrame:
    """Read TCGA masked segment files into a long-format DataFrame."""
    case_files = resolve_files_to_cases(
        data_dir, "*.nocnv_grch38.seg.v2.txt", manifest_path
    )

    if not case_files:
        raise FileNotFoundError(
            f"No SEG files found in {data_dir}. "
            "Expected *.nocnv_grch38.seg.v2.txt inside uuid subdirectories."
        )

    segments = []
    for case_id, filepath in case_files:
        df = pd.read_csv(filepath, sep="	")
        df["case_id"] = case_id
        df = df.drop(columns=["GDC_Aliquot"], errors="ignore")
        segments.append(
            df[["case_id", "Chromosome", "Start", "End", "Num_Probes", "Segment_Mean"]]
        )

    result = pd.concat(segments, ignore_index=True)
    print(
        f"[read_seg_files] Loaded {len(result)} segments across "
        f"{result['case_id'].nunique()} cases"
    )
    return result


def seg_to_cytoband_table(
    seg_df: pd.DataFrame,
    cytoband_path: str | Path | None = None,
) -> CopyNumberVariationTable:
    """
    Convert segment-level CNV data to a cytoband × sample CopyNumberVariationTable.

    For each (case, cytoband) pair, computes the overlap-weighted average of
    Segment_Mean across all overlapping segments.

    Parameters
    ----------
    seg_df : pd.DataFrame
        Long-format segment DataFrame from :func:`read_seg_files`.
        Required columns: case_id, Chromosome, Start, End, Segment_Mean.
    cytoband_path : str or Path, optional
        Path to UCSC cytoBand.txt. Defaults to the bundled file in
        ``pymaftools/data/cytoBand.txt``.

    Returns
    -------
    CopyNumberVariationTable
        Cytoband × sample matrix (weighted-mean Segment_Mean).
    """
    if cytoband_path is None:
        cytoband_path = Path(__file__).resolve().parent.parent / "data" / "cytoBand.txt"

    bands = pd.read_csv(
        cytoband_path, sep="\t", header=None,
        names=["chrom", "start", "end", "band", "stain"],
    )
    # Keep only standard chromosomes (chr1-22, chrX, chrY), drop alt/random contigs
    standard = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    bands = bands[bands["chrom"].isin(standard)].copy()
    bands["label"] = bands["chrom"] + bands["band"]  # e.g. chr1p36.33

    # Normalize chromosome naming
    seg = seg_df.copy()
    seg["Chromosome"] = seg["Chromosome"].astype(str)
    if not seg["Chromosome"].iloc[0].startswith("chr"):
        seg["Chromosome"] = "chr" + seg["Chromosome"]

    records = []
    for _, b in bands.iterrows():
        chrom, bstart, bend, label = b["chrom"], b["start"], b["end"], b["label"]
        # Find overlapping segments
        mask = (
            (seg["Chromosome"] == chrom)
            & (seg["Start"] < bend)
            & (seg["End"] > bstart)
        )
        overlap = seg.loc[mask].copy()
        if overlap.empty:
            continue

        # Compute overlap length for weighting
        overlap["ol_start"] = overlap["Start"].clip(lower=bstart)
        overlap["ol_end"] = overlap["End"].clip(upper=bend)
        overlap["ol_len"] = overlap["ol_end"] - overlap["ol_start"]

        # Weighted mean per case
        weighted = (
            overlap.groupby("case_id")
            .apply(
                lambda g: (g["Segment_Mean"] * g["ol_len"]).sum() / g["ol_len"].sum(),
                include_groups=False,
            )
            .rename(label)
        )
        records.append(weighted)

    matrix = pd.DataFrame(records)
    matrix.index.name = "cytoband"

    # Feature metadata: chromosome, arm, band, stain
    feature_meta = bands.set_index("label")[["chrom", "start", "end", "stain"]].copy()
    feature_meta = feature_meta.rename(columns={"chrom": "chromosome"})
    feature_meta["arm"] = feature_meta.index.str.extract(r"chr\w+([pq])")[0].values
    feature_meta = feature_meta.reindex(matrix.index)

    sample_meta = pd.DataFrame({"case_id": matrix.columns}, index=matrix.columns)

    table = CopyNumberVariationTable(matrix)
    table.feature_metadata = feature_meta
    table.sample_metadata = sample_meta

    print(
        f"[seg_to_cytoband_table] {table.shape[0]} cytobands × "
        f"{table.shape[1]} samples"
    )
    return table


# ------------------------------------------------------------------ #
#  Reader 2b: Gene-level CNV
# ------------------------------------------------------------------ #


def read_gene_level_cnv(
    data_dir: str | Path,
    manifest_path: str | Path,
    value_column: str = "copy_number",
    strip_gene_version: bool = True,
) -> CopyNumberVariationTable:
    """
    Read TCGA gene-level copy number files into a gene × sample CopyNumberVariationTable.

    Scans for ``*.gene_level_copy_number.v36.tsv`` files produced by the
    ASCAT3 pipeline available on GDC.

    Parameters
    ----------
    data_dir : str or Path
        gdc-client download directory containing uuid subdirectories.
    manifest_path : str or Path
        Path to manifest TSV.
    value_column : str, default "copy_number"
        Column to use as values. Typical columns in the file:
        ``copy_number``, ``min_copy_number``, ``max_copy_number``.
    strip_gene_version : bool, default True
        If True, strip ENSG version suffix (e.g. ``.15``).

    Returns
    -------
    CopyNumberVariationTable
        Gene × sample matrix with feature_metadata (gene_name, chromosome,
        start, end) and sample_metadata (case_id).
    """
    # Support both ASCAT3 filename variants
    case_files = resolve_files_to_cases(
        data_dir, "*.gene_level_copy_number.v36.tsv", manifest_path
    )
    if not case_files:
        case_files = resolve_files_to_cases(
            data_dir, "*.gene_level.copy_number_variation.tsv", manifest_path
        )

    if not case_files:
        raise FileNotFoundError(
            f"No gene-level CNV files found in {data_dir}. "
            "Expected *.gene_level_copy_number.v36.tsv or "
            "*.gene_level.copy_number_variation.tsv inside uuid subdirectories."
        )

    gene_info = None
    series_dict = {}

    for case_id, filepath in case_files:
        df = pd.read_csv(filepath, sep="\t")

        if gene_info is None:
            info_cols = [c for c in ["gene_id", "gene_name", "chromosome", "start", "end"] if c in df.columns]
            gene_info = df[info_cols].copy()

        gene_ids = df["gene_id"]
        if strip_gene_version:
            gene_ids = gene_ids.str.replace(r"\.\d+$", "", regex=True)

        series_dict[case_id] = pd.Series(
            df[value_column].values, index=gene_ids.values, name=case_id
        )

    matrix = pd.DataFrame(series_dict)

    # Build feature metadata
    if strip_gene_version:
        gene_info["gene_id"] = gene_info["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    feature_meta = gene_info.set_index("gene_id")
    feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]

    sample_meta = pd.DataFrame(
        {"case_id": list(series_dict.keys())},
        index=list(series_dict.keys()),
    )

    table = CopyNumberVariationTable(matrix)
    table.feature_metadata = feature_meta.reindex(matrix.index)
    table.sample_metadata = sample_meta

    print(
        f"[read_gene_level_cnv] Loaded {table.shape[0]} genes × "
        f"{table.shape[1]} samples (column: {value_column})"
    )
    return table


# ------------------------------------------------------------------ #
#  Reader 3: Mutation MAF
# ------------------------------------------------------------------ #


def read_maf_files(
    data_dir: str | Path,
    manifest_path: str | Path,
    nonsynonymous_only: bool = False,
) -> MAF:
    """Read TCGA masked MAF files and merge them into a single MAF object."""
    case_files = resolve_files_to_cases(data_dir, "*.maf.gz", manifest_path)

    if not case_files:
        raise FileNotFoundError(
            f"No MAF files found in {data_dir}. "
            "Expected *.maf.gz inside uuid subdirectories."
        )

    maf_frames = []
    for case_id, filepath in case_files:
        df = pd.read_csv(filepath, sep="	", comment="#", low_memory=False)
        df["sample_ID"] = case_id
        maf_frames.append(df)

    merged = pd.concat(maf_frames, ignore_index=True)
    if nonsynonymous_only:
        merged = merged[merged["Variant_Classification"].isin(MAF.nonsynonymous_types)]

    merged_maf = MAF(merged)
    merged_maf.index = merged_maf.loc[:, MAF.index_col].apply(
        lambda row: "|".join(row.astype(str)), axis=1
    )

    print(
        f"[read_maf_files] Loaded {len(merged_maf)} mutations across "
        f"{merged_maf['sample_ID'].nunique()} cases"
    )
    return merged_maf


# ------------------------------------------------------------------ #
#  Reader 4: Methylation beta values
# ------------------------------------------------------------------ #


def read_methylation_betas(
    data_dir: str | Path,
    manifest_path: str | Path,
) -> PivotTable:
    """Read TCGA methylation beta files into a probe × sample PivotTable."""
    case_files = resolve_files_to_cases(
        data_dir, "*.methylation_array.sesame.level3betas.txt", manifest_path
    )

    if not case_files:
        raise FileNotFoundError(
            f"No methylation beta files found in {data_dir}. "
            "Expected *.methylation_array.sesame.level3betas.txt inside uuid subdirectories."
        )

    series_dict = {}
    for case_id, filepath in case_files:
        df = pd.read_csv(
            filepath,
            sep="	",
            header=None,
            names=["probe_id", "beta"],
            index_col="probe_id",
        )
        series_dict[case_id] = df["beta"]

    matrix = pd.DataFrame(series_dict)
    table = PivotTable(matrix)
    table.sample_metadata = pd.DataFrame({"case_id": matrix.columns}, index=matrix.columns)

    print(
        f"[read_methylation_betas] Loaded {table.shape[0]} probes × "
        f"{table.shape[1]} samples"
    )
    return table


# ------------------------------------------------------------------ #
#  Reader 5: Clinical tables
# ------------------------------------------------------------------ #


def read_clinical(
    data_dir: str | Path,
    file_type: Literal["patient", "drug", "radiation", "follow_up", "nte"] = "patient",
) -> pd.DataFrame:
    """Read TCGA BCR clinical TXT tables."""
    file_map = scan_gdc_directory(data_dir, f"*_clinical_{file_type}_*.txt")

    if not file_map:
        raise FileNotFoundError(
            f"No clinical {file_type} files found in {data_dir}. "
            f"Expected *_clinical_{file_type}_*.txt inside uuid subdirectories."
        )

    frames = []
    for filepath in file_map.values():
        df = pd.read_csv(filepath, sep="	", skiprows=[1, 2], dtype=str)
        frames.append(df)

    clinical = pd.concat(frames, ignore_index=True)
    clinical = clinical.drop_duplicates(subset="bcr_patient_barcode", keep="first")
    clinical = clinical.set_index("bcr_patient_barcode", drop=False)

    print(
        f"[read_clinical] Loaded {len(clinical)} rows from "
        f"{len(file_map)} {file_type} file(s)"
    )
    return clinical
