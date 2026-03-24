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
    Scan a gdc-client download directory and find files matching a pattern.

    gdc-client creates ``{data_dir}/{file_uuid}/{filename}`` structure.
    This scans for files matching the glob pattern and extracts the uuid
    from the parent directory name.

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
        uuid = filepath.parent.name
        if uuid == "logs":
            continue
        result[uuid] = filepath
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
    uuid_to_path = scan_gdc_directory(data_dir, pattern)
    uuid_to_case = build_uuid_to_case_mapping(manifest_path)

    seen_cases = {}
    for uuid, filepath in uuid_to_path.items():
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
