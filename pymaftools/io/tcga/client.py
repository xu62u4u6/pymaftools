"""
TCGA / GDC data access module.

Provides GDCClient for querying, aligning, and downloading multi-omics data
from the Genomic Data Commons (GDC) API.

Examples
--------
>>> from pymaftools.io import GDCClient
>>> client = GDCClient.from_config("config.toml")
>>> client.generate_full_manifests(outdir="data/manifests/full")
>>> client.align_manifests(outdir="data/manifests/aligned")
"""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_STATUS_ENDPOINT = "https://api.gdc.cancer.gov/status"

# Default data type configs (used when no config.toml is provided)
DATA_TYPE_CONFIGS = {
    "expression": {
        "data_type": "Gene Expression Quantification",
        "workflow_type": "STAR - Counts",
        "label": "Gene Expression (STAR Counts)",
    },
    "mutation": {
        "data_type": "Masked Somatic Mutation",
        "label": "Somatic Mutation (MAF)",
    },
    "cnv_seg": {
        "data_type": "Allele-specific Copy Number Segment",
        "workflow_type": "ASCAT3",
        "label": "Copy Number Segment (ASCAT3)",
    },
    "cnv_gene": {
        "data_type": "Gene Level Copy Number",
        "workflow_type": "ASCAT3",
        "label": "Copy Number Gene Level (ASCAT3)",
    },
    "methylation": {
        "data_type": "Methylation Beta Value",
        "workflow_type": "SeSAMe Methylation Beta Estimation",
        "label": "DNA Methylation (SeSAMe)",
    },
}

BATCH_SIZE = 200
MAX_RETRIES = 3


def parse_tcga_barcode(barcode: str) -> dict:
    """
    Parse a TCGA barcode into its components.

    Parameters
    ----------
    barcode : str
        TCGA barcode (e.g., ``"TCGA-44-2655-01A-01D-0182-01"``).

    Returns
    -------
    dict
        Parsed components: project, tss, participant, case_id,
        and optionally sample_type, vial, portion, analyte, plate, center.
    """
    parts = barcode.split("-")
    result = {
        "project": parts[0],
        "tss": parts[1],
        "participant": parts[2],
        "case_id": "-".join(parts[:3]),
    }
    if len(parts) > 3:
        result["sample_type"] = int(parts[3][:2])
        result["vial"] = parts[3][2:] if len(parts[3]) > 2 else None
        result["is_tumor"] = result["sample_type"] < 10
    if len(parts) > 4:
        result["portion"] = parts[4][:2]
        result["analyte"] = parts[4][2:] if len(parts[4]) > 2 else None
    if len(parts) > 5:
        result["plate"] = parts[5]
    if len(parts) > 6:
        result["center"] = parts[6]
    return result


def _is_tumor_sample(sample: dict) -> bool:
    """Return True when sample metadata indicates a tumor sample."""
    submitter_id = sample.get("submitter_id", "")
    parts = submitter_id.split("-") if submitter_id else []
    if len(parts) > 3 and len(parts[3]) >= 2 and parts[3][:2].isdigit():
        return int(parts[3][:2]) < 10

    sample_type = sample.get("sample_type")
    if isinstance(sample_type, int):
        return sample_type < 10
    if isinstance(sample_type, str) and sample_type[:2].isdigit():
        return int(sample_type[:2]) < 10

    return False


def _pick_preferred_sample(samples: list[dict]) -> dict | None:
    """Prefer tumor sample in tumor-normal paired records."""
    if not samples:
        return None

    for sample in samples:
        if _is_tumor_sample(sample):
            return sample
    return samples[0]


def _tcga_entity_ids(aliquot_id: str) -> dict[str, str | None]:
    """Derive sample, portion, and analyte barcodes from a TCGA aliquot."""
    parts = aliquot_id.split("-")
    if len(parts) < 5:
        return {"sample_id": None, "portion_id": None, "analyte_id": None}
    sample_id = "-".join(parts[:4])
    portion_id = "-".join([*parts[:4], parts[4][:2]])
    analyte_id = "-".join(parts[:5])
    return {
        "sample_id": sample_id,
        "portion_id": portion_id,
        "analyte_id": analyte_id,
    }


def _extract_biospecimen_metadata(hit: dict) -> dict[str, object]:
    """Resolve the exact tumor aliquot associated with one GDC file.

    The GDC ``cases.samples`` tree contains every biospecimen for the case, so
    selecting its first tumor sample can link a file to the wrong specimen.
    ``associated_entities`` is the file-level relationship and is therefore
    the authoritative starting point.
    """
    cases = hit.get("cases") or []
    case = cases[0] if len(cases) == 1 else {}
    associated = [
        entity
        for entity in (hit.get("associated_entities") or [])
        if entity.get("entity_type") == "aliquot" and entity.get("entity_submitter_id")
    ]
    tumor_entities = [
        entity
        for entity in associated
        if _is_tumor_sample({"submitter_id": entity["entity_submitter_id"]})
    ]
    normal_entities = [entity for entity in associated if entity not in tumor_entities]

    selected = tumor_entities[0] if len(tumor_entities) == 1 else None
    if len(cases) != 1:
        status = "ambiguous_case"
    elif len(tumor_entities) == 0:
        status = "no_tumor_aliquot"
    elif len(tumor_entities) > 1:
        status = "ambiguous_tumor_aliquot"
    else:
        status = "resolved_tumor_aliquot"

    aliquot_id = selected.get("entity_submitter_id") if selected else None
    derived = (
        _tcga_entity_ids(str(aliquot_id))
        if aliquot_id is not None
        else {"sample_id": None, "portion_id": None, "analyte_id": None}
    )
    samples = case.get("samples") or []
    sample = next(
        (
            value
            for value in samples
            if value.get("submitter_id") == derived["sample_id"]
        ),
        {},
    )

    return {
        "case_id": case.get("submitter_id"),
        "project": (case.get("project") or {}).get("project_id"),
        "sample_id": derived["sample_id"],
        "sample_uuid": sample.get("sample_id"),
        "sample_type": sample.get("sample_type"),
        "portion_id": derived["portion_id"],
        "analyte_id": derived["analyte_id"],
        "aliquot_id": aliquot_id,
        "aliquot_uuid": selected.get("entity_id") if selected else None,
        "paired_normal_aliquot_ids": ";".join(
            sorted(str(entity["entity_submitter_id"]) for entity in normal_entities)
        ),
        "associated_entity_count": len(associated),
        "tumor_aliquot_count": len(tumor_entities),
        "mapping_status": status,
    }


class GDCClient:
    """
    Client for querying, aligning, and downloading TCGA data from GDC.

    Parameters
    ----------
    token_path : str or Path, optional
        GDC authentication token (required for controlled-access data).
    data_types : dict, optional
        Data type configurations keyed by label. Each entry must have
        ``data_type`` (GDC field) and optionally ``workflow_type``.
        Defaults to ``DATA_TYPE_CONFIGS``.
    projects : list of str, optional
        GDC project IDs (e.g., ``["TCGA-LUAD", "TCGA-LUSC"]``).
    gdc_client_path : str or Path, optional
        Path to the gdc-client binary. Auto-detected from PATH if not set.
    threads : int
        Download threads passed to gdc-client (default 8).
    retries : int
        Retry attempts for gdc-client (default 5).

    Examples
    --------
    >>> # From config file (recommended)
    >>> client = GDCClient.from_config("config.toml")
    >>> client.generate_full_manifests(outdir="data/manifests/full")
    >>> client.align_manifests(outdir="data/manifests/aligned")

    >>> # Programmatic
    >>> client = GDCClient(projects=["TCGA-LUAD"])
    >>> aligned = client.align_cases("TCGA-LUAD")
    """

    def __init__(
        self,
        token_path: Optional[str | Path] = None,
        data_types: Optional[dict] = None,
        projects: Optional[list[str]] = None,
        gdc_client_path: Optional[str | Path] = None,
        threads: int = 8,
        retries: int = 5,
    ):
        self.token = None
        self.token_path: Path | None = None
        if token_path:
            p = Path(token_path).expanduser()
            if p.exists():
                self.token = p.read_text().strip()
                self.token_path = p

        self.data_types = data_types or DATA_TYPE_CONFIGS
        self.projects = projects or []
        self.gdc_client_path = str(gdc_client_path) if gdc_client_path else None
        self.threads = threads
        self.retries = retries

    @classmethod
    def from_config(cls, config_path: str | Path = "config.toml") -> "GDCClient":
        """
        Create a GDCClient from a TOML config file.

        Parameters
        ----------
        config_path : str or Path
            Path to a TOML file with the following structure::

                projects = ["TCGA-LUAD", "TCGA-LUSC"]

                [download]
                gdc_client = "tools/gdc-client"
                token      = "/path/to/gdc-token.txt"
                threads    = 8
                retries    = 5

                [data_types.expression]
                data_type     = "Gene Expression Quantification"
                workflow_type = "STAR - Counts"

                [data_types.mutation]
                data_type     = "Masked Somatic Mutation"
                workflow_type = "Aliquot Ensemble Somatic Variant Merging and Masking"

            Each ``[data_types.<label>]`` entry must have ``data_type`` and
            optionally ``workflow_type``. Labels become manifest filename
            prefixes and raw download directory names.

        Returns
        -------
        GDCClient
        """
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        dl = cfg.get("download", {})
        return cls(
            token_path=dl.get("token"),
            data_types=cfg.get("data_types", DATA_TYPE_CONFIGS),
            projects=cfg.get("projects", []),
            gdc_client_path=dl.get("gdc_client"),
            threads=dl.get("threads", 8),
            retries=dl.get("retries", 5),
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _query_files(
        self,
        project_id: str,
        data_type: str,
        workflow_type: Optional[str] = None,
        case_ids: Optional[list[str]] = None,
        fields: str = "file_id,file_name,file_size,md5sum,state,cases.submitter_id",
        size: int = 5000,
    ) -> list[dict]:
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": project_id,
                    },
                },
                {"op": "=", "content": {"field": "data_type", "value": data_type}},
            ],
        }
        if workflow_type:
            filters["content"].append(
                {
                    "op": "=",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": workflow_type,
                    },
                }
            )
        if case_ids:
            filters["content"].append(
                {
                    "op": "in",
                    "content": {"field": "cases.submitter_id", "value": case_ids},
                }
            )
        payload = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": size,
            "format": "json",
        }
        r = requests.post(GDC_FILES_ENDPOINT, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["data"]["hits"]

    def _batch_query_metadata(self, file_ids: list[str]) -> list[dict]:
        """Batch query file and exact file-associated biospecimen metadata."""
        fields = (
            "file_id,file_name,file_size,md5sum,state,data_type,"
            "data_category,data_format,experimental_strategy,associated_entities,"
            "analysis.workflow_type,analysis.updated_datetime,"
            "cases.submitter_id,cases.samples.submitter_id,"
            "cases.samples.sample_id,cases.samples.sample_type,"
            "cases.samples.portions.submitter_id,"
            "cases.samples.portions.analytes.submitter_id,"
            "cases.samples.portions.analytes.aliquots.submitter_id,"
            "cases.project.project_id"
        )
        expand = (
            "associated_entities,cases.samples,cases.samples.portions,"
            "cases.samples.portions.analytes,"
            "cases.samples.portions.analytes.aliquots"
        )
        results = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                filters = {
                    "op": "in",
                    "content": {"field": "file_id", "value": file_ids},
                }
                payload = {
                    "filters": json.dumps(filters),
                    "fields": fields,
                    "expand": expand,
                    "size": len(file_ids),
                    "format": "json",
                }
                r = requests.post(GDC_FILES_ENDPOINT, json=payload, timeout=60)
                r.raise_for_status()
                return r.json()["data"]["hits"]
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                tqdm.write(f"  retry {attempt}: {e}")
                time.sleep(2**attempt)
        return results

    @staticmethod
    def _write_manifest(hits: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("id\tfilename\tmd5\tsize\tstate\n")
            for h in hits:
                f.write(
                    f"{h['file_id']}\t{h['file_name']}\t"
                    f"{h.get('md5sum', '')}\t{h['file_size']}\t{h.get('state', '')}\n"
                )

    def _find_gdc_client(self) -> str:
        if self.gdc_client_path and Path(self.gdc_client_path).exists():
            return self.gdc_client_path
        found = shutil.which("gdc-client")
        if found:
            return found
        for candidate in [
            Path("tools/gdc-client"),
            Path.home() / ".local/bin/gdc-client",
        ]:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "gdc-client not found. Set gdc_client_path or add it to PATH."
        )

    # ── Case alignment ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return the current public GDC API and data-release metadata."""
        response = requests.get(GDC_STATUS_ENDPOINT, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_file_manifest(
        self,
        projects: Optional[list[str]] = None,
        data_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Query a provenance-rich file manifest without downloading data."""
        projects = projects or self.projects
        modalities = data_types or list(self.data_types)
        if not projects:
            raise ValueError("projects must contain at least one GDC project ID.")

        records = []
        for project in projects:
            for modality in modalities:
                config = self.data_types[modality]
                hits = self._query_files(
                    project,
                    config["data_type"],
                    config.get("workflow_type"),
                )
                for hit in hits:
                    records.append(
                        {
                            "file_id": hit["file_id"],
                            "filename": hit["file_name"],
                            "data_type": modality,
                            "gdc_data_type": config["data_type"],
                            "workflow_type": config.get("workflow_type"),
                            "md5": hit.get("md5sum", ""),
                            "size": hit.get("file_size", 0),
                            "state": hit.get("state", ""),
                        }
                    )
        return self.build_file_mapping(records)

    def get_cases(self, project_id: str, data_type_key: str) -> set[str]:
        """Get all case submitter_ids that have a given data type."""
        cfg = self.data_types[data_type_key]
        hits = self._query_files(
            project_id,
            cfg["data_type"],
            cfg.get("workflow_type"),
            fields="cases.submitter_id",
        )
        return {c["submitter_id"] for hit in hits for c in hit.get("cases", [])}

    def align_cases(
        self,
        project_id: str,
        data_types: Optional[list[str]] = None,
    ) -> list[str]:
        """Find cases with all specified data types (set intersection)."""
        keys = data_types or list(self.data_types.keys())
        case_sets = {}
        for key in keys:
            cases = self.get_cases(project_id, key)
            case_sets[key] = cases
            label = self.data_types[key].get("label", key)
            print(f"  {label}: {len(cases)} cases")
        aligned = set.intersection(*case_sets.values())
        print(f"  >>> aligned ({len(keys)}-way): {len(aligned)} cases")
        return sorted(aligned)

    def align_multi_project(
        self,
        project_ids: Optional[list[str]] = None,
        data_types: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """Align cases across multiple projects."""
        projects = project_ids or self.projects
        result = {}
        for proj in projects:
            print(f"\n{proj}")
            result[proj] = self.align_cases(proj, data_types)
        total = sum(len(v) for v in result.values())
        print(f"\nTotal aligned: {total} cases")
        return result

    # ── Manifest generation ───────────────────────────────────────────────────

    def generate_full_manifests(
        self,
        projects: Optional[list[str]] = None,
        outdir: str | Path = "data/manifests/full",
        mapping_path: str | Path = "data/file_to_case.tsv",
    ) -> Path:
        """
        Generate complete per-dtype manifests (no case filtering) and
        build a file_id → case_id / sample_type / project mapping table.

        Parameters
        ----------
        projects : list of str, optional
            Project IDs. Defaults to ``self.projects``.
        outdir : str or Path
            Output directory for full manifests.
        mapping_path : str or Path
            Output path for file_to_case.tsv.

        Returns
        -------
        Path
            Output manifest directory.
        """
        projects = projects or self.projects
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        print("=== Generating full manifests ===\n")
        records = []

        for label, dt in self.data_types.items():
            label_hits = []
            for proj in projects:
                hits = self._query_files(proj, dt["data_type"], dt.get("workflow_type"))
                label_hits.extend(hits)
                print(f"  {proj} / {label}: {len(hits)} files")

            self._write_manifest(label_hits, outdir / f"manifest_{label}.tsv")
            for h in label_hits:
                records.append(
                    {
                        "file_id": h["file_id"],
                        "filename": h["file_name"],
                        "data_type": label,
                        "gdc_data_type": dt["data_type"],
                        "workflow_type": dt.get("workflow_type"),
                        "md5": h.get("md5sum", ""),
                        "size": h.get("file_size", 0),
                        "state": h.get("state", ""),
                    }
                )

            total_gb = sum(h["file_size"] for h in label_hits) / 1e9
            print(
                f"  → manifest_{label}.tsv ({len(label_hits)} files, {total_gb:.1f} GB)\n"
            )

        mapping_df = self.build_file_mapping(records)
        mapping_df.to_csv(mapping_path, sep="\t", index=False)
        print(f"Saved file mapping: {mapping_path} ({len(mapping_df)} rows)")
        return outdir

    def build_file_mapping(self, records: list[dict]) -> pd.DataFrame:
        """
        Build a file-level mapping to exact tumor biospecimen entities.

        Parameters
        ----------
        records : list of dict
            Each dict must have ``file_id`` and ``filename``. ``data_type`` is
            the Pymaftools modality label; legacy ``dtype`` is accepted.

        Returns
        -------
        pd.DataFrame
            One row per file, including case, sample, portion, analyte,
            aliquot, workflow, checksum, and mapping status.
        """
        unique_ids = list({r["file_id"] for r in records})
        print(f"\nBuilding file mapping ({len(unique_ids)} files)...")

        mapping: dict[str, dict[str, object]] = {}
        for i in tqdm(
            range(0, len(unique_ids), BATCH_SIZE), desc="Querying GDC", unit="batch"
        ):
            batch = unique_ids[i : i + BATCH_SIZE]
            for hit in self._batch_query_metadata(batch):
                fid = hit["file_id"]
                analysis = hit.get("analysis") or {}
                mapping[fid] = {
                    **_extract_biospecimen_metadata(hit),
                    "gdc_data_type": hit.get("data_type"),
                    "workflow_type": analysis.get("workflow_type"),
                    "analysis_updated_datetime": analysis.get("updated_datetime"),
                    "md5": hit.get("md5sum", ""),
                    "size": hit.get("file_size", 0),
                    "state": hit.get("state", ""),
                }

        df = pd.DataFrame(records)
        if "data_type" not in df.columns and "dtype" in df.columns:
            df = df.rename(columns={"dtype": "data_type"})
        if mapping:
            metadata = pd.DataFrame.from_dict(mapping, orient="index")
            metadata.index.name = "file_id"
            metadata = metadata.reset_index()
        else:
            metadata = pd.DataFrame({"file_id": pd.Series(dtype=str)})
        api_columns = [
            column
            for column in metadata.columns
            if column == "file_id" or column not in df.columns
        ]
        df = df.merge(metadata[api_columns], on="file_id", how="left")
        if "mapping_status" not in df.columns:
            df["mapping_status"] = "missing_api_metadata"
        else:
            df["mapping_status"] = df["mapping_status"].fillna("missing_api_metadata")
        return df

    @staticmethod
    def align_specimens(
        file_mapping: pd.DataFrame,
        data_types: Optional[list[str]] = None,
        *,
        sample_type: str = "Primary Tumor",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select only cases with one exact shared specimen across modalities.

        Case identifiers are participant-level and cannot establish that two
        files came from the same tissue specimen. This method intersects exact
        TCGA sample barcodes and rejects missing, ambiguous, duplicated, or
        cross-vial inputs instead of selecting a file by UUID order.

        Returns
        -------
        selected_files, alignment_report : tuple of pandas.DataFrame
            ``selected_files`` contains exactly one file per case and modality.
            ``alignment_report`` contains every candidate case and its attrition
            reason.
        """
        required = {
            "file_id",
            "case_id",
            "data_type",
            "sample_id",
            "sample_type",
            "mapping_status",
        }
        missing_columns = sorted(required - set(file_mapping.columns))
        if missing_columns:
            raise ValueError(
                "file_mapping is missing specimen-alignment column(s): "
                f"{missing_columns}."
            )
        if file_mapping["file_id"].duplicated().any():
            duplicates = file_mapping.loc[
                file_mapping["file_id"].duplicated(keep=False), "file_id"
            ].unique()
            raise ValueError(
                f"file_mapping contains duplicate file_id values: {duplicates.tolist()}."
            )

        modalities = data_types or sorted(file_mapping["data_type"].dropna().unique())
        if not modalities:
            raise ValueError("data_types must contain at least one modality.")

        candidates = file_mapping.loc[
            file_mapping["sample_type"].eq(sample_type)
            & file_mapping["data_type"].isin(modalities)
            & file_mapping["mapping_status"].eq("resolved_tumor_aliquot")
        ].copy()
        case_ids = sorted(file_mapping["case_id"].dropna().unique())
        selected_rows: list[pd.DataFrame] = []
        report_rows = []

        for case_id in case_ids:
            case_all = file_mapping.loc[file_mapping["case_id"].eq(case_id)]
            case = candidates.loc[candidates["case_id"].eq(case_id)]
            available = {
                modality: sorted(
                    case.loc[case["data_type"].eq(modality), "sample_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                for modality in modalities
            }
            all_present_modalities = set(case_all["data_type"])
            resolved_modalities = set(case["data_type"])
            missing_modalities = [
                modality
                for modality in modalities
                if modality not in all_present_modalities
            ]
            unresolved_modalities = [
                modality
                for modality in modalities
                if modality in all_present_modalities
                and modality not in resolved_modalities
            ]
            selected_sample_id = None

            if missing_modalities:
                status = "missing_modality"
                detail = ",".join(missing_modalities)
            elif unresolved_modalities:
                status = "unresolved_mapping"
                unresolved_states = case_all.loc[
                    case_all["data_type"].isin(unresolved_modalities),
                    "mapping_status",
                ].unique()
                detail = (
                    f"modalities={','.join(unresolved_modalities)}; statuses="
                    f"{','.join(sorted(unresolved_states))}"
                )
            else:
                shared_samples = set(available[modalities[0]])
                for modality in modalities[1:]:
                    shared_samples &= set(available[modality])

                if not shared_samples:
                    status = "specimen_mismatch"
                    detail = "no exact sample barcode shared by all modalities"
                elif len(shared_samples) > 1:
                    status = "ambiguous_shared_samples"
                    detail = ",".join(sorted(shared_samples))
                else:
                    selected_sample_id = next(iter(shared_samples))
                    chosen = case.loc[case["sample_id"].eq(selected_sample_id)]
                    counts = chosen["data_type"].value_counts()
                    duplicate_modalities = [
                        modality
                        for modality in modalities
                        if int(counts.get(modality, 0)) != 1
                    ]
                    if duplicate_modalities:
                        status = "duplicate_files"
                        detail = ",".join(duplicate_modalities)
                    else:
                        status = "selected"
                        detail = "exact sample barcode shared by all modalities"
                        selected_rows.append(chosen)

            report_rows.append(
                {
                    "case_id": case_id,
                    "project": (
                        case_all["project"].dropna().iloc[0]
                        if "project" in case_all
                        and not case_all["project"].dropna().empty
                        else None
                    ),
                    "status": status,
                    "selected_sample_id": selected_sample_id,
                    "detail": detail,
                    "available_samples": json.dumps(available, sort_keys=True),
                }
            )

        selected = (
            pd.concat(selected_rows, ignore_index=True)
            if selected_rows
            else file_mapping.iloc[0:0].copy()
        )
        selected = selected.sort_values(
            ["case_id", "data_type", "file_id"]
        ).reset_index(drop=True)
        report = pd.DataFrame(
            report_rows,
            columns=[
                "case_id",
                "project",
                "status",
                "selected_sample_id",
                "detail",
                "available_samples",
            ],
        )
        report = report.sort_values("case_id").reset_index(drop=True)
        return selected, report

    # ── Manifest alignment ────────────────────────────────────────────────────

    def align_manifests(
        self,
        mode: str = "pipeline",
        full_manifest_dir: str | Path = "data/manifests/full",
        portal_path: Optional[str | Path] = None,
        mapping_path: str | Path = "data/file_to_case.tsv",
        outdir: str | Path = "data/manifests/aligned",
        aligned_cases_path: str | Path = "data/aligned_cases.tsv",
        alignment_report_path: str | Path = "data/alignment_report.tsv",
    ) -> Path:
        """
        Align manifests across data types — keep only cases with all omics.

        Parameters
        ----------
        mode : str
            ``"pipeline"`` (read full manifests + file_to_case.tsv) or
            ``"portal"`` (read a GDC portal manifest, query GDC for metadata).
        full_manifest_dir : str or Path
            Directory containing full per-dtype manifests (pipeline mode).
        portal_path : str or Path, optional
            Path to a GDC portal manifest file (portal mode).
        mapping_path : str or Path
            Path to file_to_case.tsv (pipeline mode).
        outdir : str or Path
            Output directory for aligned manifests.
        aligned_cases_path : str or Path
            Output path for aligned_cases.tsv.

        Returns
        -------
        Path
            Aligned manifest directory.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"=== Aligning manifests (mode: {mode}) ===\n")

        if mode == "pipeline":
            manifests, file_map = self._load_pipeline_data(
                Path(full_manifest_dir), Path(mapping_path)
            )
        elif mode == "portal":
            if portal_path is None:
                raise ValueError("portal_path is required for mode='portal'")
            manifests, file_map = self._load_portal_data(Path(portal_path))
        else:
            raise ValueError(f"mode must be 'pipeline' or 'portal', got {mode!r}")

        # Candidate cases per dtype
        for label in self.data_types:
            df = manifests.get(label)
            if df is None or df.empty:
                raise ValueError(
                    f"No files for dtype '{label}'. "
                    f"{'Check your portal cart.' if mode == 'portal' else 'Run generate_full_manifests first.'}"
                )
            cases = {
                file_map[fid]["case_id"]
                for fid in df["file_id"]
                if fid in file_map and file_map[fid].get("case_id")
            }
            print(f"  {label}: {len(cases)} cases")

        mapping_df = pd.DataFrame(
            [{"file_id": file_id, **info} for file_id, info in file_map.items()]
        )
        selected, alignment_report = self.align_specimens(
            mapping_df,
            list(self.data_types),
        )
        aligned = set(selected["case_id"])
        alignment_report.to_csv(alignment_report_path, sep="\t", index=False)
        print(
            f"\n  → specimen-aligned ({len(self.data_types)}-way): "
            f"{len(aligned)} cases\n"
        )

        # Write aligned manifests
        for label in self.data_types:
            df = manifests[label]
            keep = {
                file_id
                for file_id in selected.loc[selected["data_type"].eq(label), "file_id"]
            }
            filtered = df[df["file_id"].isin(keep)]
            out = outdir / f"manifest_{label}.tsv"
            with open(out, "w") as f:
                f.write("id\tfilename\tmd5\tsize\tstate\n")
                for _, row in filtered.iterrows():
                    f.write(
                        f"{row['file_id']}\t{row['filename']}\t"
                        f"{row.get('md5', '')}\t{row.get('size', 0)}\t{row.get('state', '')}\n"
                    )
            print(f"  {label}: {len(filtered)} files → {out}")

        # aligned_cases.tsv
        aligned_df = (
            selected[["case_id", "project", "sample_id"]]
            .drop_duplicates()
            .rename(columns={"case_id": "submitter_id"})
            .sort_values("submitter_id")
        )
        aligned_df.to_csv(aligned_cases_path, sep="\t", index=False)
        print(f"\nSaved: {aligned_cases_path}")
        print(f"Saved: {alignment_report_path}")
        print(aligned_df["project"].value_counts().to_string())
        return outdir

    def _load_pipeline_data(
        self, full_dir: Path, mapping_path: Path
    ) -> tuple[dict, dict]:
        """Load full manifests + file_to_case mapping."""
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"{mapping_path} not found. Run generate_full_manifests() first."
            )
        mapping_df = pd.read_csv(mapping_path, sep="\t")
        if "data_type" not in mapping_df.columns and "dtype" in mapping_df.columns:
            mapping_df = mapping_df.rename(columns={"dtype": "data_type"})
        file_map = mapping_df.set_index("file_id").to_dict(orient="index")
        manifests = {}
        for label in self.data_types:
            path = full_dir / f"manifest_{label}.tsv"
            if not path.exists():
                raise FileNotFoundError(f"Full manifest not found: {path}")
            manifests[label] = pd.read_csv(path, sep="\t").rename(
                columns={"id": "file_id"}
            )
        return manifests, file_map

    def _load_portal_data(self, portal_path: Path) -> tuple[dict, dict]:
        """Load portal manifest, query GDC for metadata, classify by dtype."""
        dtype_to_label = {
            dt["data_type"]: label for label, dt in self.data_types.items()
        }

        portal_df = pd.read_csv(portal_path, sep="\t")
        file_ids = portal_df["id"].tolist()
        print(f"Portal manifest: {len(file_ids)} files\n")

        missing = self._preview_portal_dtypes(portal_df, dtype_to_label)
        if missing:
            raise SystemExit(
                f"Aborted: add {missing} to your portal cart and re-download the manifest."
            )

        print(f"\nQuerying GDC metadata ({len(file_ids)} files)...")
        meta: dict[str, dict] = {}
        for i in tqdm(
            range(0, len(file_ids), BATCH_SIZE), desc="Fetching", unit="batch"
        ):
            batch = file_ids[i : i + BATCH_SIZE]
            for hit in self._batch_query_metadata(batch):
                fid = hit["file_id"]
                biospecimen = _extract_biospecimen_metadata(hit)
                meta[fid] = {
                    "filename": hit.get("file_name", ""),
                    "md5": hit.get("md5sum", ""),
                    "size": hit.get("file_size", 0),
                    "state": hit.get("state", ""),
                    "data_type": hit.get("data_type", ""),
                    **biospecimen,
                    "workflow_type": (hit.get("analysis") or {}).get("workflow_type"),
                }

        manifests_rows: dict[str, list] = {label: [] for label in self.data_types}
        file_map: dict[str, dict] = {}

        for fid in file_ids:
            m = meta.get(fid)
            if not m:
                continue
            label = dtype_to_label.get(m["data_type"])
            if not label:
                continue
            manifests_rows[label].append(
                {
                    "file_id": fid,
                    "filename": m["filename"],
                    "md5": m["md5"],
                    "size": m["size"],
                    "state": m["state"],
                }
            )
            file_map[fid] = {
                "data_type": label,
                **{
                    key: value
                    for key, value in m.items()
                    if key not in {"filename", "md5", "size", "state", "data_type"}
                },
            }

        manifests = {
            label: pd.DataFrame(rows) for label, rows in manifests_rows.items()
        }
        return manifests, file_map

    def _preview_portal_dtypes(
        self, portal_df: pd.DataFrame, dtype_to_label: dict
    ) -> list[str]:
        """Quick filename-based preview; returns list of missing dtype labels."""
        patterns = {
            "aliquot_ensemble_masked": "mutation",
            "ascat3.gene_level": "cnv_gene",
            "ascat3.allelic_specific": "cnv_seg",
            "sesame.level3betas": "methylation",
            "star_gene_counts": "expression",
        }
        counts: dict[str, int] = {}
        for fname in portal_df["filename"]:
            for pat, label in patterns.items():
                if pat in fname:
                    counts[label] = counts.get(label, 0) + 1
                    break

        print("Portal manifest contents (by filename pattern):")
        for label, n in sorted(counts.items()):
            mark = "" if label in dtype_to_label.values() else " ← not in config"
            print(f"  {label}: {n}{mark}")

        missing = [label for label in dtype_to_label.values() if label not in counts]
        if missing:
            print(f"\n  WARNING: missing data types: {missing}")
            print("  Add them to your portal cart before continuing.")
        return missing

    # ── Clinical data ─────────────────────────────────────────────────────────

    def fetch_clinical_table(self, case_ids: list[str]) -> pd.DataFrame:
        """
        Fetch structured clinical data from GDC Cases API.

        Parameters
        ----------
        case_ids : list of str
            Case submitter_ids.

        Returns
        -------
        pd.DataFrame
            Clinical table indexed by case_id.
        """
        fields = [
            "submitter_id",
            "project.project_id",
            "demographic.gender",
            "demographic.vital_status",
            "demographic.days_to_death",
            "demographic.age_at_index",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.ajcc_pathologic_t",
            "diagnoses.ajcc_pathologic_n",
            "diagnoses.ajcc_pathologic_m",
            "diagnoses.primary_diagnosis",
            "diagnoses.morphology",
            "diagnoses.tissue_or_organ_of_origin",
            "diagnoses.days_to_last_follow_up",
            "exposures.tobacco_smoking_status",
            "exposures.pack_years_smoked",
        ]
        all_hits = []
        for i in range(0, len(case_ids), 200):
            batch = case_ids[i : i + 200]
            filters = {"op": "in", "content": {"field": "submitter_id", "value": batch}}
            payload = {
                "filters": json.dumps(filters),
                "fields": ",".join(fields),
                "size": 500,
                "format": "json",
            }
            r = requests.post(GDC_CASES_ENDPOINT, json=payload, timeout=60)
            r.raise_for_status()
            all_hits.extend(r.json()["data"]["hits"])

        rows = []
        for h in all_hits:
            demo = h.get("demographic", {}) or {}
            diag = (h.get("diagnoses") or [{}])[0]
            exp = (h.get("exposures") or [{}])[0]
            rows.append(
                {
                    "case_id": h.get("submitter_id"),
                    "project": (h.get("project") or {}).get("project_id"),
                    "gender": demo.get("gender"),
                    "age": demo.get("age_at_index"),
                    "vital_status": demo.get("vital_status"),
                    "days_to_death": demo.get("days_to_death"),
                    "primary_diagnosis": diag.get("primary_diagnosis"),
                    "morphology": diag.get("morphology"),
                    "stage": diag.get("ajcc_pathologic_stage"),
                    "T": diag.get("ajcc_pathologic_t"),
                    "N": diag.get("ajcc_pathologic_n"),
                    "M": diag.get("ajcc_pathologic_m"),
                    "tissue_origin": diag.get("tissue_or_organ_of_origin"),
                    "days_to_last_followup": diag.get("days_to_last_follow_up"),
                    "smoking_status": exp.get("tobacco_smoking_status"),
                    "pack_years": exp.get("pack_years_smoked"),
                }
            )

        return pd.DataFrame(rows).set_index("case_id").sort_index()

    # ── Download ──────────────────────────────────────────────────────────────

    def get_file_metadata(
        self,
        project_id: str,
        data_type_key: str,
        case_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Get file metadata for a project + data type, optionally filtered by cases."""
        cfg = self.data_types[data_type_key]
        hits = self._query_files(
            project_id, cfg["data_type"], cfg.get("workflow_type"), case_ids=case_ids
        )
        rows = []
        for h in hits:
            case_id = h["cases"][0]["submitter_id"] if h.get("cases") else None
            rows.append(
                {
                    "file_id": h["file_id"],
                    "file_name": h["file_name"],
                    "file_size": h["file_size"],
                    "md5sum": h.get("md5sum", ""),
                    "state": h.get("state", ""),
                    "case_id": case_id,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def verify_manifest_downloads(
        manifest_path: str | Path,
        download_dir: str | Path,
    ) -> pd.DataFrame:
        """Verify exact filename, size, and MD5 for every manifest row."""
        manifest_path = Path(manifest_path)
        download_dir = Path(download_dir)
        manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
        id_column = "id" if "id" in manifest.columns else manifest.columns[0]
        filename_column = (
            "filename" if "filename" in manifest.columns else manifest.columns[1]
        )
        records = []
        for _, row in manifest.iterrows():
            file_id = row[id_column]
            filename = row[filename_column]
            path = download_dir / file_id / filename
            expected_md5 = row.get("md5", "").lower()
            expected_size = row.get("size", "")
            observed_size = path.stat().st_size if path.is_file() else None
            observed_md5 = None

            if not path.is_file():
                status = "missing"
            elif expected_size and observed_size != int(expected_size):
                status = "size_mismatch"
            else:
                digest = hashlib.md5()  # noqa: S324 - GDC publishes MD5 manifests
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                observed_md5 = digest.hexdigest()
                status = (
                    "verified"
                    if not expected_md5 or observed_md5 == expected_md5
                    else "checksum_mismatch"
                )
            records.append(
                {
                    "file_id": file_id,
                    "filename": filename,
                    "path": str(path),
                    "expected_size": int(expected_size) if expected_size else None,
                    "observed_size": observed_size,
                    "expected_md5": expected_md5 or None,
                    "observed_md5": observed_md5,
                    "status": status,
                }
            )
        return pd.DataFrame(
            records,
            columns=[
                "file_id",
                "filename",
                "path",
                "expected_size",
                "observed_size",
                "expected_md5",
                "observed_md5",
                "status",
            ],
        )

    @staticmethod
    def _filter_manifest_skip_existing(
        manifest_path: Path, dl_dir: Path
    ) -> tuple[Path, int, int]:
        """Skip only files that pass exact manifest verification."""
        df = pd.read_csv(manifest_path, sep="\t", dtype=str)
        id_col = "id" if "id" in df.columns else df.columns[0]
        verification = GDCClient.verify_manifest_downloads(manifest_path, dl_dir)
        existing = set(
            verification.loc[verification["status"].eq("verified"), "file_id"]
        )

        n_total = len(df)
        df_filtered = df[~df[id_col].isin(existing)]
        n_skipped = n_total - len(df_filtered)

        filtered_path = (
            manifest_path.parent
            / f"{manifest_path.stem}_remaining{manifest_path.suffix}"
        )
        df_filtered.to_csv(filtered_path, sep="\t", index=False)
        return filtered_path, n_total, n_skipped

    def _download_gdc_client(self, manifests: dict[str, Path], outdir: Path):
        for label, manifest_path in manifests.items():
            dl_dir = outdir / label
            dl_dir.mkdir(parents=True, exist_ok=True)

            filtered, n_total, n_skipped = self._filter_manifest_skip_existing(
                manifest_path, dl_dir
            )
            n_remaining = n_total - n_skipped
            if n_skipped:
                print(
                    f"\n  {label}: {n_skipped}/{n_total} already downloaded, {n_remaining} remaining"
                )
            if n_remaining == 0:
                print(f"  Skipping {label}: all done")
                continue

            cmd = [
                self._find_gdc_client(),
                "download",
                "-m",
                str(filtered),
                "-d",
                str(dl_dir),
                "-n",
                str(self.threads),
                "--retry-amount",
                str(self.retries),
            ]
            if self.token:
                if self.token_path is None:
                    raise ValueError(
                        "A token was loaded without a reusable token path."
                    )
                cmd += ["-t", str(self.token_path)]

            print(f"  Downloading {label} ({n_remaining} files)...")
            subprocess.run(cmd, check=True)
            verification = self.verify_manifest_downloads(manifest_path, dl_dir)
            failed = verification.loc[~verification["status"].eq("verified")]
            if not failed.empty:
                counts = failed["status"].value_counts().to_dict()
                raise RuntimeError(
                    f"GDC download verification failed for {label}: {counts}."
                )

    def download(
        self,
        case_ids: list[str],
        project_id: str,
        data_types: Optional[list[str]] = None,
        outdir: str | Path = "data/raw",
        manifest_dir: Optional[str | Path] = None,
    ) -> Path:
        """
        Generate manifests for aligned cases and download via gdc-client.

        Parameters
        ----------
        case_ids : list of str
            Aligned case submitter_ids.
        project_id : str
            GDC project ID.
        data_types : list of str, optional
            Data type keys. Defaults to all.
        outdir : str or Path
            Output base directory for downloaded files.
        manifest_dir : str or Path, optional
            Where to write intermediate manifests. Defaults to ``{outdir}/manifests``.
        """
        keys = data_types or list(self.data_types.keys())
        outdir = Path(outdir)
        mdir = Path(manifest_dir) if manifest_dir else outdir / "manifests"

        manifests = self.generate_manifests(case_ids, project_id, keys, str(mdir))
        self._download_gdc_client(manifests, outdir)

        case_file = outdir / "downloaded_cases.tsv"
        pd.DataFrame({"project": project_id, "submitter_id": case_ids}).to_csv(
            case_file, sep="\t", index=False
        )
        print(f"\nSaved case list → {case_file}")
        return outdir

    def generate_manifests(
        self,
        case_ids: list[str],
        project_id: str,
        data_types: Optional[list[str]] = None,
        outdir: str = "manifests",
    ) -> dict[str, Path]:
        """Generate GDC download manifests for specific aligned cases."""
        keys = data_types or list(self.data_types.keys())
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        manifests = {}
        for key in keys:
            df = self.get_file_metadata(project_id, key, case_ids)
            manifest_path = outdir / f"manifest_{key}.tsv"
            with open(manifest_path, "w") as f:
                f.write("id\tfilename\tmd5\tsize\tstate\n")
                for _, row in df.iterrows():
                    f.write(
                        f"{row['file_id']}\t{row['file_name']}\t{row['md5sum']}\t{row['file_size']}\t{row['state']}\n"
                    )
            total_mb = df["file_size"].sum() / 1e6
            print(f"  {key}: {len(df)} files, {total_mb:.1f} MB → {manifest_path}")
            manifests[key] = manifest_path
        return manifests
