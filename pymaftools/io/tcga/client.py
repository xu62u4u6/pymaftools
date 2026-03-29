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

import gzip
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"

# Default data type configs (used when no config.toml is provided)
DATA_TYPE_CONFIGS = {
    "expression": {
        "data_type":    "Gene Expression Quantification",
        "workflow_type": "STAR - Counts",
        "label":        "Gene Expression (STAR Counts)",
    },
    "mutation": {
        "data_type": "Masked Somatic Mutation",
        "label":     "Somatic Mutation (MAF)",
    },
    "cnv_seg": {
        "data_type":    "Allele-specific Copy Number Segment",
        "workflow_type": "ASCAT3",
        "label":        "Copy Number Segment (ASCAT3)",
    },
    "cnv_gene": {
        "data_type":    "Gene Level Copy Number",
        "workflow_type": "ASCAT3",
        "label":        "Copy Number Gene Level (ASCAT3)",
    },
    "methylation": {
        "data_type":    "Methylation Beta Value",
        "workflow_type": "SeSAMe Methylation Beta Estimation",
        "label":        "DNA Methylation (SeSAMe)",
    },
}

BATCH_SIZE  = 200
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
        "project":     parts[0],
        "tss":         parts[1],
        "participant": parts[2],
        "case_id":     "-".join(parts[:3]),
    }
    if len(parts) > 3:
        result["sample_type"] = int(parts[3][:2])
        result["vial"]        = parts[3][2:] if len(parts[3]) > 2 else None
        result["is_tumor"]    = result["sample_type"] < 10
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
        if token_path:
            p = Path(token_path).expanduser()
            if p.exists():
                self.token = p.read_text().strip()

        self.data_types      = data_types or DATA_TYPE_CONFIGS
        self.projects        = projects or []
        self.gdc_client_path = str(gdc_client_path) if gdc_client_path else None
        self.threads         = threads
        self.retries         = retries

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
            token_path      = dl.get("token"),
            data_types      = cfg.get("data_types", DATA_TYPE_CONFIGS),
            projects        = cfg.get("projects", []),
            gdc_client_path = dl.get("gdc_client"),
            threads         = dl.get("threads", 8),
            retries         = dl.get("retries", 5),
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
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "=", "content": {"field": "data_type", "value": data_type}},
            ],
        }
        if workflow_type:
            filters["content"].append(
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}}
            )
        if case_ids:
            filters["content"].append(
                {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}}
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
        """Batch query file_id → case_id, sample_type, project."""
        fields = (
            "file_id,file_name,file_size,md5sum,state,data_type,"
            "analysis.workflow_type,"
            "cases.submitter_id,cases.samples.submitter_id,"
            "cases.samples.sample_type,cases.project.project_id"
        )
        results = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                filters = {"op": "in", "content": {"field": "file_id", "value": file_ids}}
                payload = {
                    "filters": json.dumps(filters),
                    "fields": fields,
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
                time.sleep(2 ** attempt)
        return results

    @staticmethod
    def _write_manifest(hits: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("id\tfilename\tmd5\tsize\tstate\n")
            for h in hits:
                f.write(
                    f"{h['file_id']}\t{h['file_name']}\t"
                    f"{h.get('md5sum','')}\t{h['file_size']}\t{h.get('state','')}\n"
                )

    def _find_gdc_client(self) -> str:
        if self.gdc_client_path and Path(self.gdc_client_path).exists():
            return self.gdc_client_path
        found = shutil.which("gdc-client")
        if found:
            return found
        for candidate in [Path("tools/gdc-client"), Path.home() / ".local/bin/gdc-client"]:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "gdc-client not found. Set gdc_client_path or add it to PATH."
        )

    # ── Case alignment ────────────────────────────────────────────────────────

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
        projects  = projects or self.projects
        outdir    = Path(outdir)
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
                records.append({"file_id": h["file_id"], "filename": h["file_name"], "dtype": label})

            total_gb = sum(h["file_size"] for h in label_hits) / 1e9
            print(f"  → manifest_{label}.tsv ({len(label_hits)} files, {total_gb:.1f} GB)\n")

        mapping_df = self.build_file_mapping(records)
        mapping_df.to_csv(mapping_path, sep="\t", index=False)
        print(f"Saved file mapping: {mapping_path} ({len(mapping_df)} rows)")
        return outdir

    def build_file_mapping(self, records: list[dict]) -> pd.DataFrame:
        """
        Build file_id → case_id / sample_type / project mapping via GDC API.

        Parameters
        ----------
        records : list of dict
            Each dict must have ``file_id``, ``filename``, ``dtype``.

        Returns
        -------
        pd.DataFrame
            Columns: file_id, filename, dtype, case_id, sample_type, project.
        """
        unique_ids = list({r["file_id"] for r in records})
        print(f"\nBuilding file mapping ({len(unique_ids)} files)...")

        mapping: dict[str, tuple] = {}
        for i in tqdm(range(0, len(unique_ids), BATCH_SIZE), desc="Querying GDC", unit="batch"):
            batch = unique_ids[i: i + BATCH_SIZE]
            for hit in self._batch_query_metadata(batch):
                fid        = hit["file_id"]
                case_id    = sample_type = project = None
                if hit.get("cases"):
                    c           = hit["cases"][0]
                    case_id     = c.get("submitter_id")
                    project     = (c.get("project") or {}).get("project_id")
                    samples     = c.get("samples", [])
                    preferred   = _pick_preferred_sample(samples)
                    if preferred:
                        sample_type = preferred.get("sample_type")
                mapping[fid] = (case_id, sample_type, project)

        df = pd.DataFrame(records)
        df["case_id"]     = df["file_id"].map(lambda x: mapping.get(x, (None, None, None))[0])
        df["sample_type"] = df["file_id"].map(lambda x: mapping.get(x, (None, None, None))[1])
        df["project"]     = df["file_id"].map(lambda x: mapping.get(x, (None, None, None))[2])
        return df

    # ── Manifest alignment ────────────────────────────────────────────────────

    def align_manifests(
        self,
        mode: str = "pipeline",
        full_manifest_dir: str | Path = "data/manifests/full",
        portal_path: Optional[str | Path] = None,
        mapping_path: str | Path = "data/file_to_case.tsv",
        outdir: str | Path = "data/manifests/aligned",
        aligned_cases_path: str | Path = "data/aligned_cases.tsv",
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

        # Cases per dtype
        dtype_cases: dict[str, set] = {}
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
            dtype_cases[label] = cases
            print(f"  {label}: {len(cases)} cases")

        aligned = set.intersection(*dtype_cases.values())
        print(f"\n  → aligned ({len(self.data_types)}-way): {len(aligned)} cases\n")

        # Write aligned manifests
        for label in self.data_types:
            df = manifests[label]
            keep = {
                fid for fid, info in file_map.items()
                if info.get("dtype") == label and info.get("case_id") in aligned
            }
            filtered = df[df["file_id"].isin(keep)]
            out = outdir / f"manifest_{label}.tsv"
            with open(out, "w") as f:
                f.write("id\tfilename\tmd5\tsize\tstate\n")
                for _, row in filtered.iterrows():
                    f.write(
                        f"{row['file_id']}\t{row['filename']}\t"
                        f"{row.get('md5','')}\t{row.get('size',0)}\t{row.get('state','')}\n"
                    )
            print(f"  {label}: {len(filtered)} files → {out}")

        # aligned_cases.tsv
        project_map = {
            info["case_id"]: info["project"]
            for info in file_map.values()
            if info.get("case_id") in aligned and info.get("project")
        }
        aligned_df = pd.DataFrame([
            {"submitter_id": c, "project": project_map.get(c, "unknown")}
            for c in sorted(aligned)
        ])
        aligned_df.to_csv(aligned_cases_path, sep="\t", index=False)
        print(f"\nSaved: {aligned_cases_path}")
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
        file_map = {
            row["file_id"]: {
                "dtype":       row["dtype"],
                "case_id":     row["case_id"],
                "sample_type": row["sample_type"],
                "project":     row.get("project"),
            }
            for _, row in mapping_df.iterrows()
        }
        manifests = {}
        for label in self.data_types:
            path = full_dir / f"manifest_{label}.tsv"
            if not path.exists():
                raise FileNotFoundError(f"Full manifest not found: {path}")
            manifests[label] = (
                pd.read_csv(path, sep="\t").rename(columns={"id": "file_id"})
            )
        return manifests, file_map

    def _load_portal_data(
        self, portal_path: Path
    ) -> tuple[dict, dict]:
        """Load portal manifest, query GDC for metadata, classify by dtype."""
        dtype_to_label = {dt["data_type"]: label for label, dt in self.data_types.items()}

        portal_df = pd.read_csv(portal_path, sep="\t")
        file_ids  = portal_df["id"].tolist()
        print(f"Portal manifest: {len(file_ids)} files\n")

        missing = self._preview_portal_dtypes(portal_df, dtype_to_label)
        if missing:
            raise SystemExit(
                f"Aborted: add {missing} to your portal cart and re-download the manifest."
            )

        print(f"\nQuerying GDC metadata ({len(file_ids)} files)...")
        meta: dict[str, dict] = {}
        for i in tqdm(range(0, len(file_ids), BATCH_SIZE), desc="Fetching", unit="batch"):
            batch = file_ids[i: i + BATCH_SIZE]
            for hit in self._batch_query_metadata(batch):
                fid         = hit["file_id"]
                case_id     = sample_type = project = None
                if hit.get("cases"):
                    c           = hit["cases"][0]
                    case_id     = c.get("submitter_id")
                    project     = (c.get("project") or {}).get("project_id")
                    samples     = c.get("samples", [])
                    preferred   = _pick_preferred_sample(samples)
                    if preferred:
                        sample_type = preferred.get("sample_type")
                meta[fid] = {
                    "filename":    hit.get("file_name", ""),
                    "md5":         hit.get("md5sum", ""),
                    "size":        hit.get("file_size", 0),
                    "state":       hit.get("state", ""),
                    "data_type":   hit.get("data_type", ""),
                    "case_id":     case_id,
                    "sample_type": sample_type,
                    "project":     project,
                }

        manifests_rows: dict[str, list] = {label: [] for label in self.data_types}
        file_map: dict[str, dict] = {}

        for fid in file_ids:
            m     = meta.get(fid)
            if not m:
                continue
            label = dtype_to_label.get(m["data_type"])
            if not label:
                continue
            manifests_rows[label].append({
                "file_id":  fid,
                "filename": m["filename"],
                "md5":      m["md5"],
                "size":     m["size"],
                "state":    m["state"],
            })
            file_map[fid] = {
                "dtype":       label,
                "case_id":     m["case_id"],
                "sample_type": m["sample_type"],
                "project":     m["project"],
            }

        manifests = {label: pd.DataFrame(rows) for label, rows in manifests_rows.items()}
        return manifests, file_map

    def _preview_portal_dtypes(
        self, portal_df: pd.DataFrame, dtype_to_label: dict
    ) -> list[str]:
        """Quick filename-based preview; returns list of missing dtype labels."""
        patterns = {
            "aliquot_ensemble_masked": "mutation",
            "ascat3.gene_level":       "cnv_gene",
            "ascat3.allelic_specific": "cnv_seg",
            "sesame.level3betas":      "methylation",
            "star_gene_counts":        "expression",
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

        missing = [l for l in dtype_to_label.values() if l not in counts]
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
            "submitter_id", "project.project_id",
            "demographic.gender", "demographic.vital_status",
            "demographic.days_to_death", "demographic.age_at_index",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n",
            "diagnoses.ajcc_pathologic_m",
            "diagnoses.primary_diagnosis", "diagnoses.morphology",
            "diagnoses.tissue_or_organ_of_origin",
            "diagnoses.days_to_last_follow_up",
            "exposures.tobacco_smoking_status",
            "exposures.pack_years_smoked",
        ]
        all_hits = []
        for i in range(0, len(case_ids), 200):
            batch = case_ids[i: i + 200]
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
            exp  = (h.get("exposures")  or [{}])[0]
            rows.append({
                "case_id":             h.get("submitter_id"),
                "project":             (h.get("project") or {}).get("project_id"),
                "gender":              demo.get("gender"),
                "age":                 demo.get("age_at_index"),
                "vital_status":        demo.get("vital_status"),
                "days_to_death":       demo.get("days_to_death"),
                "primary_diagnosis":   diag.get("primary_diagnosis"),
                "morphology":          diag.get("morphology"),
                "stage":               diag.get("ajcc_pathologic_stage"),
                "T":                   diag.get("ajcc_pathologic_t"),
                "N":                   diag.get("ajcc_pathologic_n"),
                "M":                   diag.get("ajcc_pathologic_m"),
                "tissue_origin":       diag.get("tissue_or_organ_of_origin"),
                "days_to_last_followup": diag.get("days_to_last_follow_up"),
                "smoking_status":      exp.get("tobacco_smoking_status"),
                "pack_years":          exp.get("pack_years_smoked"),
            })

        return pd.DataFrame(rows).set_index("case_id").sort_index()

    # ── Download ──────────────────────────────────────────────────────────────

    def get_file_metadata(
        self,
        project_id: str,
        data_type_key: str,
        case_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Get file metadata for a project + data type, optionally filtered by cases."""
        cfg  = self.data_types[data_type_key]
        hits = self._query_files(
            project_id, cfg["data_type"], cfg.get("workflow_type"), case_ids=case_ids
        )
        rows = []
        for h in hits:
            case_id = h["cases"][0]["submitter_id"] if h.get("cases") else None
            rows.append({
                "file_id":   h["file_id"],
                "file_name": h["file_name"],
                "file_size": h["file_size"],
                "md5sum":    h.get("md5sum", ""),
                "state":     h.get("state", ""),
                "case_id":   case_id,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _filter_manifest_skip_existing(
        manifest_path: Path, dl_dir: Path
    ) -> tuple[Path, int, int]:
        """Filter manifest to skip already-downloaded file UUIDs."""
        df     = pd.read_csv(manifest_path, sep="\t", dtype=str)
        id_col = "id" if "id" in df.columns else df.columns[0]

        existing = set()
        if dl_dir.exists():
            for child in dl_dir.iterdir():
                if child.is_dir() and child.name != "logs":
                    data_files = [
                        f for f in child.iterdir()
                        if f.is_file() and f.name != "annotations.txt"
                    ]
                    if data_files:
                        existing.add(child.name)

        n_total    = len(df)
        df_filtered = df[~df[id_col].isin(existing)]
        n_skipped  = n_total - len(df_filtered)

        filtered_path = manifest_path.parent / f"{manifest_path.stem}_remaining{manifest_path.suffix}"
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
                print(f"\n  {label}: {n_skipped}/{n_total} already downloaded, {n_remaining} remaining")
            if n_remaining == 0:
                print(f"  Skipping {label}: all done")
                continue

            cmd = [
                self._find_gdc_client(), "download",
                "-m", str(filtered),
                "-d", str(dl_dir),
                "-n", str(self.threads),
                "--retry-amount", str(self.retries),
            ]
            if self.token:
                token_file = outdir / ".gdc-token"
                token_file.write_text(self.token)
                cmd += ["-t", str(token_file)]

            print(f"  Downloading {label} ({n_remaining} files)...")
            subprocess.run(cmd, check=True)

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
        keys     = data_types or list(self.data_types.keys())
        outdir   = Path(outdir)
        mdir     = Path(manifest_dir) if manifest_dir else outdir / "manifests"

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
        keys   = data_types or list(self.data_types.keys())
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        manifests = {}
        for key in keys:
            df           = self.get_file_metadata(project_id, key, case_ids)
            manifest_path = outdir / f"manifest_{key}.tsv"
            with open(manifest_path, "w") as f:
                f.write("id\tfilename\tmd5\tsize\tstate\n")
                for _, row in df.iterrows():
                    f.write(f"{row['file_id']}\t{row['file_name']}\t{row['md5sum']}\t{row['file_size']}\t{row['state']}\n")
            total_mb = df["file_size"].sum() / 1e6
            print(f"  {key}: {len(df)} files, {total_mb:.1f} MB → {manifest_path}")
            manifests[key] = manifest_path
        return manifests
