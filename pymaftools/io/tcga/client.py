"""
TCGA / GDC data access module.

Provides GDCClient for querying, aligning, and downloading multi-omics data
from the Genomic Data Commons (GDC) API.

Examples
--------
>>> from pymaftools.io import GDCClient
>>> client = GDCClient()
>>> aligned = client.align_cases("TCGA-LUAD", ["expression", "mutation", "cnv"])
>>> client.download(aligned, data_types=["expression", "mutation"], outdir="data/raw")
"""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

# Predefined data type configurations
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
    "cnv": {
        "data_type": "Masked Copy Number Segment",
        "label": "Copy Number (Masked Segment)",
    },
    "cnv_gene": {
        "data_type": "Gene Level Copy Number",
        "label": "Copy Number (Gene Level)",
    },
    "methylation": {
        "data_type": "Methylation Beta Value",
        "label": "DNA Methylation (450K Beta)",
    },
    "clinical": {
        "data_type": "Clinical Supplement",
        "label": "Clinical (BCR XML)",
    },
}


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


class GDCClient:
    """
    Client for querying and downloading data from GDC.

    Parameters
    ----------
    token_path : str or Path, optional
        Path to GDC authentication token file. Required for controlled-access
        data but optional for open-access data.

    Examples
    --------
    >>> client = GDCClient(token_path="~/gdc-token.txt")
    >>> aligned = client.align_cases("TCGA-LUAD")
    >>> print(f"Aligned: {len(aligned)} cases")
    >>> client.generate_manifests(aligned, "TCGA-LUAD", outdir="manifests/")
    """

    def __init__(self, token_path: Optional[str] = None):
        self.token = None
        if token_path:
            self.token = Path(token_path).expanduser().read_text().strip()

    def _query_files(
        self,
        project_id: str,
        data_type: str,
        workflow_type: Optional[str] = None,
        case_ids: Optional[list[str]] = None,
        fields: str = "file_id,file_name,file_size,md5sum,state,cases.submitter_id",
        size: int = 5000,
    ) -> list[dict]:
        """Query GDC files endpoint with filters."""
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
        r = requests.post(GDC_FILES_ENDPOINT, json=payload)
        r.raise_for_status()
        return r.json()["data"]["hits"]

    def get_cases(
        self, project_id: str, data_type_key: str
    ) -> set[str]:
        """
        Get all case submitter_ids that have a given data type.

        Parameters
        ----------
        project_id : str
            GDC project ID (e.g., ``"TCGA-LUAD"``).
        data_type_key : str
            Key from ``DATA_TYPE_CONFIGS``
            (``"expression"``, ``"mutation"``, ``"cnv"``, ``"methylation"``, ``"clinical"``).

        Returns
        -------
        set of str
            Set of case submitter_ids.
        """
        config = DATA_TYPE_CONFIGS[data_type_key]
        hits = self._query_files(
            project_id,
            config["data_type"],
            config.get("workflow_type"),
            fields="cases.submitter_id",
        )
        cases = set()
        for hit in hits:
            for c in hit.get("cases", []):
                cases.add(c["submitter_id"])
        return cases

    def align_cases(
        self,
        project_id: str,
        data_types: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Find cases that have all specified data types (set intersection).

        Parameters
        ----------
        project_id : str
            GDC project ID (e.g., ``"TCGA-LUAD"``).
        data_types : list of str, optional
            Data type keys to align. Defaults to all 5:
            expression, mutation, cnv, methylation, clinical.

        Returns
        -------
        list of str
            Sorted list of aligned case submitter_ids.

        Examples
        --------
        >>> client = GDCClient()
        >>> aligned = client.align_cases("TCGA-LUAD")
        >>> len(aligned)
        506
        """
        if data_types is None:
            data_types = list(DATA_TYPE_CONFIGS.keys())

        case_sets = {}
        for dt_key in data_types:
            cases = self.get_cases(project_id, dt_key)
            case_sets[dt_key] = cases
            label = DATA_TYPE_CONFIGS[dt_key]["label"]
            print(f"  {label}: {len(cases)} cases")

        aligned = set.intersection(*case_sets.values())
        print(f"  >>> Aligned ({len(case_sets)}-way): {len(aligned)} cases")
        return sorted(aligned)

    def align_multi_project(
        self,
        project_ids: list[str],
        data_types: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Align cases across multiple projects.

        Parameters
        ----------
        project_ids : list of str
            List of GDC project IDs.
        data_types : list of str, optional
            Data type keys to align.

        Returns
        -------
        dict
            ``{project_id: [aligned_case_ids]}``.
        """
        result = {}
        for proj in project_ids:
            print(f"\n{'='*50}")
            print(f"  {proj}")
            print(f"{'='*50}")
            result[proj] = self.align_cases(proj, data_types)
        total = sum(len(v) for v in result.values())
        print(f"\nTotal aligned: {total} cases")
        return result

    def get_file_metadata(
        self,
        project_id: str,
        data_type_key: str,
        case_ids: list[str],
    ) -> pd.DataFrame:
        """
        Get file metadata for specific cases and data type.

        Parameters
        ----------
        project_id : str
            GDC project ID.
        data_type_key : str
            Data type key.
        case_ids : list of str
            Case submitter_ids to query.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, file_name, file_size, md5sum,
            state, case_id.
        """
        config = DATA_TYPE_CONFIGS[data_type_key]
        hits = self._query_files(
            project_id,
            config["data_type"],
            config.get("workflow_type"),
            case_ids=case_ids,
        )
        rows = []
        for h in hits:
            case_id = h["cases"][0]["submitter_id"] if h.get("cases") else None
            rows.append({
                "file_id": h["file_id"],
                "file_name": h["file_name"],
                "file_size": h["file_size"],
                "md5sum": h.get("md5sum", ""),
                "state": h.get("state", ""),
                "case_id": case_id,
            })
        return pd.DataFrame(rows)

    def generate_manifests(
        self,
        case_ids: list[str],
        project_id: str,
        data_types: Optional[list[str]] = None,
        outdir: str = "manifests",
    ) -> dict[str, Path]:
        """
        Generate GDC download manifests for aligned cases.

        Parameters
        ----------
        case_ids : list of str
            Aligned case submitter_ids.
        project_id : str
            GDC project ID.
        data_types : list of str, optional
            Data type keys. Defaults to all 5.
        outdir : str
            Output directory for manifest files.

        Returns
        -------
        dict
            ``{data_type_key: manifest_path}``.
        """
        if data_types is None:
            data_types = list(DATA_TYPE_CONFIGS.keys())

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        manifests = {}

        for dt_key in data_types:
            df = self.get_file_metadata(project_id, dt_key, case_ids)
            manifest_path = outdir / f"manifest_{dt_key}.tsv"
            with open(manifest_path, "w") as f:
                f.write("id\tfilename\tmd5\tsize\tstate\n")
                for _, row in df.iterrows():
                    f.write(f"{row['file_id']}\t{row['file_name']}\t{row['md5sum']}\t{row['file_size']}\t{row['state']}\n")
            total_mb = df["file_size"].sum() / 1e6
            print(f"  {dt_key}: {len(df)} files, {total_mb:.1f} MB → {manifest_path}")
            manifests[dt_key] = manifest_path

        return manifests

    def download(
        self,
        case_ids: list[str],
        project_id: str,
        data_types: Optional[list[str]] = None,
        outdir: str = "data/raw",
        n_threads: int = 8,
        method: str = "gdc-client",
    ) -> Path:
        """
        Download data for aligned cases.

        Parameters
        ----------
        case_ids : list of str
            Aligned case submitter_ids.
        project_id : str
            GDC project ID.
        data_types : list of str, optional
            Data type keys. Defaults to all 5.
        outdir : str
            Output base directory.
        n_threads : int
            Number of parallel download threads (gdc-client only).
        method : str
            ``"gdc-client"`` (recommended) or ``"api"`` (direct HTTP).

        Returns
        -------
        Path
            Output directory path.
        """
        if data_types is None:
            data_types = list(DATA_TYPE_CONFIGS.keys())

        outdir = Path(outdir)

        # Generate manifests first
        manifest_dir = outdir / "manifests"
        manifests = self.generate_manifests(
            case_ids, project_id, data_types, str(manifest_dir)
        )

        if method == "gdc-client":
            self._download_gdc_client(manifests, outdir, n_threads)
        elif method == "api":
            self._download_api(case_ids, project_id, data_types, outdir)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gdc-client' or 'api'.")

        # Save case list
        case_file = outdir / "downloaded_cases.tsv"
        pd.DataFrame({"project": project_id, "submitter_id": case_ids}).to_csv(
            case_file, sep="\t", index=False
        )
        print(f"\nSaved case list → {case_file}")
        return outdir

    @staticmethod
    def _filter_manifest_skip_existing(
        manifest_path: Path,
        dl_dir: Path,
    ) -> tuple[Path, int, int]:
        """
        Filter a manifest to skip file UUIDs already downloaded.

        gdc-client creates ``{dl_dir}/{file_uuid}/`` directories.  If a UUID
        directory already exists and contains at least one non-log file, we
        consider it downloaded and remove it from the manifest.

        Parameters
        ----------
        manifest_path : Path
            Original manifest TSV.
        dl_dir : Path
            Download target directory.

        Returns
        -------
        tuple of (Path, int, int)
            (filtered_manifest_path, n_total, n_skipped).
        """
        df = pd.read_csv(manifest_path, sep="\t", dtype=str)
        id_col = "id" if "id" in df.columns else df.columns[0]

        existing_uuids = set()
        if dl_dir.exists():
            for child in dl_dir.iterdir():
                if child.is_dir() and child.name != "logs":
                    # Check the uuid dir has at least one data file
                    data_files = [
                        f for f in child.iterdir()
                        if f.is_file() and f.name != "annotations.txt"
                    ]
                    if data_files:
                        existing_uuids.add(child.name)

        n_total = len(df)
        df_filtered = df[~df[id_col].isin(existing_uuids)]
        n_skipped = n_total - len(df_filtered)

        filtered_path = manifest_path.parent / f"{manifest_path.stem}_filtered{manifest_path.suffix}"
        df_filtered.to_csv(filtered_path, sep="\t", index=False)
        return filtered_path, n_total, n_skipped

    def _download_gdc_client(
        self,
        manifests: dict[str, Path],
        outdir: Path,
        n_threads: int,
    ):
        """Download using gdc-client CLI tool, skipping already-downloaded files."""
        for dt_key, manifest_path in manifests.items():
            dl_dir = outdir / dt_key
            dl_dir.mkdir(parents=True, exist_ok=True)

            filtered_manifest, n_total, n_skipped = self._filter_manifest_skip_existing(
                manifest_path, dl_dir,
            )
            n_remaining = n_total - n_skipped

            if n_skipped:
                print(f"\n  {dt_key}: {n_skipped}/{n_total} already downloaded, "
                      f"{n_remaining} remaining")

            if n_remaining == 0:
                print(f"  Skipping {dt_key}: all files already downloaded")
                continue

            cmd = [
                "gdc-client", "download",
                "-m", str(filtered_manifest),
                "-d", str(dl_dir),
                "-n", str(n_threads),
                "--retry-amount", "5",
            ]
            if self.token:
                # Write token to temp file for gdc-client
                token_file = outdir / ".gdc-token"
                token_file.write_text(self.token)
                cmd.extend(["-t", str(token_file)])

            print(f"  Downloading {dt_key} ({n_remaining} files)...")
            subprocess.run(cmd, check=True)
            print(f"  Done: {dt_key}")

    def _download_api(
        self,
        case_ids: list[str],
        project_id: str,
        data_types: list[str],
        outdir: Path,
        batch_size: int = 50,
    ):
        """Download using GDC REST API (for smaller datasets), skipping existing files."""
        headers = {}
        if self.token:
            headers["X-Auth-Token"] = self.token

        for dt_key in data_types:
            dl_dir = outdir / dt_key
            dl_dir.mkdir(parents=True, exist_ok=True)

            # Collect already-downloaded file UUIDs
            existing_uuids = set()
            if dl_dir.exists():
                for child in dl_dir.iterdir():
                    if child.is_dir() and child.name != "logs":
                        if any(f.is_file() for f in child.iterdir()):
                            existing_uuids.add(child.name)

            df = self.get_file_metadata(project_id, dt_key, case_ids)
            df = df[~df["file_id"].isin(existing_uuids)]
            file_ids = df["file_id"].tolist()

            if not file_ids:
                print(f"\n  Skipping {dt_key}: all files already downloaded")
                continue

            print(f"\n  Downloading {dt_key}: {len(file_ids)} files...")

            for i in range(0, len(file_ids), batch_size):
                batch = file_ids[i: i + batch_size]
                payload = {"ids": batch}
                r = requests.post(GDC_DATA_ENDPOINT, json=payload, headers=headers, stream=True)
                r.raise_for_status()

                content_type = r.headers.get("Content-Type", "")
                if "application/x-tar" in content_type or len(batch) > 1:
                    buf = io.BytesIO(r.content)
                    with tarfile.open(fileobj=buf) as tar:
                        for member in tar.getmembers():
                            if member.isfile() and os.path.basename(member.name) != "MANIFEST.txt":
                                f = tar.extractfile(member)
                                if f:
                                    (dl_dir / os.path.basename(member.name)).write_bytes(f.read())
                else:
                    cd = r.headers.get("Content-Disposition", "")
                    fname = cd.split("filename=")[-1].strip('"') if "filename=" in cd else f"{batch[0]}.dat"
                    (dl_dir / fname).write_bytes(r.content)

            print(f"  Done: {dt_key} ({len(file_ids)} files)")

    def fetch_clinical_table(
        self,
        case_ids: list[str],
    ) -> pd.DataFrame:
        """
        Fetch structured clinical data from GDC API.

        This queries the cases endpoint directly for structured fields,
        which is simpler than parsing BCR XML files.

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
        # Batch in groups of 200 to avoid URI too long
        for i in range(0, len(case_ids), 200):
            batch = case_ids[i: i + 200]
            filters = {"op": "in", "content": {"field": "submitter_id", "value": batch}}
            payload = {
                "filters": json.dumps(filters),
                "fields": ",".join(fields),
                "size": 500,
                "format": "json",
            }
            r = requests.post(GDC_CASES_ENDPOINT, json=payload)
            r.raise_for_status()
            all_hits.extend(r.json()["data"]["hits"])

        rows = []
        for h in all_hits:
            demo = h.get("demographic", {}) or {}
            diag = (h.get("diagnoses") or [{}])[0]
            exp = (h.get("exposures") or [{}])[0]
            rows.append({
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
            })

        df = pd.DataFrame(rows).set_index("case_id").sort_index()
        return df
