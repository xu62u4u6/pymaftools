"""
Download a small aligned demo dataset (10 TCGA-LUAD cases, 6 data types).

This script is self-contained: it queries GDC for file metadata, downloads
each file via the REST API, and writes manifest TSVs — all into a single
output directory.  Already-downloaded files (identified by UUID subdirectory)
are skipped automatically.

Usage
-----
    python scripts/download_demo_samples.py [--outdir tmp/sample]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"

# 10 aligned TCGA-LUAD cases used across all demo / test data
CASE_IDS = [
    "TCGA-05-4244",
    "TCGA-05-4249",
    "TCGA-05-4250",
    "TCGA-05-4382",
    "TCGA-05-4384",
    "TCGA-05-4389",
    "TCGA-05-4390",
    "TCGA-05-4395",
    "TCGA-05-4396",
    "TCGA-05-4397",
]

PROJECT_ID = "TCGA-LUAD"

# Each entry: (dir_name, GDC data_type, optional workflow_type, file glob for ascat3 filter)
DATA_TYPES = [
    ("expression", "Gene Expression Quantification", "STAR - Counts", None),
    ("mutation", "Masked Somatic Mutation", None, None),
    ("cnv", "Masked Copy Number Segment", None, None),
    ("cnv_gene", "Gene Level Copy Number", None, "ascat3"),
    ("methylation", "Methylation Beta Value", None, None),
    ("clinical", "Clinical Supplement", None, None),
]


def query_files(
    data_type: str,
    workflow_type: str | None,
    filename_filter: str | None,
) -> list[dict]:
    """Query GDC for files matching our 10 cases + data type."""
    filters: dict = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": CASE_IDS}},
            {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT_ID}},
            {"op": "=", "content": {"field": "data_type", "value": data_type}},
        ],
    }
    if workflow_type:
        filters["content"].append(
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}}
        )

    payload = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,md5sum,state,cases.submitter_id",
        "size": 200,
        "format": "json",
    }
    r = requests.post(GDC_FILES, json=payload, timeout=60)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]

    # Optional filename filter (e.g. keep only ascat3, skip absolute_liftover)
    if filename_filter:
        hits = [h for h in hits if filename_filter in h["file_name"]]

    # Deduplicate: keep one file per case (first encountered)
    seen_cases: set[str] = set()
    deduped: list[dict] = []
    for h in hits:
        case_id = h["cases"][0]["submitter_id"] if h.get("cases") else None
        if case_id and case_id not in seen_cases:
            seen_cases.add(case_id)
            deduped.append(h)
    return deduped


def download_file(file_id: str, dest: Path) -> None:
    """Download a single file from GDC data endpoint."""
    r = requests.get(f"{GDC_DATA}/{file_id}", timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download demo TCGA samples")
    parser.add_argument("--outdir", default="data/sample", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    manifest_dir = outdir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for dir_name, data_type, workflow_type, fname_filter in DATA_TYPES:
        print(f"\n{'='*60}")
        print(f"  {dir_name} ({data_type})")
        print(f"{'='*60}")

        dl_dir = outdir / dir_name
        dl_dir.mkdir(parents=True, exist_ok=True)

        hits = query_files(data_type, workflow_type, fname_filter)
        print(f"  Found {len(hits)} files on GDC")

        # Determine which UUIDs are already downloaded
        existing = {
            d.name
            for d in dl_dir.iterdir()
            if d.is_dir()
            and d.name != "logs"
            and any(f.is_file() and f.name != "annotations.txt" for f in d.iterdir())
        } if dl_dir.exists() else set()

        manifest_rows = []
        downloaded = 0
        skipped = 0

        for h in hits:
            uuid = h["file_id"]
            fname = h["file_name"]
            case_id = h["cases"][0]["submitter_id"]

            manifest_rows.append({
                "id": uuid,
                "filename": fname,
                "md5": h.get("md5sum", ""),
                "size": h["file_size"],
                "state": h.get("state", "released"),
            })

            if uuid in existing:
                skipped += 1
                continue

            uuid_dir = dl_dir / uuid
            uuid_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Downloading {fname} (case={case_id})...")
            download_file(uuid, uuid_dir / fname)
            downloaded += 1

        # Write manifest
        manifest_path = manifest_dir / f"manifest_{dir_name}.tsv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)

        print(f"  Downloaded: {downloaded}, Skipped: {skipped}, "
              f"Total: {len(manifest_rows)}")
        print(f"  Manifest → {manifest_path}")

    # Save case list
    cases_file = outdir / "downloaded_cases.tsv"
    pd.DataFrame({"project": PROJECT_ID, "submitter_id": CASE_IDS}).to_csv(
        cases_file, sep="\t", index=False,
    )
    print(f"\nCase list → {cases_file}")
    print("Done!")


if __name__ == "__main__":
    main()
