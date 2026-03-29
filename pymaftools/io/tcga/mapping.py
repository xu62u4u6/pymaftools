"""
TCGA file-to-case mapping utilities.

Loads a pre-built mapping table (file_to_case.tsv) and resolves
files on disk to their associated case_id and sample metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_file_mapping(mapping_path: str | Path) -> pd.DataFrame:
    """
    Load file_to_case.tsv mapping table.

    Expected columns: file_id, filename, data_type, case_id, sample_type.

    Parameters
    ----------
    mapping_path : str or Path
        Path to the TSV mapping file.

    Returns
    -------
    pd.DataFrame
        Mapping table indexed by file_id.
    """
    df = pd.read_csv(mapping_path, sep="\t", dtype=str)
    return df.set_index("file_id")


def resolve_files(
    data_dir: str | Path,
    pattern: str,
    mapping_df: pd.DataFrame,
) -> list[dict]:
    """
    Scan a download directory and resolve each file to its case metadata.

    Supports two directory layouts:
    - gdc-client: ``{data_dir}/{file_uuid}/{filename}``
    - flat: ``{data_dir}/{filename}``

    Parameters
    ----------
    data_dir : str or Path
        Base directory containing downloaded files.
    pattern : str
        Glob pattern for the data files.
    mapping_df : pd.DataFrame
        Mapping table from :func:`load_file_mapping`, indexed by file_id.

    Returns
    -------
    list of dict
        Each dict has keys: case_id, sample_type, data_type, file_id, filepath.
        Sorted by case_id.
    """
    data_dir = Path(data_dir)

    # Build filename → file_id lookup for flat layout
    fname_to_fid = {}
    for fid, row in mapping_df.iterrows():
        fname_to_fid[row["filename"]] = fid

    results = []
    for filepath in data_dir.rglob(pattern):
        parent = filepath.parent.name
        if parent == "logs":
            continue

        # Determine file_id
        if parent == data_dir.name:
            # Flat layout: match by filename
            file_id = fname_to_fid.get(filepath.name)
        else:
            # gdc-client layout: parent dir is UUID
            file_id = parent if parent in mapping_df.index else fname_to_fid.get(filepath.name)

        if file_id is None or file_id not in mapping_df.index:
            continue

        row = mapping_df.loc[file_id]
        results.append({
            "case_id": row["case_id"],
            "sample_type": row.get("sample_type"),
            "data_type": row.get("data_type"),
            "file_id": file_id,
            "filepath": filepath,
        })

    return sorted(results, key=lambda x: (x["case_id"], x["filepath"].name))
