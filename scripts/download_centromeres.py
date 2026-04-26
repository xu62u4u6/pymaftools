"""Download UCSC hg38 centromere coordinates and save to pymaftools/data/.

Usage:
    uv run python scripts/download_centromeres.py [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/centromeres.txt.gz"
_DATA_DIR = Path(__file__).parent.parent / "pymaftools" / "data"
_OUT = _DATA_DIR / "centromeres_hg38.tsv"


def download_centromeres(force: bool = False) -> pd.DataFrame:
    """Download and cache hg38 centromere coordinates.

    Parameters
    ----------
    force:
        Re-download even if the cache file already exists.

    Returns
    -------
    pd.DataFrame
        Columns: chrom, centro_start, centro_end
        One row per chromosome; centromere region spans
        [centro_start, centro_end).
    """
    if _OUT.exists() and not force:
        print(f"[centromeres] Cache exists: {_OUT}. Use --force to re-download.")
        return pd.read_csv(_OUT, sep="\t")

    print(f"[centromeres] Downloading from {_URL} ...")
    raw = pd.read_csv(
        _URL,
        sep="\t",
        header=None,
        names=["bin", "chrom", "chromStart", "chromEnd", "name"],
        compression="gzip",
    )

    # Collapse multiple centromere segments per chromosome into one range
    centro = (
        raw.groupby("chrom")
        .agg(centro_start=("chromStart", "min"), centro_end=("chromEnd", "max"))
        .reset_index()
    )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    centro.to_csv(_OUT, sep="\t", index=False)
    print(f"[centromeres] Saved {len(centro)} chromosomes → {_OUT}")
    return centro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download hg38 centromere coordinates.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()
    download_centromeres(force=args.force)
