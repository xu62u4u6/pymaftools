"""Download per-gene canonical-transcript sizes from Ensembl BioMart.

Writes ``pymaftools/data/ensembl_gene_sizes.tsv`` (hugo_symbol, ensembl_gene_id,
transcript_length, cds_length), used by ``PivotTable.add_exon_size()`` and
``pymaftools.utils.geneinfo.get_exon_size`` for size-based gene grouping.

The fetch/caching logic lives in ``geneinfo.load_gene_sizes`` (single source of
truth); this script is just the CLI entry point.

Usage:
    uv run python scripts/download_gene_sizes.py [--force]
"""

from __future__ import annotations

import argparse

from pymaftools.utils.geneinfo import load_gene_sizes, _GENE_SIZE_CACHE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Ensembl canonical-transcript gene sizes."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached."
    )
    args = parser.parse_args()

    if _GENE_SIZE_CACHE.exists() and not args.force:
        print(f"[gene_sizes] Cache exists: {_GENE_SIZE_CACHE}. Use --force to refresh.")
    df = load_gene_sizes(force=args.force)
    print(f"[gene_sizes] {len(df)} genes → {_GENE_SIZE_CACHE}")
