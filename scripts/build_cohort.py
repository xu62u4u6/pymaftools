"""
Build cohort from raw GDC downloads.

Reads raw data via pymaftools readers, assembles a Cohort object with
aligned samples, and saves as SQLite.

Usage:
    uv run python scripts/build_cohort.py
    uv run python scripts/build_cohort.py --types expression mutation cnv_gene
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pymaftools.core.Cohort import Cohort
from pymaftools.io import (
    GDCClient,
    read_gene_level_cnv,
    read_maf_files,
    read_methylation_betas,
    read_star_counts,
)

RAW_DIR = Path("data/sample")
MANIFEST_DIR = RAW_DIR / "manifests"
PROCESSED_DIR = Path("data/processed")


# ------------------------------------------------------------------ #
#  Builders — one per data type
# ------------------------------------------------------------------ #


def build_expression():
    """Build expression table from STAR-Counts files.

    Data file format (TSV, comment lines start with '#'):
        gene_id          gene_name  gene_type  unstranded  stranded_first  stranded_second  tpm_unstranded  fpkm_unstranded  fpkm_uq_unstranded
        ENSG00000000003.15  TSPAN6  protein_coding  5765  2870  2895  18.4066  13.9424  17.7419

    Manifest format (TSV):
        file_id  filename  data_type  case_id  sample_type
    """
    print("\n=== Expression ===")
    expr = read_star_counts(RAW_DIR / "expression", MANIFEST_DIR / "manifest_expression.tsv")
    print(f"  {expr.shape[0]} genes × {expr.shape[1]} samples")
    return expr


def build_mutation():
    """Build mutation table from MAF files.

    Data file format (TSV, comment lines start with '#', gzipped .maf.gz):
        Hugo_Symbol  Entrez_Gene_Id  Center  NCBI_Build  Chromosome  Start_Position  End_Position  Strand  Variant_Classification  Variant_Type  ...
        TP53         7157            BCM     GRCh38      chr17       7675088         7675088       +       Missense_Mutation       SNP           ...

    Manifest format (TSV):
        file_id  filename  data_type  case_id  sample_type
    """
    print("\n=== Mutation (MAF) ===")
    maf = read_maf_files(RAW_DIR / "mutation", MANIFEST_DIR / "manifest_mutation.tsv")
    pt = maf.to_pivot_table()
    print(f"  {pt.shape[0]} genes × {pt.shape[1]} samples")
    return pt


def build_cnv_gene():
    """Build gene-level CNV table from ASCAT3 files.

    Data file format (TSV, no comment lines):
        gene_id              gene_name  chromosome  start      end        copy_number  min_copy_number  max_copy_number
        ENSG00000000003.15   TSPAN6     chrX        100627108  100639991  2            2                2

    Supports two filename variants:
        *.gene_level_copy_number.v36.tsv
        *.gene_level.copy_number_variation.tsv

    Manifest format (TSV):
        file_id  filename  data_type  case_id  sample_type
    """
    print("\n=== Gene-level CNV ===")
    cnv = read_gene_level_cnv(RAW_DIR / "cnv_gene", MANIFEST_DIR / "manifest_cnv_gene.tsv")
    print(f"  {cnv.shape[0]} genes × {cnv.shape[1]} samples")
    return cnv


def build_methylation():
    """Build methylation table from SeSAMe beta value files.

    Data file format (TSV, no header, two columns):
        cg00000029\t0.643598
        cg00000165\t0.384754

    Column semantics: probe_id (Illumina CpG probe ID), beta (methylation beta value 0-1).

    Manifest format (TSV):
        file_id  filename  data_type  case_id  sample_type
    """
    print("\n=== Methylation ===")
    meth = read_methylation_betas(RAW_DIR / "methylation", MANIFEST_DIR / "manifest_methylation.tsv")
    print(f"  {meth.shape[0]} probes × {meth.shape[1]} samples")
    return meth


BUILDERS = {
    "expression": build_expression,
    "mutation": build_mutation,
    "cnv_gene": build_cnv_gene,
    "methylation": build_methylation,
}


def fetch_clinical(case_ids: list[str]) -> pd.DataFrame:
    """Fetch clinical table from GDC API for the given cases."""
    print("\n=== Clinical ===")
    client = GDCClient()
    clinical = client.fetch_clinical_table(case_ids)
    print(f"  {clinical.shape[0]} patients × {clinical.shape[1]} fields")
    return clinical


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(description="Build TCGA lung cohort from raw downloads")
    parser.add_argument(
        "--types", nargs="+", default=list(BUILDERS.keys()),
        help="Data types to process (default: all)",
    )
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR / "tcga_lung.h5"))
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Build each data type
    tables: dict[str, object] = {}
    for dtype in args.types:
        if dtype not in BUILDERS:
            print(f"Unknown type: {dtype}, skipping")
            continue
        try:
            tables[dtype] = BUILDERS[dtype]()
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")

    if not tables:
        print("No tables loaded, exiting.")
        return

    # Find aligned cases (intersection across all loaded tables)
    sample_sets = [set(t.columns) for t in tables.values()]
    aligned = set.intersection(*sample_sets)
    print(f"\nAligned cases across {len(tables)} data types: {len(aligned)}")

    # Fetch clinical for aligned cases
    clinical = fetch_clinical(sorted(aligned))

    # Assemble cohort — add clinical first to establish sample_IDs,
    # then add_table auto-subsets each table to those IDs
    cohort = Cohort(name="TCGA-Lung")
    cohort.add_sample_metadata(clinical)

    for name, table in tables.items():
        cohort.add_table(table, name)

    # Save
    cohort.to_hdf5(args.output)
    print(f"\nCohort saved: {args.output}")
    print(f"  Tables: {list(cohort.tables.keys())}")
    for name, t in cohort.tables.items():
        print(f"    {name}: {t.shape}")


if __name__ == "__main__":
    main()
