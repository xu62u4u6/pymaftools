from __future__ import annotations

from pathlib import Path

from pymaftools.io import (
    GDCClient,
    read_gene_level_cnv,
    read_maf_files,
    read_methylation_betas,
    read_seg_files,
    read_star_counts,
)


def main() -> None:
    sample_root = Path("data/sample")
    manifests = sample_root / "manifests"

    expr = read_star_counts(
        sample_root / "expression",
        manifests / "manifest_expression.tsv",
    )
    print("\n=== Expression ===")
    print(type(expr).__name__, expr.shape)
    print(expr.iloc[:5, :3])

    seg = read_seg_files(
        sample_root / "cnv",
        manifests / "manifest_cnv.tsv",
    )
    print("\n=== CNV Segments ===")
    print(type(seg).__name__, seg.shape)
    print(seg.head())

    cnv_gene = read_gene_level_cnv(
        sample_root / "cnv_gene",
        manifests / "manifest_cnv_gene.tsv",
    )
    print("\n=== Gene-level CNV ===")
    print(type(cnv_gene).__name__, cnv_gene.shape)
    print(cnv_gene.iloc[:5, :3])

    maf = read_maf_files(
        sample_root / "mutation",
        manifests / "manifest_mutation.tsv",
        nonsynonymous_only=True,
    )
    print("\n=== MAF ===")
    print(type(maf).__name__, maf.shape)
    print(maf[["Hugo_Symbol", "Variant_Classification", "sample_ID"]].head())

    meth = read_methylation_betas(
        sample_root / "methylation",
        manifests / "manifest_methylation.tsv",
    )
    print("\n=== Methylation ===")
    print(type(meth).__name__, meth.shape)
    print(meth.iloc[:5, :3])

    # Clinical: fetch via GDC API (the downloaded XML supplements need
    # a dedicated parser; fetch_clinical_table is the simpler approach)
    client = GDCClient()
    case_ids = list(expr.columns)
    clinical = client.fetch_clinical_table(case_ids)
    print("\n=== Clinical ===")
    print(type(clinical).__name__, clinical.shape)
    print(clinical[["gender", "vital_status", "stage"]].head())

    print("\nSummary:")
    print(f"expression -> {type(expr).__name__}")
    print(f"seg        -> {type(seg).__name__}")
    print(f"cnv_gene   -> {type(cnv_gene).__name__}")
    print(f"maf        -> {type(maf).__name__}")
    print(f"methyl     -> {type(meth).__name__}")
    print(f"clinical   -> {type(clinical).__name__}")


if __name__ == "__main__":
    main()
