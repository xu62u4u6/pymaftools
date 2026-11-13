"""Rebuild the producer-style MAF fixtures in ``tests/data/maf_sources``.

Downloads two small public files and writes slices of them, plus files derived
from them in the GDC and Funcotator layouts. Run from the repository root::

    python scripts/build_maf_source_fixtures.py
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path("tests/data/maf_sources")
CBIO_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
    "public/acc_tcga_pan_can_atlas_2018/data_mutations.txt"
)
MAFTOOLS = "https://raw.githubusercontent.com/PoisonAlien/maftools/master/inst/extdata/"
KEEP = [
    "Hugo_Symbol", "Entrez_Gene_Id", "Center", "NCBI_Build", "Chromosome",
    "Start_Position", "End_Position", "Strand", "Variant_Classification",
    "Variant_Type", "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode", "Matched_Norm_Sample_Barcode", "HGVSc", "HGVSp",
    "HGVSp_Short", "Transcript_ID", "t_depth", "t_ref_count", "t_alt_count",
    "Protein_position", "FILTER",
]  # fmt: skip


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    cbio = pd.read_csv(
        io.BytesIO(fetch(CBIO_URL)), sep="\t", dtype=str, keep_default_na=False
    )
    tp53_ctnnb1 = cbio[cbio["Hugo_Symbol"].isin(["TP53", "CTNNB1"])]
    rest = cbio[~cbio.index.isin(tp53_ctnnb1.index)].sample(80, random_state=0)
    sub = pd.concat([tp53_ctnnb1, rest]).sort_index()
    sub = sub[[column for column in KEEP if column in sub.columns]]
    sub.to_csv(
        OUT / "cbioportal_acc_tcga_pan_can_atlas_2018.maf", sep="\t", index=False
    )

    # GDC layout: comment block, chr-prefixed GRCh38 labels, gzip.
    gdc = sub.copy()
    gdc["Chromosome"] = "chr" + gdc["Chromosome"]
    gdc["NCBI_Build"] = "GRCh38"
    header = (
        "#version gdc-1.0.0\n#filedate 20220101\n#annotation.spec gdc-1.0.0-public\n"
    )
    with gzip.open(OUT / "gdc_style_grch38.maf.gz", "wt") as handle:
        handle.write(header + gdc.to_csv(sep="\t", index=False))

    # Funcotator layout: ## block, Protein_Change only, Oncotator vocabulary.
    func = sub.drop(columns=["HGVSp", "HGVSp_Short", "Protein_position"])
    func["Protein_Change"] = sub["HGVSp_Short"]
    func.loc[func.index[:4], "Variant_Classification"] = [
        "lincRNA", "De_novo_Start_InFrame", "Start_Codon_SNP", "COULD_NOT_DETERMINE",
    ]  # fmt: skip
    (OUT / "funcotator_style.maf").write_text(
        "#version 2.4\n## Funcotator 4.6.0.0 | Date 20240101T000000 | Gencode 34 CANONICAL\n"
        + func.to_csv(sep="\t", index=False)
    )

    laml = (
        gzip.decompress(fetch(MAFTOOLS + "tcga_laml.maf.gz")).decode().splitlines(True)
    )
    with gzip.open(OUT / "maftools_tcga_laml_head.maf.gz", "wt") as handle:
        handle.writelines(laml[:81])
    brca = gzip.decompress(fetch(MAFTOOLS + "brca.maf.gz")).decode().splitlines(True)
    (OUT / "maftools_brca_head.maf").write_text("".join(brca[:61]))


if __name__ == "__main__":
    main()
