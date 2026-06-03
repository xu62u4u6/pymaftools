"""Demo: group genes by exon size in a grouped oncoplot (real TCGA lung data).

Takes the bundled TCGA lung fixture, keeps recurrently-mutated genes
(freq >= 0.2), looks up each gene's exon size (longest-transcript length) from
Ensembl BioMart via ``PivotTable.add_exon_size``, bins genes into Small / Medium
/ Large, and renders a grouped oncoplot: rows sectioned by size band, columns by
subtype, each subtype carrying its own per-section frequency strip plus an
overall freq bar. Within each band, genes are ordered by subtype enrichment
(``delta_freq`` = LUSC_freq - LUAD_freq).

This separates the large "mutated-because-they're-huge" passenger genes (TTN,
MUC16, SYNE1, ...) from compact recurrently-mutated drivers (TP53, ...).

Needs network (Ensembl BioMart). Run from the project root:
    uv run python scripts/demo_exon_size_grouping.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import pandas as pd

import pymaftools
from pymaftools import OncoPlot

DATA_DIR = Path(__file__).parent.parent / "pymaftools" / "data"
OUT = Path("img") / "real_tcga_lung_grouped_oncoplot_exon_size.png"
FREQ_THRESHOLD = 0.2
# Bands in bp. Labels wrap to two lines so the rotated row-group titles keep the
# kb range yet still fit a short (few-row) band without overflowing into its
# neighbour.
SIZE_BINS = [0, 5000, 20000, 1e12]
SIZE_LABELS = ["Small\n(<5kb)", "Medium\n(5-20kb)", "Large\n(>20kb)"]


def main() -> None:
    table = pymaftools.read_h5(DATA_DIR / "example_tcga_lung_mutation_grouped.h5")
    genes = table.feature_metadata.index[
        table.feature_metadata["freq"] >= FREQ_THRESHOLD
    ].tolist()

    # exon size for just these genes (targeted BioMart fetch, no genome-wide dump)
    sub = table.subset(features=genes).add_exon_size()
    sub.feature_metadata["size_group"] = pd.cut(
        sub.feature_metadata["exon_size"], bins=SIZE_BINS, labels=SIZE_LABELS
    )
    # Subtype enrichment per gene: positive => mutated more often in LUSC.
    sub.feature_metadata["delta_freq"] = (
        sub.feature_metadata["LUSC_freq"] - sub.feature_metadata["LUAD_freq"]
    )
    # Bands contiguous, Small -> Medium -> Large (compact recurrent genes on top,
    # large passenger-prone genes below); within each band, LUSC-enriched on top.
    sub = sub.sort_features(by=["size_group", "delta_freq"], ascending=[True, False])
    # Waterfall samples within each subtype by the Small band (top rows, TP53/CDH10).
    n_small = int((sub.feature_metadata["size_group"] == SIZE_LABELS[0]).sum())
    sub = sub.sort_samples_by_group(
        group_col="subtype", group_order=["LUAD", "LUSC"], top=n_small
    )

    op = (
        OncoPlot(sub, figsize=(15, 9))
        .main(yticklabels=True, linewidths=0)
        .add_freq(side="right", freq_columns=["freq"], annot=True, linewidths=0)
        .add_sample_annotation(
            ["subtype"],
            side="bottom",
            cmap_dict={"subtype": {"LUAD": "#4C78A8", "LUSC": "#F58518"}},
            linewidths=0,
            size=0.6,
        )
        .group_features(by="size_group")
        .group_samples(by="subtype", freq=True, freq_annot=True)
        .render(wspace=0.03, hspace=0.03)  # show_sample_labels auto-off (958 samples)
    )
    OUT.parent.mkdir(exist_ok=True)
    op.save(str(OUT), dpi=110)
    op.close()
    print(f"[exon_size demo] {len(genes)} genes -> {OUT}")


if __name__ == "__main__":
    main()
