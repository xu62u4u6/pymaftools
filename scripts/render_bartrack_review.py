"""Render BarTrack 4-side composition for human-eye QC (CLAUDE.md s4).

Uses the real TCGA lung fixture (62 genes x 958 samples). The stacked TMB frame
is DERIVED FROM THE MATRIX CELLS here (the fixture predates the sample_metadata
tmb_* cache), which is enough to exercise the stacked rendering visually.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

import pymaftools
from pymaftools.core.variant_groups import FUNCTIONAL_GROUP, FUNCTIONAL_ORDER
from pymaftools.plot import style
from pymaftools.plot.ColorManager import ColorManager
from pymaftools.plot.OncoPlot import OncoPlot
from pymaftools.plot.Track import BarTrack

OUT = Path("outputs/bartrack_review")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("pymaftools/data/example_tcga_lung_mutation_grouped.h5")


def functional_tmb(table) -> pd.DataFrame:
    """sample x functional-group counts, derived from matrix cells."""
    mat = pd.DataFrame(table)  # feat x samp, values = Variant_Classification or False
    mapped = mat.apply(
        lambda col: col.map(
            lambda v: FUNCTIONAL_GROUP.get(v) if isinstance(v, str) else None
        )
    )
    tmb = pd.DataFrame(0, index=mat.columns, columns=FUNCTIONAL_ORDER)
    for g in FUNCTIONAL_ORDER:
        tmb[g] = (mapped == g).sum(axis=0)
    return tmb


def four_side_tracks(op, table):
    tmb = functional_tmb(table)
    mat = pd.DataFrame(table)
    mutated_per_gene = (mat != False).sum(axis=1)  # feature-aligned single series
    return (
        op
        # TOP: stacked TMB by functional group, grows up (out)
        .add_track(BarTrack(tmb, side="top", label="TMB",
                            cmap=ColorManager.FUNCTIONAL_CMAP))
        # RIGHT: per-gene mutation frequency, grows right (out)
        .add_track(BarTrack(table.feature_metadata["freq"], side="right",
                            label="freq", color=style.ACCENT))
        # LEFT: per-gene mutated-sample count, grows LEFT (out, mirror)
        .add_track(BarTrack(mutated_per_gene, side="left", label="n samp",
                            color=style.MUTED))
        # BOTTOM: total burden, grows DOWN (out, away from the matrix)
        .add_track(BarTrack(table.sample_metadata["mutations_count"],
                            side="bottom", label="total", grow="out"))
    )


def main():
    table = pymaftools.read_h5(DATA)

    # (1) ungrouped, 40-sample subset (under the sample-label auto-hide limit so
    # names show). Demo the sample_labels override with short aliases.
    small = table.subset(samples=table.sample_metadata.index[:40])
    aliases = [f"P{i:02d}" for i in range(small.shape[1])]
    op = OncoPlot(small, figsize=(16, 9)).main()
    four_side_tracks(op, small).render(sample_labels=aliases)
    op.fig.savefig(OUT / "four_side_ungrouped_40.png", dpi=110, bbox_inches="tight")
    print("saved four_side_ungrouped_40.png")

    # (2) full 958 samples, grouped by subtype -> exercises sectioning + shared scale
    op2 = OncoPlot(table, figsize=(20, 10)).main()
    four_side_tracks(op2, table).group_samples(by="subtype").render()
    op2.fig.savefig(OUT / "four_side_grouped_958.png", dpi=110, bbox_inches="tight")
    print("saved four_side_grouped_958.png")


if __name__ == "__main__":
    main()
