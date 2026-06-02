"""Demo: OncoPlot visualization workflows on synthetic data.

Generates a synthetic mutation cohort (categorical mutation matrix + sample /
feature metadata) and renders a series of oncoplots into ``img/``. Doubles as a
runnable example and as a smoke test for the OncoPlot API.

Run from the project root::

    uv run python scripts/demo_oncoplot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import numpy as np
import pandas as pd

from pymaftools.core.PivotTable import PivotTable
from pymaftools.plot.OncoPlot import OncoPlot

OUT_DIR = Path("img")
RNG = np.random.default_rng(0)

# Variant-classification strings for mutated cells. Wild-type cells are the
# boolean ``False`` (matching MAF.to_pivot_table()'s fillna(False)), so the
# table is an object DataFrame mixing strings and False.
VARIANT_TYPES = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Splice_Site",
]


def make_mutation_table(n_genes: int = 15, n_samples: int = 30) -> PivotTable:
    """Build a synthetic categorical mutation PivotTable with metadata."""
    genes = [f"GENE{i:02d}" for i in range(n_genes)]
    samples = [f"S{i:02d}" for i in range(n_samples)]

    # Long-tail mutation frequency: a few high-frequency "driver" genes and a
    # long tail of rarely-mutated genes (per-gene rate decays geometrically).
    per_gene_rate = 0.7 * (0.78 ** np.arange(n_genes))

    # Object array: False for wild-type, a variant string where mutated.
    matrix = np.full((n_genes, n_samples), False, dtype=object)
    mutated = RNG.random((n_genes, n_samples)) < per_gene_rate[:, None]
    matrix[mutated] = RNG.choice(VARIANT_TYPES, size=int(mutated.sum()))
    table = PivotTable(pd.DataFrame(matrix, index=genes, columns=samples))

    # sample-level metadata (drawn as bottom strips / TMB bar)
    table.sample_metadata["subtype"] = RNG.choice(["LUAD", "LUSC", "ASC"], n_samples)
    table.sample_metadata["sex"] = RNG.choice(["M", "F"], n_samples)
    table.sample_metadata["age"] = RNG.integers(45, 80, n_samples).astype(float)
    table.sample_metadata["TMB"] = RNG.gamma(2.0, 1.0, n_samples)

    # feature-level metadata (gene annotations a user may want as a row-side strip)
    table.feature_metadata["pathway"] = RNG.choice(
        ["RTK-RAS", "TP53", "PI3K"], n_genes
    )
    table.feature_metadata["is_driver"] = RNG.choice([True, False], n_genes)

    # drawn last so adding it doesn't perturb the other columns' RNG draws
    table.sample_metadata["purity"] = RNG.uniform(0.2, 0.95, n_samples)

    return table


def make_cnv_table(n_genes: int = 15, n_samples: int = 30) -> PivotTable:
    """Build a synthetic numeric (log2 ratio) CNV PivotTable."""
    rng = np.random.default_rng(1)  # independent of the mutation-table RNG draws
    genes = [f"GENE{i:02d}" for i in range(n_genes)]
    samples = [f"S{i:02d}" for i in range(n_samples)]
    values = rng.normal(0.0, 0.8, size=(n_genes, n_samples))
    table = PivotTable(pd.DataFrame(values, index=genes, columns=samples))
    table.feature_metadata["freq"] = rng.random(n_genes)
    return table


def prepare(table: PivotTable) -> PivotTable:
    """Standard oncoplot prep: compute freq, sort genes by freq, waterfall-sort samples."""
    return (
        table.add_freq()
        .sort_features(by="freq", ascending=False)
        .sort_samples_by_mutations(top=10)
    )


def demo_default(table: PivotTable) -> None:
    """Figure 1: a basic oncoplot (heatmap + freq + TMB bar + legends).

    The convenience methods (``mutation_heatmap``/``plot_freq``/``plot_bar``)
    now register tracks; a single ``render()`` draws the figure.
    """
    op = OncoPlot(table, figsize=(12, 8))
    op.mutation_heatmap()
    op.plot_freq()
    op.plot_bar()
    op.render()
    op.add_xticklabel()
    op.save(str(OUT_DIR / "demo_oncoplot_default.png"))
    op.close()


def demo_with_metadata(table: PivotTable) -> None:
    """Figure 2: oncoplot with categorical + numeric sample-metadata strips.

    Two numeric columns with different cmaps; their colorbars are collected in
    the legend area (``colorbar="legend"``, the default) so they stay readable
    instead of cramped insets on the thin strips.
    """
    op = OncoPlot(
        table,
        figsize=(12, 9),
        categorical_columns=["subtype", "sex"],
        numeric_columns=["age", "purity"],
    )
    op.mutation_heatmap()
    op.plot_freq()
    op.plot_bar()
    op.plot_categorical_metadata()
    op.plot_numeric_metadata(cmap_dict={"age": "Blues", "purity": "viridis"})
    op.render()
    op.add_xticklabel()
    op.save(str(OUT_DIR / "demo_oncoplot_metadata.png"))
    op.close()


def demo_numeric(table: PivotTable) -> None:
    """Figure 3: numeric (CNV) heatmap with a diverging colormap."""
    op = OncoPlot(table, figsize=(12, 8))
    op.numeric_heatmap(cmap="coolwarm", symmetric=True)
    op.render()
    op.add_xticklabel()
    op.save(str(OUT_DIR / "demo_oncoplot_numeric.png"))
    op.close()


def demo_declarative(table: PivotTable) -> None:
    """Figure 4: a full oncoplot built entirely through the declarative
    ``render()`` path, including a feature-side annotation (``pathway``) that the
    eager layout has no slot for.
    """
    op = (
        OncoPlot(table, figsize=(13, 9))
        .main()
        .add_bar("TMB", side="top")
        .add_freq(side="right")
        .add_feature_annotation(["pathway"], side="right")
        .add_sample_annotation(["subtype", "sex"], side="bottom")
        .add_sample_annotation(["age"], side="bottom")
    )
    op.render()
    op.add_xticklabel()
    op.save(str(OUT_DIR / "demo_oncoplot_declarative.png"))
    op.close()


def demo_grouped(table: PivotTable) -> None:
    """Figure 5: features grouped by pathway (rows) and samples by subtype
    (columns), with separator lines and group titles in both directions."""
    grouped = (
        table.add_freq()
        .sort_features(by="pathway")  # genes contiguous by pathway
        .sort_samples_by_group(
            group_col="subtype", group_order=["LUAD", "ASC", "LUSC"], top=10
        )
    )
    op = (
        OncoPlot(grouped, figsize=(13, 9))
        .main()
        .add_bar("TMB", side="top")
        .add_freq(side="right")
        .group_features(by="pathway")
        .group_samples(by="subtype")
        .render()
    )
    op.save(str(OUT_DIR / "demo_oncoplot_grouped.png"))
    op.close()


def demo_grouped_per_section_freq(table: PivotTable) -> None:
    """Figure 6: per-section frequency bars. With samples grouped by subtype,
    each group gets its OWN freq strip immediately right of its block (LUAD_freq
    next to the LUAD section, ...), plus one overall freq bar at the far right.
    Enabled with ``group_samples(by="subtype", freq=True)``; the per-group freq
    columns come from ``add_freq(groups=...)``.
    """
    subtypes = ["LUAD", "ASC", "LUSC"]
    grouped = table.sort_samples_by_group(
        group_col="subtype", group_order=subtypes, top=10
    )
    # per-group freq columns (LUAD_freq / ASC_freq / LUSC_freq) + overall "freq"
    groups = {
        s: grouped.subset(samples=grouped.sample_metadata.subtype == s)
        for s in subtypes
    }
    grouped = grouped.add_freq(groups=groups).sort_features(
        by="freq", ascending=False
    )
    op = (
        OncoPlot(grouped, figsize=(14, 9))
        .main()
        .add_freq(freq_columns=["freq"], side="right")  # overall, far right
        .add_sample_annotation(["subtype"], side="bottom")
        .group_samples(by="subtype", freq=True)
        .render()
    )
    op.save(str(OUT_DIR / "demo_oncoplot_grouped_freq.png"))
    op.close()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    mutation = prepare(make_mutation_table())
    cnv = make_cnv_table()

    demo_default(mutation)
    demo_with_metadata(mutation)
    demo_numeric(cnv)
    demo_declarative(mutation)
    demo_grouped(make_mutation_table())
    demo_grouped_per_section_freq(make_mutation_table())
    print(f"[INFO] demo figures written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
