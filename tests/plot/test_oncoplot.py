"""Smoke tests for OncoPlot.

Covers the Stage 1 track-based refactor:
- ``default_oncoplot`` no longer crashes (regression for the 3-vs-4 column
  ``width_ratios`` bug, PLOTTING_REVIEW P0#1).
- ``mutation_heatmap`` is a thin wrapper that registers a ``MainMatrixTrack``
  and contributes the expected ``Mutation`` legend.
- the declarative ``OncoPlot(table).main().render()`` path derives a layout from
  registered tracks and draws without touching the eager ``update_layout`` axes.

These are the first tests for OncoPlot at all (PLOTTING_REVIEW P0#7).
"""

import matplotlib

matplotlib.use("Agg")  # headless rendering

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import pymaftools

from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.SmallVariationTable import SmallVariationTable
from pymaftools.plot.OncoPlot import OncoPlot
from pymaftools.plot.Track import (
    MainMatrixTrack,
    NumericMatrixTrack,
    BarTrack,
    FreqTrack,
    CategoricalTrack,
    NumericTrack,
)

VARIANT_TYPES = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del"]
DATA_DIR = Path(__file__).resolve().parents[2] / "pymaftools" / "data"


@pytest.fixture
def mutation_table():
    """Synthetic categorical mutation PivotTable.

    Wild-type cells are boolean ``False`` (matching ``MAF.to_pivot_table()``'s
    ``fillna(False)``), mutated cells hold a variant-classification string, so
    the matrix is object dtype mixing ``False`` and strings.
    """
    rng = np.random.default_rng(0)
    genes = [f"GENE{i:02d}" for i in range(6)]
    samples = [f"S{i:02d}" for i in range(8)]

    matrix = np.full((len(genes), len(samples)), False, dtype=object)
    mutated = rng.random(matrix.shape) < 0.4
    matrix[mutated] = rng.choice(VARIANT_TYPES, size=int(mutated.sum()))

    table = PivotTable(pd.DataFrame(matrix, index=genes, columns=samples))
    table.sample_metadata["TMB"] = rng.gamma(2.0, 1.0, len(samples))
    table.sample_metadata["subtype"] = rng.choice(["LUAD", "LUSC"], len(samples))
    table.sample_metadata["sex"] = rng.choice(["M", "F"], len(samples))
    table.sample_metadata["age"] = rng.integers(45, 80, len(samples)).astype(float)
    table.feature_metadata["pathway"] = rng.choice(["RTK-RAS", "TP53"], len(genes))
    return table.add_freq()


def teardown_function(_):
    plt.close("all")


def test_default_oncoplot_does_not_crash(mutation_table):
    """P0#1 regression: the convenience entry point must build and render
    without error. It derives a layout with a spacer before the legend."""
    op = OncoPlot.default_oncoplot(mutation_table, figsize=(8, 6))

    assert op.fig is not None
    # 0 left + main + 1 right (freq) + spacer + legend = 4 cols;
    # 1 top (bar) + main = 2 rows
    assert op.gs.ncols == 4
    assert op.gs.nrows == 2


def test_mutation_heatmap_registers_track_then_renders(mutation_table):
    """The convenience wrapper registers exactly one MainMatrixTrack; render()
    draws the categorical heatmap (a QuadMesh)."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.mutation_heatmap()

    assert len(op.tracks) == 1
    assert isinstance(op.tracks[0], MainMatrixTrack)

    op.render()
    assert len(op.ax_heatmap.collections) >= 1


def test_mutation_legend_content(mutation_table):
    """After render the legend carries the present variant types and never the
    'Unknown' sentinel."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()

    assert op.has_legend("Mutation")
    legend = op.legend_manager.legend_dict["Mutation"]
    # variant types present in the data appear; wild-type/absent are filtered
    assert set(legend).issubset(set(VARIANT_TYPES))
    assert "Unknown" not in legend


def test_legend_swatch_default_is_wide_and_short(mutation_table):
    """Legend swatches should be squat rectangles, not near-square blocks."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()

    patch = op.ax_legend.patches[0]
    assert patch.get_width() == pytest.approx(0.06)
    assert patch.get_height() == pytest.approx(0.02)


def test_render_path_derives_layout_from_tracks(mutation_table):
    """Declarative path: .main().render() must produce a heatmap + legend axis
    purely from the registered track, independent of the eager axes."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()

    assert op.fig is not None
    assert op.gs.ncols == 3  # heatmap + spacer + legend
    assert len(op.ax_heatmap.collections) >= 1
    assert op.has_legend("Mutation")


def test_render_without_main_track_raises(mutation_table):
    """render() must fail loudly when nothing was registered."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    with pytest.raises(ValueError, match="No main track"):
        op.render()


def test_main_rejects_unknown_kind(mutation_table):
    """main() accepts 'mutation' and 'cnv'; anything else must error."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    with pytest.raises(ValueError, match="Unknown kind"):
        op.main(kind="bogus")


def test_main_cnv_renders_numeric_matrix_with_colorbar():
    """main(kind='cnv') renders a continuous matrix with its own inset colorbar,
    so CNV/numeric main matrices are reachable through render()."""
    rng = np.random.default_rng(1)
    table = PivotTable(
        pd.DataFrame(
            rng.normal(0.0, 0.8, (5, 6)),
            index=[f"g{i}" for i in range(5)],
            columns=[f"s{i}" for i in range(6)],
        )
    )
    op = OncoPlot(table, figsize=(6, 4)).main(
        kind="cnv", cmap="coolwarm", symmetric=True
    ).render()

    assert any(isinstance(t, NumericMatrixTrack) for t in op.tracks)
    assert len(op.ax_heatmap.collections) >= 1
    # the colorbar fills the legend column (a QuadMesh on ax_legend), not an inset
    assert len(op.ax_legend.collections) >= 1
    assert len(op.ax_heatmap.child_axes) == 0


# --- Stage 2: sample-annotation tracks -------------------------------------


def test_plot_bar_registers_bartrack_and_draws(mutation_table):
    """plot_bar registers a top BarTrack; render() draws one bar per sample."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.main().plot_bar(bar_col="TMB").render()

    bar_tracks = [t for t in op.tracks if isinstance(t, BarTrack)]
    assert len(bar_tracks) == 1 and bar_tracks[0].side == "top"
    # one bar patch per sample, drawn on the (top) bar axis
    assert any(len(ax.patches) == mutation_table.shape[1] for ax in op.fig.axes)


def test_plot_freq_registers_freqtrack_and_draws(mutation_table):
    """plot_freq registers a right FreqTrack; render() draws the freq heatmap."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.main().plot_freq().render()

    freq_tracks = [t for t in op.tracks if isinstance(t, FreqTrack)]
    assert len(freq_tracks) == 1 and freq_tracks[0].side == "right"
    assert op.fig is not None


def test_plot_categorical_metadata_tracks_and_legends(mutation_table):
    """One CategoricalTrack per configured column, each contributing its own
    legend after render()."""
    op = OncoPlot(mutation_table, figsize=(8, 6), categorical_columns=["subtype", "sex"])
    op.main().plot_categorical_metadata().render()

    cat_tracks = [t for t in op.tracks if isinstance(t, CategoricalTrack)]
    assert len(cat_tracks) == 2
    assert op.has_legend("subtype") and op.has_legend("sex")


def _total_child_axes(op):
    return sum(len(ax.child_axes) for ax in op.fig.axes)


def test_numeric_colorbar_legend_mode_default(mutation_table):
    """P1#3: by default a numeric column's colorbar goes to the shared legend
    area (a numeric_legend), readable even with several columns."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.main().plot_numeric_metadata().render()  # colorbar defaults "legend"

    assert [t for t in op.tracks if isinstance(t, NumericTrack)]
    assert "age" in op.legend_manager.numeric_legends


def test_numeric_colorbar_inset_mode(mutation_table):
    """colorbar='inset' draws a small colorbar on the strip itself, not in the
    legend area."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.main().plot_numeric_metadata(colorbar="inset").render()

    assert "age" not in op.legend_manager.numeric_legends
    assert _total_child_axes(op) >= 1  # an inset colorbar exists somewhere


def test_numeric_colorbar_off(mutation_table):
    """colorbar='off' (or False) draws no colorbar at all."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.main().plot_numeric_metadata(colorbar="off").render()

    assert "age" not in op.legend_manager.numeric_legends
    assert _total_child_axes(op) == 0


def test_numeric_colorbar_per_column_cmaps(mutation_table):
    """Different cmaps per numeric column are honoured (each track keeps its own
    cmap and contributes its own legend colorbar)."""
    mutation_table.sample_metadata["score"] = np.linspace(0, 1, mutation_table.shape[1])
    op = OncoPlot(mutation_table, figsize=(9, 6), numeric_columns=["age", "score"])
    op.main().plot_numeric_metadata(
        cmap_dict={"age": "Blues", "score": "coolwarm"}
    ).render()

    tracks = {t.label: t for t in op.tracks if isinstance(t, NumericTrack)}
    assert tracks["age"].cmap == "Blues"
    assert tracks["score"].cmap == "coolwarm"
    assert {"age", "score"} <= set(op.legend_manager.numeric_legends)


# --- Stage 3: feature annotation + multi-side render() ----------------------


def test_add_feature_annotation_builds_feature_aligned_track(mutation_table):
    """The S3 headline: feature_metadata becomes a feature-aligned track.

    Data must be Nx1 (one row per feature, aligned to the matrix rows), not the
    1xN used for sample annotations."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main()
    op.add_feature_annotation(["pathway"], side="right")

    feat_tracks = [t for t in op.tracks if isinstance(t, CategoricalTrack)]
    assert len(feat_tracks) == 1
    track = feat_tracks[0]
    assert track.side == "right"
    # Nx1: rows == number of features, single column
    assert track.data.shape == (mutation_table.shape[0], 1)


def test_render_derives_multi_side_layout(mutation_table):
    """render() must size the GridSpec from the registered tracks' sides:
    rows = top + main + bottom, cols = left + main + right + legend."""
    op = (
        OncoPlot(mutation_table, figsize=(10, 8))
        .main()
        .add_bar("TMB", side="top")  # 1 top
        .add_freq(side="right")  # 1 right
        .add_feature_annotation(["pathway"], side="right")  # +1 right
        .add_sample_annotation(["subtype"], side="bottom")  # 1 bottom
    )
    op.render()

    # rows: 1 top + main + 1 bottom = 3
    assert op.gs.nrows == 3
    # cols: 0 left + main + 2 right + spacer + 1 legend = 5
    assert op.gs.ncols == 5


def test_render_feature_annotation_legend_present(mutation_table):
    """The feature track must contribute its legend through the single-source
    legend gather in render()."""
    op = (
        OncoPlot(mutation_table, figsize=(8, 6))
        .main()
        .add_feature_annotation(["pathway"], side="right")
    )
    op.render()

    assert op.has_legend("Mutation")
    assert op.has_legend("pathway")


def test_add_sample_annotation_dtype_inference(mutation_table):
    """A numeric column must become a NumericTrack, a string column a
    CategoricalTrack, both from one add_sample_annotation entry point."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main()
    op.add_sample_annotation(["age"], side="bottom")  # numeric
    op.add_sample_annotation(["subtype"], side="bottom")  # categorical

    assert any(isinstance(t, NumericTrack) for t in op.tracks)
    assert any(isinstance(t, CategoricalTrack) for t in op.tracks)


def test_full_declarative_render_smoke(mutation_table):
    """End-to-end: a full oncoplot through render() draws every side without
    error and yields a heatmap plus all expected legends."""
    op = (
        OncoPlot(mutation_table, figsize=(12, 8))
        .main()
        .add_bar("TMB", side="top")
        .add_freq(side="right")
        .add_feature_annotation(["pathway"], side="right")
        .add_sample_annotation(["subtype", "sex"], side="bottom")
        .add_sample_annotation(["age"], side="bottom")
    )
    op.render()

    assert len(op.ax_heatmap.collections) >= 1
    for name in ("Mutation", "pathway", "subtype", "sex"):
        assert op.has_legend(name)


# --- Stage 4: legend filter / xticklabel params / render spacing -----------


def test_mutation_legend_filters_wildtype_and_absent(mutation_table):
    """P1#4: by default the legend drops wild-type ('False') and any colormap
    category that does not occur in this cohort."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()

    legend = op.legend_manager.legend_dict["Mutation"]
    assert "False" not in legend  # wild-type sentinel gone
    assert "Multi_Hit" not in legend  # in the cmap but absent from the data
    # only variant types actually present remain
    assert set(legend).issubset(set(VARIANT_TYPES))
    assert len(legend) >= 1


def test_show_all_categories_restores_full_legend(mutation_table):
    """The escape hatch must bring back the full colormap (minus Unknown)."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.mutation_heatmap(show_all_categories=True).render()

    legend = op.legend_manager.legend_dict["Mutation"]
    assert "False" in legend  # full colormap includes the wild-type key
    assert "Multi_Hit" in legend  # and categories absent from the data


def test_add_xticklabel_rotation_and_fontsize(mutation_table):
    """P2#5: add_xticklabel must honour rotation and fontsize instead of a
    hardcoded rotation=90."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()
    op.add_xticklabel(rotation=45, fontsize=7)

    label = op.ax_heatmap.get_xticklabels()[0]
    assert label.get_rotation() == 45
    assert label.get_fontsize() == 7


def test_render_legend_pad_adds_named_spacer_column(mutation_table):
    """legend_pad inserts one explicit spacer column (the named replacement for
    the old phantom column); without it there is none."""
    op_no_pad = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_no_pad.render(legend_pad=0)

    op_pad = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_pad.render(legend_pad=2)

    assert op_pad.gs.ncols == op_no_pad.gs.ncols + 1


def test_render_default_legend_pad_adds_space_before_legend(mutation_table):
    """The default oncoplot keeps the legend off the main plot."""
    op_no_pad = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_no_pad.render(legend_pad=0)

    op_default = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_default.render()

    assert op_default.gs.ncols == op_no_pad.gs.ncols + 1


def test_freq_track_default_annotation_fontsize_is_smaller(mutation_table):
    """Frequency cell annotations default one point smaller than before."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")

    track = next(t for t in op.tracks if isinstance(t, FreqTrack))
    assert track.annot_fontsize == 8


# --- S6: unified entry + PivotTablePlot rename backward-compat --------------


def test_table_plot_oncoplot_entry(mutation_table):
    """oncoplot is reachable from the same table.plot accessor as the stats
    plots (unified entry, S6)."""
    op = mutation_table.plot.oncoplot(figsize=(8, 6))
    assert isinstance(op, OncoPlot)
    assert op.pivot_table is mutation_table


def _grouped_table():
    rng = np.random.default_rng(2)
    genes = [f"g{i}" for i in range(6)]
    samples = [f"s{i}" for i in range(8)]
    m = np.full((6, 8), False, dtype=object)
    m[rng.random((6, 8)) < 0.4] = "Missense_Mutation"
    t = PivotTable(pd.DataFrame(m, index=genes, columns=samples))
    t.feature_metadata["pathway"] = ["A", "A", "B", "B", "C", "C"]  # 3 groups
    t.sample_metadata["grp"] = ["X", "X", "X", "X", "Y", "Y", "Y", "Y"]  # 2 groups
    t.sample_metadata["TMB"] = rng.gamma(2, 1, 8)
    return t


def test_grouping_sections_matrix_and_titles():
    """group_features/group_samples split the matrix into Gf x Gs section axes
    (real gaps) and draw a title per group in both directions."""
    op = (
        OncoPlot(_grouped_table(), figsize=(7, 5))
        .main()
        .group_features(by="pathway")
        .group_samples(by="grp")
        .render()
    )

    # 3 feature groups x 2 sample groups = 6 heatmap sub-axes (each a QuadMesh)
    heatmap_axes = [ax for ax in op.fig.axes if ax.collections]
    assert len(heatmap_axes) == 3 * 2
    titles = {txt.get_text() for ax in op.fig.axes for txt in ax.texts}
    assert {"A", "B", "C", "X", "Y"} <= titles


def test_grouping_sections_aligned_tracks_with_shared_scale():
    """A sample-aligned track (TMB bar) is split per sample group, and the bar's
    y-range is pinned (shared) before slicing so the sections stay comparable."""
    op = (
        OncoPlot(_grouped_table(), figsize=(7, 5))
        .main()
        .add_bar("TMB", side="top")
        .group_samples(by="grp")
        .render()
    )

    # no feature grouping -> 1 row; 2 sample groups -> 2 matrix sections
    heatmap_axes = [ax for ax in op.fig.axes if ax.collections]
    assert len(heatmap_axes) == 2
    # the bar got a shared y-range fixed before the per-section slicing
    bar = next(tr for tr in op.tracks if isinstance(tr, BarTrack))
    assert getattr(bar, "_shared_ymax", None) is not None
    # only the first bar section keeps the y-label
    assert sum(a.get_ylabel() == "TMB" for a in op.fig.axes) == 1


def test_real_tcga_lung_grouped_oncoplot_full_layout():
    """Real TCGA lung fixture: large grouped oncoplot with sample grouping,
    feature grouping, subtype sample annotation, and frequency track."""
    table = pymaftools.read_h5(DATA_DIR / "example_tcga_lung_mutation_grouped.h5")

    assert isinstance(table, SmallVariationTable)
    assert table.shape == (62, 958)
    assert table.sample_metadata["subtype"].value_counts().to_dict() == {
        "LUAD": 490,
        "LUSC": 468,
    }
    assert table.feature_metadata["gene_family"].value_counts().to_dict() == {
        "Other": 58,
        "ZNF": 2,
        "MUC": 2,
    }

    op = (
        OncoPlot(table, figsize=(18, 12))
        .main(yticklabels=False, linewidths=0)
        .add_freq(side="right", annot=False, linewidths=0)
        .add_sample_annotation(
            ["subtype"],
            side="bottom",
            cmap_dict={"subtype": {"LUAD": "#4C78A8", "LUSC": "#F58518"}},
            linewidths=0,
            size=0.6,
        )
        .group_features(by="gene_family")
        .group_samples(by="subtype")
        .render(wspace=0.02, hspace=0.02)
    )

    titles = {txt.get_text() for ax in op.fig.axes for txt in ax.texts}
    assert {"LUAD", "LUSC", "ZNF", "MUC", "Other"} <= titles
    assert op.has_legend("Mutation")
    assert op.has_legend("subtype")


def test_pivottableplot_rename_backward_compat():
    """The old PivotTablePlot import path still resolves to the renamed
    PivotStatsPlot class (S6)."""
    from pymaftools.plot.PivotTablePlot import PivotTablePlot
    from pymaftools.plot.PivotStatsPlot import PivotStatsPlot

    assert PivotTablePlot is PivotStatsPlot
