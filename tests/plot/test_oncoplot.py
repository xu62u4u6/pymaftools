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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pymaftools.core.PivotTable import PivotTable
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
    without error. It derives a 3-column layout (heatmap + freq + legend)."""
    op = OncoPlot.default_oncoplot(mutation_table, figsize=(8, 6))

    assert op.fig is not None
    # 0 left + main + 1 right (freq) + legend = 3 cols; 1 top (bar) + main = 2 rows
    assert op.gs.ncols == 3
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


def test_render_path_derives_layout_from_tracks(mutation_table):
    """Declarative path: .main().render() must produce a heatmap + legend axis
    purely from the registered track, independent of the eager axes."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()

    assert op.fig is not None
    assert op.gs.ncols == 2  # heatmap + legend, derived from the single main track
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


def test_plot_numeric_metadata_colorbar_default_on(mutation_table):
    """P1#3: a numeric strip carries a colorbar by default, drawn as an inset
    (child) axis, so the value scale is interpretable."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.main().plot_numeric_metadata().render()  # cbar defaults True

    num_tracks = [t for t in op.tracks if isinstance(t, NumericTrack)]
    assert len(num_tracks) == 1
    assert _total_child_axes(op) == 1  # the inset colorbar


def test_plot_numeric_metadata_colorbar_opt_out(mutation_table):
    """cbar=False must draw no colorbar."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.main().plot_numeric_metadata(cbar=False).render()

    assert _total_child_axes(op) == 0  # no colorbar


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
    # cols: 0 left + main + 2 right + 1 legend = 4
    assert op.gs.ncols == 4


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


# --- S6: unified entry + PivotTablePlot rename backward-compat --------------


def test_table_plot_oncoplot_entry(mutation_table):
    """oncoplot is reachable from the same table.plot accessor as the stats
    plots (unified entry, S6)."""
    op = mutation_table.plot.oncoplot(figsize=(8, 6))
    assert isinstance(op, OncoPlot)
    assert op.pivot_table is mutation_table


def test_pivottableplot_rename_backward_compat():
    """The old PivotTablePlot import path still resolves to the renamed
    PivotStatsPlot class (S6)."""
    from pymaftools.plot.PivotTablePlot import PivotTablePlot
    from pymaftools.plot.PivotStatsPlot import PivotStatsPlot

    assert PivotTablePlot is PivotStatsPlot
