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
    return table.add_freq()


def teardown_function(_):
    plt.close("all")


def test_default_oncoplot_does_not_crash(mutation_table):
    """P0#1 regression: the convenience entry point must build a 4-column
    GridSpec, not raise ValueError on a width_ratios/column-count mismatch."""
    op = OncoPlot.default_oncoplot(mutation_table, figsize=(8, 6))

    assert op.fig is not None
    # eager layout is a 4-column GridSpec; default_oncoplot must agree
    assert op.gs.ncols == 4
    assert len(op.width_ratios) == 4


def test_mutation_heatmap_registers_track_and_draws(mutation_table):
    """The thin wrapper must register exactly one MainMatrixTrack and draw the
    categorical heatmap (a QuadMesh) onto the eager heatmap axis."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.mutation_heatmap()

    assert len(op.tracks) == 1
    assert isinstance(op.tracks[0], MainMatrixTrack)
    # sns.heatmap adds a QuadMesh collection
    assert len(op.ax_heatmap.collections) >= 1


def test_mutation_legend_content(mutation_table):
    """Legend must carry every cmap category except the sentinel 'Unknown'
    (current behaviour; wild-type 'False' filtering is deferred to Stage 4)."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.mutation_heatmap()

    assert op.has_legend("Mutation")
    legend = op.legend_manager.legend_dict["Mutation"]
    # variant types that exist in the default nonsynonymous cmap are present
    for variant in VARIANT_TYPES:
        assert variant in legend
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


def test_main_rejects_non_mutation_kind(mutation_table):
    """Stage 1 only implements the mutation matrix; other kinds must error."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    with pytest.raises(ValueError, match="kind='mutation'"):
        op.main(kind="cnv")


# --- Stage 2: sample-annotation tracks -------------------------------------


def test_plot_bar_registers_bartrack_and_draws(mutation_table):
    """plot_bar must register a BarTrack and draw one bar per sample."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.plot_bar(bar_col="TMB")

    bar_tracks = [t for t in op.tracks if isinstance(t, BarTrack)]
    assert len(bar_tracks) == 1
    # one bar patch per sample
    assert len(op.ax_bar.patches) == mutation_table.shape[1]
    assert op.ax_bar.get_ylabel() == "TMB"


def test_plot_freq_registers_freqtrack_and_draws(mutation_table):
    """plot_freq must register a FreqTrack and draw the freq heatmap."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.plot_freq()

    assert any(isinstance(t, FreqTrack) for t in op.tracks)
    assert len(op.ax_freq.collections) >= 1


def test_plot_categorical_metadata_tracks_and_legends(mutation_table):
    """One CategoricalTrack per column, each contributing its own legend."""
    op = OncoPlot(mutation_table, figsize=(8, 6), categorical_columns=["subtype", "sex"])
    op.plot_categorical_metadata()

    cat_tracks = [t for t in op.tracks if isinstance(t, CategoricalTrack)]
    assert len(cat_tracks) == 2
    assert op.has_legend("subtype") and op.has_legend("sex")


def test_plot_numeric_metadata_colorbar_default_on(mutation_table):
    """P1#3: a numeric strip carries a colorbar by default, drawn as an inset
    (child) axis of the strip, so the value scale is interpretable."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.plot_numeric_metadata()  # cbar defaults True

    num_tracks = [t for t in op.tracks if isinstance(t, NumericTrack)]
    assert len(num_tracks) == 1
    # the inset colorbar is a child axis of the strip
    assert len(op.axs_numeric_columns["age"].child_axes) == 1


def test_plot_numeric_metadata_colorbar_opt_out(mutation_table):
    """cbar=False must preserve the legacy no-colorbar behaviour."""
    op = OncoPlot(mutation_table, figsize=(8, 6), numeric_columns=["age"])
    op.plot_numeric_metadata(cbar=False)

    assert len(op.axs_numeric_columns["age"].child_axes) == 0  # no colorbar
