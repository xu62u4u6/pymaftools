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


def test_set_config_copies_mutable_inputs(mutation_table):
    width_ratios = [4, 1]
    categorical_columns = ["subtype"]
    plot = OncoPlot(
        mutation_table,
        width_ratios=width_ratios,
        categorical_columns=categorical_columns,
    )

    width_ratios.append(99)
    categorical_columns.append("sex")

    assert plot.width_ratios == [4, 1]
    assert plot.categorical_columns == ["subtype"]


def test_default_oncoplot_does_not_crash(mutation_table):
    """P0#1 regression: the convenience entry point must build and render
    without error. It derives a layout with a spacer before the legend."""
    op = OncoPlot.default_oncoplot(mutation_table, figsize=(8, 6))

    assert op.fig is not None
    # feature gutter + main + 1 right (freq) + spacer + legend = 5 cols;
    # 1 top (bar) + main + sample gutter = 3 rows
    assert op.gs.ncols == 5
    assert op.gs.nrows == 3


def test_mutation_heatmap_registers_track_then_renders(mutation_table):
    """The convenience wrapper registers exactly one MainMatrixTrack; render()
    draws the categorical heatmap (a QuadMesh)."""
    op = OncoPlot(mutation_table, figsize=(8, 6))
    op.mutation_heatmap()

    assert len(op.tracks) == 1
    assert isinstance(op.tracks[0], MainMatrixTrack)

    op.render()
    assert len(op.ax_heatmap.collections) >= 1


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
    assert op.gs.ncols == 4  # feature gutter + heatmap + spacer + legend
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
    op = (
        OncoPlot(table, figsize=(6, 4))
        .main(kind="cnv", cmap="coolwarm", symmetric=True)
        .render()
    )

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
    op = OncoPlot(
        mutation_table, figsize=(8, 6), categorical_columns=["subtype", "sex"]
    )
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

    # rows: 1 top + main + 1 bottom + sample gutter = 4
    assert op.gs.nrows == 4
    # cols: feature gutter + main + 2 right + spacer + 1 legend = 6
    assert op.gs.ncols == 6


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


def test_annotation_accepts_bare_strings_and_routes_bool_as_categorical(
    mutation_table,
):
    """Issue #44: a column name is one column, not an iterable of characters;
    bool metadata is a binary category rather than a continuous measurement."""
    mutation_table.sample_metadata["is_smoker"] = [True, False] * 4
    mutation_table.feature_metadata["is_driver"] = [True, False] * 3
    op = OncoPlot(mutation_table, figsize=(8, 6)).main()

    op.add_sample_annotation("is_smoker")
    op.add_feature_annotation("is_driver")

    annotation_tracks = op.tracks[1:]
    assert [type(track) for track in annotation_tracks] == [
        CategoricalTrack,
        CategoricalTrack,
    ]
    assert [track.name for track in annotation_tracks] == ["is_smoker", "is_driver"]


@pytest.mark.parametrize("grouped", [False, True])
def test_render_twice_preserves_feature_labels(grouped):
    """Issue #44: render may suppress matrix ticks locally, but must restore
    the track configuration so a later render still creates label gutters."""
    table = _grouped_table()
    op = OncoPlot(table, figsize=(7, 5)).main()
    if grouped:
        op.group_features(by="pathway")

    main_track = op.tracks[0]
    op.render()
    assert main_track.yticklabels is True
    op.render()

    shown = {text.get_text() for ax in op.fig.axes for text in ax.texts}
    assert set(table.index) <= shown
    assert main_track.yticklabels is True


@pytest.mark.parametrize("sectioned", [False, True])
def test_left_track_registration_order_and_widths(sectioned):
    """The first left track is outermost and each requested size belongs to
    the column where that track is actually rendered."""
    table = _grouped_table()
    op = OncoPlot(table, figsize=(9, 5), wspace=0).main(yticklabels=False)
    op.add_track(
        BarTrack(np.arange(1, len(table) + 1), side="left", label="outer", size=1)
    )
    op.add_track(
        BarTrack(np.arange(1, len(table) + 1), side="left", label="inner", size=3)
    )
    if sectioned:
        op.group_features(by="pathway", show_titles=False)
    op.render(show_sample_labels=False, legend_pad=0)

    outer_axes = [ax for ax in op.fig.axes if ax.get_title() == "outer"]
    inner_axes = [ax for ax in op.fig.axes if ax.get_title() == "inner"]
    assert len(outer_axes) == len(inner_axes) == (3 if sectioned else 1)
    for outer_ax, inner_ax in zip(outer_axes, inner_axes):
        outer_box = outer_ax.get_position()
        inner_box = inner_ax.get_position()
        assert outer_box.x1 <= inner_box.x0
        assert inner_box.width / outer_box.width == pytest.approx(3.0)
        assert inner_box.x1 <= op.ax_heatmap.get_position().x0


# --- Stage 4: legend filter / xticklabel params / render spacing -----------


def test_mutation_legend_filters_wildtype_and_absent(mutation_table):
    """P1#4: by default the legend drops wild-type ('False'), 'Unknown', and any
    colormap category that does not occur in this cohort."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()

    assert op.has_legend("Mutation")
    legend = op.legend_manager.legend_dict["Mutation"]
    assert "False" not in legend  # wild-type sentinel gone
    assert "Unknown" not in legend  # unknown sentinel gone
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
    """P2#5: the legacy add_xticklabel must honour rotation and fontsize — now by
    restyling the auto-drawn sample-gutter labels (not the heatmap axis)."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render()
    op.add_xticklabel(rotation=45, fontsize=7)

    label = op._sample_gutter_axes[0].texts[0]
    assert label.get_rotation() == 45
    assert label.get_fontsize() == 7


def test_render_legend_pad_adds_named_spacer_column(mutation_table):
    """Default and explicit legend_pad insert one spacer column before the
    legend; legend_pad=0 omits it."""
    op_no_pad = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_no_pad.render(legend_pad=0)

    op_default = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_default.render()

    op_pad = OncoPlot(mutation_table, figsize=(8, 6)).main().add_freq(side="right")
    op_pad.render(legend_pad=2)

    assert op_default.gs.ncols == op_no_pad.gs.ncols + 1
    assert op_pad.gs.ncols == op_no_pad.gs.ncols + 1


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
    assert getattr(bar, "_shared_max", None) is not None
    # only the first bar section keeps the value-axis title
    assert sum(a.get_title() == "TMB" for a in op.fig.axes) == 1


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


def test_group_samples_per_section_freq_strips():
    """group_samples(freq=True) draws each group's own freq strip (X_freq, Y_freq)
    right of its section, independent of an overall freq bar at the far right."""
    t = _grouped_table()
    groups = {g: t.subset(samples=t.sample_metadata["grp"] == g) for g in ["X", "Y"]}
    t = t.add_freq(groups=groups)  # creates X_freq, Y_freq, and overall freq

    op = (
        OncoPlot(t, figsize=(7, 5))
        .main()
        .add_freq(freq_columns=["freq"], side="right")  # overall, far right
        .group_samples(by="grp", freq=True, freq_annot=False)
        .render()
    )

    # Each section drew a strip labelled with ITS OWN freq column (not a shared
    # one), alongside the overall "freq" bar.
    labels = {lbl.get_text() for ax in op.fig.axes for lbl in ax.get_xticklabels()}
    assert {"X_freq", "Y_freq", "freq"} <= labels


def test_group_samples_freq_missing_column_raises():
    """freq=True without the per-group freq columns fails with a guiding error."""
    t = _grouped_table()  # no X_freq / Y_freq computed

    op = OncoPlot(t, figsize=(7, 5)).main().group_samples(by="grp", freq=True)
    with pytest.raises(ValueError, match="add_freq"):
        op.render()


def test_pivottableplot_rename_backward_compat():
    """The old PivotTablePlot import path still resolves to the renamed
    PivotStatsPlot class (S6)."""
    from pymaftools.plot.PivotTablePlot import PivotTablePlot
    from pymaftools.plot.PivotStatsPlot import PivotStatsPlot

    assert PivotTablePlot is PivotStatsPlot


# --- label gutter "blocks" (step 2.5) -------------------------------------- #


def test_feature_names_render_in_gutter_not_on_matrix(mutation_table):
    """The regression: gene names go to a dedicated gutter, so a LEFT track can
    no longer cover them (and they are off the heatmap axis)."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main()
    op.add_track(
        BarTrack(mutation_table.feature_metadata["freq"], side="left", label="freq")
    )
    op.render()

    genes = set(mutation_table.index)
    # the matrix no longer carries gene tick labels
    assert [t.get_text() for t in op.ax_heatmap.get_yticklabels()] == []
    # they appear as text on a dedicated (gutter) axis instead
    gutter_texts = set()
    for ax in op.fig.axes:
        gutter_texts.update(t.get_text() for t in ax.texts)
    assert genes.issubset(gutter_texts)


def test_sample_labels_override_is_applied_positionally(mutation_table):
    """A list-like sample_labels override replaces the default sample index."""
    aliases = [f"alias{i}" for i in range(mutation_table.shape[1])]
    op = OncoPlot(mutation_table, figsize=(8, 6)).main().render(sample_labels=aliases)

    shown = set()
    for ax in op._sample_gutter_axes:
        shown.update(t.get_text() for t in ax.texts)
    assert {"alias0", "alias7"}.issubset(shown)
    # the real barcodes are not shown when overridden
    assert "S00" not in shown


def test_label_override_wrong_length_raises(mutation_table):
    """A mismatched override length fails loud rather than silently misaligning."""
    op = OncoPlot(mutation_table, figsize=(8, 6)).main()
    with pytest.raises(ValueError, match="feature_labels has length"):
        op.render(feature_labels=["only_one_gene"])


def test_show_sample_labels_false_hides_the_sample_gutter(mutation_table):
    """Forcing labels off leaves no sample gutter axis (the auto-hide path)."""
    op = (
        OncoPlot(mutation_table, figsize=(8, 6)).main().render(show_sample_labels=False)
    )
    assert op._sample_gutter_axes == []


def test_grouped_mode_draws_one_sample_gutter_per_section(mutation_table):
    """Grouped layout slices the gutter per section so labels stay aligned."""
    op = (
        OncoPlot(mutation_table, figsize=(9, 6))
        .main()
        .group_samples(by="subtype")
        .render()
    )
    # exactly one sample-gutter axis per sample section (labels sliced per section)
    n_sections = len(op._axis_sections("sample"))
    assert n_sections >= 2
    assert len(op._sample_gutter_axes) == n_sections


def _label_centre_offsets_px(op, gutter_ax, heatmap_ax, positions):
    """Display-space distance between each gutter label and its heatmap cell centre."""
    op.fig.canvas.draw()
    renderer = op.fig.canvas.get_renderer()
    offsets = []
    for text, col in zip(gutter_ax.texts, positions):
        bbox = text.get_window_extent(renderer)
        label_x = bbox.x0 + bbox.width / 2
        cell_x = heatmap_ax.transData.transform((col + 0.5, 0))[0]
        offsets.append(abs(label_x - cell_x))
    return offsets


def _label_y_centre_offsets_px(op, gutter_ax, heatmap_ax, positions):
    """Display-space distance between each feature label and its row centre."""
    op.fig.canvas.draw()
    renderer = op.fig.canvas.get_renderer()
    offsets = []
    for text, row in zip(gutter_ax.texts, positions):
        bbox = text.get_window_extent(renderer)
        label_y = bbox.y0 + bbox.height / 2
        cell_y = heatmap_ax.transData.transform((0, row + 0.5))[1]
        offsets.append(abs(label_y - cell_y))
    return offsets


@pytest.mark.parametrize("n_samples", [6, 33])
def test_sample_labels_centred_on_heatmap_cells(n_samples):
    """Issue #16: sample gutter shares the heatmap's (0, n) data limits, so each
    label sits on its column centre instead of half a cell to the right."""
    table = PivotTable(
        pd.DataFrame(
            "mut", index=["gene"], columns=[f"sample_{i}" for i in range(n_samples)]
        )
    )
    op = OncoPlot(table).main(cmap_dict={"mut": "blue"})
    op.render(show_sample_labels=True)

    gutter = op._sample_gutter_axes[0]
    assert gutter.get_xlim() == op.ax_heatmap.get_xlim()
    offsets = _label_centre_offsets_px(op, gutter, op.ax_heatmap, range(n_samples))
    assert max(offsets) < 1.0


def test_grouped_sample_labels_centred_on_section_cells():
    """Issue #16, grouped layout: each section's gutter aligns with its own
    heatmap section."""
    op = (
        OncoPlot(_grouped_table(), figsize=(7, 5))
        .main()
        .group_samples(by="grp")
        .render(show_sample_labels=True)
    )
    heatmap_axes = sorted(
        (ax for ax in op.fig.axes if ax.collections),
        key=lambda ax: ax.get_position().x0,
    )
    gutters = sorted(op._sample_gutter_axes, key=lambda ax: ax.get_position().x0)
    assert len(gutters) == len(heatmap_axes) == 2
    for gutter, heatmap_ax in zip(gutters, heatmap_axes):
        n = len(gutter.texts)
        assert gutter.get_xlim() == heatmap_ax.get_xlim() == (0, n)
        assert max(_label_centre_offsets_px(op, gutter, heatmap_ax, range(n))) < 1.0


@pytest.mark.parametrize("grouped", [False, True])
def test_feature_labels_centred_on_heatmap_rows(grouped):
    """Issue #16 companion check: the y-gutter uses the same inverted row
    coordinates as each ungrouped or feature-grouped heatmap section."""
    table = _grouped_table()
    op = OncoPlot(table, figsize=(7, 5)).main()
    if grouped:
        op.group_features(by="pathway")
    op.render(show_sample_labels=False)

    features = set(table.index)
    gutters = [
        ax
        for ax in op.fig.axes
        if ax.texts and {text.get_text() for text in ax.texts}.issubset(features)
    ]
    heatmaps = [ax for ax in op.fig.axes if ax.collections]
    gutters.sort(key=lambda ax: ax.get_position().y0, reverse=True)
    heatmaps.sort(key=lambda ax: ax.get_position().y0, reverse=True)
    expected_sections = 3 if grouped else 1
    assert len(gutters) == len(heatmaps) == expected_sections
    for gutter, heatmap_ax in zip(gutters, heatmaps):
        n = len(gutter.texts)
        assert gutter.get_ylim() == heatmap_ax.get_ylim() == (n, 0)
        assert max(_label_y_centre_offsets_px(op, gutter, heatmap_ax, range(n))) < 1.0


@pytest.mark.parametrize(
    ("fig_height", "fontsize", "title_fontsize"),
    [(3, 8, 10), (4, 12, 16), (6, 10, 14), (10, 14, 18)],
)
def test_legend_title_does_not_overlap_first_item(fig_height, fontsize, title_fontsize):
    """Issue #17: on short figures the fixed axes-fraction title offset was
    smaller than the glyph height; spacing is now floored in physical units."""
    table = PivotTable(
        pd.DataFrame("mut", index=["gene"], columns=[f"sample_{i}" for i in range(6)])
    )
    op = OncoPlot(table).set_config(figsize=(9, fig_height))
    op.main(cmap_dict={"mut": "steelblue"}, linewidths=1)
    op.render(show_sample_labels=True)
    op.legend_manager.plot_legends(
        ax=op.ax_legend,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    op.fig.canvas.draw()
    renderer = op.fig.canvas.get_renderer()

    title, first_item = op.ax_legend.texts[:2]
    assert title.get_text() == "Mutation"
    assert first_item.get_text() == "mut"
    title_box = title.get_window_extent(renderer)
    item_box = first_item.get_window_extent(renderer)
    assert title_box.y0 >= item_box.y1


def test_legend_spacing_unchanged_on_tall_axes():
    """The physical floor only kicks in when needed: a tall legend axis keeps
    the caller's fractional spacing."""
    from pymaftools.plot.LegendManager import LegendManager

    fig, ax = plt.subplots(figsize=(3, 12))
    floors = LegendManager._physical_spacing_floor(
        ax,
        fontsize=8,
        title_fontsize=10,
        rect_height=0.02,
        title_offset_y=0.05,
        item_offset_y=0.035,
    )
    plt.close(fig)
    assert floors == (0.05, 0.035)


def test_legend_overflow_warns():
    from pymaftools.plot.LegendManager import LegendManager

    fig, ax = plt.subplots(figsize=(3, 1))
    manager = LegendManager(ax).add_legend(
        "Many", {f"category_{i}": "#888888" for i in range(30)}
    )
    with pytest.warns(UserWarning, match="does not fit"):
        manager.plot_legends()
    plt.close(fig)


def test_top_bartrack_zero_tick_is_never_occluded_by_opaque_label_gutter(
    mutation_table,
):
    """Label gutters must use transparent patches so the top BarTrack '0' tick
    is never sliced by a white axes box drawn afterward."""
    table = mutation_table.copy()
    n_samples = table.shape[1]
    table.sample_metadata["mutations_count"] = np.linspace(10, 80, n_samples)
    table.sample_metadata["subtype"] = ["LUAD"] * (n_samples // 2) + ["LUSC"] * (
        n_samples - n_samples // 2
    )

    for grouped in (False, True):
        op = OncoPlot(table, figsize=(10, 4.2)).main()
        if grouped:
            op = op.group_samples("subtype")
        op = op.add_bar("mutations_count", side="top").render(show_sample_labels=True)
        op.fig.canvas.draw()
        renderer = op.fig.canvas.get_renderer()

        bar_axes = [
            (idx, ax)
            for idx, ax in enumerate(op.fig.axes)
            if ax.get_title() == "mutations_count"
        ]
        assert bar_axes
        for bar_idx, bar_ax in bar_axes:
            assert not bar_ax.patch.get_visible()
            zero_ticks = [
                t
                for t in bar_ax.get_yticklabels()
                if t.get_visible() and t.get_text() in {"0", "0.0"}
            ]
            assert zero_ticks
            tb = zero_ticks[0].get_window_extent(renderer)
            for other_idx, other_ax in enumerate(op.fig.axes):
                if other_idx <= bar_idx or not other_ax.patch.get_visible():
                    continue
                obb = other_ax.get_window_extent(renderer)
                overlap_x = min(tb.x1, obb.x1) - max(tb.x0, obb.x0)
                overlap_y = min(tb.y1, obb.y1) - max(tb.y0, obb.y0)
                assert not (overlap_x > 0 and overlap_y > 0)
        plt.close(op.fig)
