"""Plot fixes whose failure is silent: the figure renders but means the wrong thing."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize

from pymaftools.core.ObservationMask import ObservationMask
from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.SmallVariationTable import SmallVariationTable
from pymaftools.core.MAF import MAF
from pymaftools.plot.ColorManager import ColorManager
from pymaftools.plot.OncoPlot import OncoPlot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_every_nonsynonymous_class_has_a_colour():
    # Missing classes fell back to white, i.e. drawn as wild-type.
    missing = set(MAF.nonsynonymous_types) - set(ColorManager.NONSYNONYMOUS_CMAP)
    assert missing == set()


def test_unobserved_cells_are_drawn_and_listed_as_not_assessed():
    data = pd.DataFrame(
        [["Missense_Mutation", False, False], [False, "Nonsense_Mutation", False]],
        index=["TP53", "KRAS"],
        columns=["S1", "S2", "S3"],
        dtype=object,
    )
    table = SmallVariationTable(data)
    table.observation_mask = ObservationMask(
        pd.DataFrame([[True] * 3, [True, True, False]], index=data.index, columns=data.columns)
    )

    op = table.plot.oncoplot(figsize=(5, 2)).main().render()
    main_track = op.tracks[0]

    assert main_track.legend_entries()["Mutation"]["Not assessed"] == (
        ColorManager.NOT_ASSESSED_STYLE
    )
    pale = matplotlib.colors.to_rgba(ColorManager.NOT_ASSESSED_COLOR)
    hatched_cells = [
        p
        for p in op.ax_heatmap.patches
        if p.get_facecolor() == pale
        and p.get_hatch() == ColorManager.NOT_ASSESSED_HATCH
        and p.get_width() == 1
    ]
    assert len(hatched_cells) == 1

    hatched_legend = [
        p
        for p in op.ax_legend.patches
        if p.get_facecolor() == pale
        and p.get_hatch() == ColorManager.NOT_ASSESSED_HATCH
    ]
    assert len(hatched_legend) == 1


def test_freq_track_annotations_fit_within_narrow_bars():
    data = pd.DataFrame(
        [
            ["Missense_Mutation", False, False, False],
            [False, "Nonsense_Mutation", False, False],
        ],
        index=["TP53", "KRAS"],
        columns=["S1", "S2", "S3", "S4"],
        dtype=object,
    )
    table = SmallVariationTable(data)
    table.observation_mask = ObservationMask(
        pd.DataFrame(
            [[True, True, True, True], [True, True, False, False]],
            index=data.index,
            columns=data.columns,
        )
    )
    table = table.add_freq()

    op = table.plot.oncoplot(figsize=(5.5, 2.2)).main().add_freq().render()
    op.fig.canvas.draw()
    renderer = op.fig.canvas.get_renderer()
    freq_ax = next(
        ax for ax in op.fig.axes if [t.get_text() for t in ax.get_xticklabels()] == ["freq"]
    )
    ax_bbox = freq_ax.get_window_extent(renderer=renderer)
    for text in freq_ax.texts:
        tb = text.get_window_extent(renderer=renderer)
        assert tb.x0 >= ax_bbox.x0 - 1.0
        assert tb.x1 <= ax_bbox.x1 + 1.0


def _numeric_table():
    return PivotTable(
        pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            index=["G1", "G2"],
            columns=["S1", "S2", "S3", "S4"],
        )
    )


def test_numeric_heatmap_cell_colours_match_the_colorbar_scale():
    # center=0 squeezed values into the top half of the colormap while the
    # colorbar used the full vmin..vmax range.
    op = OncoPlot(_numeric_table()).numeric_heatmap(cmap="viridis", vmin=0.0, vmax=5.0).render()
    mesh = next(c for ax in op.fig.axes for c in ax.collections if hasattr(c, "get_array"))

    expected = matplotlib.colormaps["viridis"](Normalize(0.0, 5.0)(1.0))
    np.testing.assert_allclose(mesh.get_facecolors()[0], expected, atol=1e-6)


def test_grouped_numeric_heatmap_with_explicit_range_renders():
    table = _numeric_table()
    table.sample_metadata["grp"] = ["A", "A", "B", "B"]

    op = OncoPlot(table).numeric_heatmap(vmin=0.0, vmax=5.0).group_samples("grp").render()

    assert op.fig is not None


@pytest.mark.parametrize("max_count", [5, 10])
def test_lollipop_heads_sit_on_their_count_ticks(max_count):
    from pymaftools.plot import LollipopPlot

    mutations = [
        {"position": 30 + 20 * i, "type": "Missense_Mutation", "count": count}
        for i, count in enumerate(range(1, max_count + 1))
    ]
    plot = LollipopPlot("DEMO", 400, [], mutations).plot()

    ticks = dict(
        zip(
            [int(t.get_text()) for t in plot.ax_main.get_yticklabels()],
            plot.ax_main.get_yticks(),
        )
    )
    heads = {}
    for collection in plot.ax_main.collections:
        for x, y in collection.get_offsets():
            heads[round(float(x))] = float(y)
    for i, count in enumerate(range(1, max_count + 1)):
        if count in ticks:
            assert heads[30 + 20 * i] == pytest.approx(ticks[count])


def test_pca_shape_groups_share_one_colour_scale():
    from pymaftools.plot.PivotStatsPlot import PivotStatsPlot

    rng = np.random.default_rng(0)
    samples = [f"S{i}" for i in range(6)]
    table = PivotTable(pd.DataFrame(rng.normal(size=(5, 6)), index=list("ABCDE"), columns=samples))
    table.sample_metadata["score"] = [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]
    table.sample_metadata["grp"] = ["a", "a", "a", "b", "b", "b"]

    plot = PivotStatsPlot(table)
    plot.plot_pca_samples(color_col="score", shape_col="grp", is_numeric=True)

    clims = {c.get_clim() for ax in plot.fig.axes for c in ax.collections if c.get_array() is not None}
    assert clims == {(1.0, 300.0)}


def test_paired_boxplot_keeps_only_complete_pairs():
    # Incomplete subjects used to stay in, so position-based pairing matched
    # P2's Pre value with P1's Post value.
    from pymaftools.plot.PivotStatsPlot import PivotStatsPlot

    plot = PivotStatsPlot(PivotTable(pd.DataFrame([[0.0]], index=["G"], columns=["S"])))
    data = pd.DataFrame(
        {
            "group": ["Pre"] * 4 + ["Post"] * 4,
            "patient": ["P1", "P2", "P3", "P4"] * 2,
            "val": [np.nan, 100.0, 2.0, 3.0, 101.0, np.nan, 12.0, 13.0],
        }
    )
    _, ax = plt.subplots()

    plot.plot_boxplot_with_annot(
        data, group_col="group", test_col="val", order=["Pre", "Post"],
        is_paired=True, pair_col="patient", ax=ax, verbose=0,
    )

    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Pre (n=2)", "Post (n=2)"]
