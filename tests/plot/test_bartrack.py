"""Intent tests for the generalized BarTrack (two-axis, stackable, directional).

These encode WHY the generalization matters: a maftools-style oncoplot must be
able to attach a mutation-type-stacked bar (or any bar) on ANY of the four
sides, growing away from or toward the matrix, and stay aligned/scaled when the
matrix is split into grouped sections.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymaftools.plot.Track import BarTrack


def _stacked_frame(n=4):
    # rows = positions (samples/features), cols = stack categories
    return pd.DataFrame(
        {"A": np.arange(1, n + 1), "B": np.arange(n, 0, -1)},
        index=[f"P{i}" for i in range(n)],
    )


CMAP = {"A": "#C0392B", "B": "#7F7F7F"}


def _render(track):
    fig, ax = plt.subplots()
    track.render(ax)
    return fig, ax


def test_vertical_sides_use_position_x_axis():
    """top/bottom are sample-aligned: bars run along x with x pinned to columns."""
    for side in ("top", "bottom"):
        fig, ax = _render(BarTrack(_stacked_frame(), side=side, cmap=CMAP))
        # position axis = x, aligned to 4 columns via the half-cell trick
        assert ax.get_xlim() == (-0.5, 3.5)
        plt.close(fig)


def test_horizontal_sides_use_position_y_axis_inverted():
    """left/right are feature-aligned: bars run along y, inverted to row order."""
    for side in ("left", "right"):
        fig, ax = _render(BarTrack(_stacked_frame(), side=side, cmap=CMAP))
        # position axis = y, inverted so row 0 sits on top (matches heatmap)
        lo, hi = ax.get_ylim()
        assert lo > hi  # inverted
        plt.close(fig)


def test_grow_out_vs_in_flips_value_axis_direction():
    """`grow` controls bar direction: out = away from matrix, in = toward it."""
    # top: value axis = y. out -> grows up (normal), in -> inverted (down).
    _, ax_out = _render(BarTrack(_stacked_frame(), side="top", cmap=CMAP, grow="out"))
    _, ax_in = _render(BarTrack(_stacked_frame(), side="top", cmap=CMAP, grow="in"))
    assert ax_out.get_ylim()[0] < ax_out.get_ylim()[1]   # up
    assert ax_in.get_ylim()[0] > ax_in.get_ylim()[1]     # down (toward matrix)

    # right: value axis = x. out -> grows right (normal), in -> inverted (left).
    _, rx_out = _render(BarTrack(_stacked_frame(), side="right", cmap=CMAP, grow="out"))
    _, rx_in = _render(BarTrack(_stacked_frame(), side="right", cmap=CMAP, grow="in"))
    assert rx_out.get_xlim()[0] < rx_out.get_xlim()[1]   # right
    assert rx_in.get_xlim()[0] > rx_in.get_xlim()[1]     # left (toward matrix)

    # left default (out) mirrors right: grows left (inverted x)
    _, lx_out = _render(BarTrack(_stacked_frame(), side="left", cmap=CMAP, grow="out"))
    assert lx_out.get_xlim()[0] > lx_out.get_xlim()[1]
    plt.close("all")


def test_stacked_draws_one_patch_per_category_per_position():
    """A stacked bar must draw every category segment, not collapse to a total."""
    frame = _stacked_frame(n=4)
    fig, ax = _render(BarTrack(frame, side="top", cmap=CMAP))
    # 4 positions x 2 categories = 8 bar patches
    assert len(ax.patches) == frame.shape[0] * frame.shape[1]
    plt.close(fig)


def test_stacked_contributes_legend_single_series_does_not():
    """Legend only makes sense for the coloured (cmap) stack form."""
    stacked = BarTrack(_stacked_frame(), side="top", label="TMB", cmap=CMAP)
    entries = stacked.legend_entries()
    assert entries == {"TMB": CMAP}

    single = BarTrack(np.array([1, 2, 3, 4]), side="top", label="total")
    assert single.legend_entries() is None


def test_subset_slices_the_side_aligned_axis():
    """top/bottom subset by sample positions; left/right by feature positions.

    This is the bug the rewrite fixes: the old BarTrack ignored `feat`, so a
    feature-aligned bar would not slice with the matrix under grouped rendering.
    """
    top = BarTrack(_stacked_frame(n=4), side="top", cmap=CMAP)
    assert top.subset(samp=[0, 1]).frame.shape[0] == 2
    assert top.subset(feat=[0]).frame.shape[0] == 4  # feat ignored for top

    left = BarTrack(_stacked_frame(n=4), side="left", cmap=CMAP)
    assert left.subset(feat=[0, 1, 2]).frame.shape[0] == 3
    assert left.subset(samp=[0]).frame.shape[0] == 4  # samp ignored for left


def test_backward_compatible_single_value_top_bar():
    """The legacy BarTrack(values_1d, label) call still draws a vertical top bar."""
    track = BarTrack(np.array([5, 3, 8]), "TMB")
    assert track.side == "top"
    fig, ax = _render(track)
    assert ax.get_xlim() == (-0.5, 2.5)         # 3 positions, vertical
    assert len(ax.patches) == 3                  # single series -> 3 bars
    np.testing.assert_array_equal(track.values, [5, 3, 8])
    plt.close(fig)


def test_invalid_grow_raises():
    """grow must be 'out' or 'in' — fail loud on a typo."""
    import pytest

    with pytest.raises(ValueError, match="grow"):
        BarTrack(_stacked_frame(), side="top", grow="up")
