"""House visual style for pymaftools figures.

A single place for the shared look: accent colours, standard continuous
colormaps, and an :func:`style_axes` helper that despines and lightens every
Axes the same way. Plotting code should pull colours from here (and from
:class:`~pymaftools.plot.ColorManager.ColorManager` for categorical palettes)
instead of hard-coding hex values, so figures stay visually consistent.
"""

from __future__ import annotations

from matplotlib.axes import Axes

# Single-series accents (bars / histograms that carry no category meaning).
ACCENT = "#4C78A8"   # primary blue
ACCENT_2 = "#F58518"  # secondary orange
MUTED = "#9AA0A6"     # neutral grey

# House continuous colormaps.
SEQUENTIAL_CMAP = "Blues"   # magnitudes / counts (0..max)
DIVERGING_CMAP = "RdBu_r"   # signed values centred on 0 (log2 OR, deltas)

# Shared line / text colours.
SPINE_COLOR = "#444444"
GRID_COLOR = "#E6E6E6"
TEXT_COLOR = "#222222"


def style_axes(ax: Axes, *, despine: bool = True, grid: str | None = None) -> Axes:
    """Apply the house look to one Axes.

    Frameless by default to match OncoPlot / LollipopPlot: removes the top/right
    spines, thins and greys the remaining frame, draws no grid, and tones down
    ticks/labels. Pass ``grid="y"`` only where a faint grid genuinely aids
    reading (it is off by default on purpose).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to restyle.
    despine : bool, default True
        Hide the top and right spines.
    grid : {"y", "x", "both", None}, default None
        Axis to draw the light grid on; ``None`` (default) draws no grid.

    Returns
    -------
    matplotlib.axes.Axes
        The same axes, for chaining.
    """
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(SPINE_COLOR)
    if grid:
        ax.grid(axis=grid, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, color=SPINE_COLOR, labelcolor=TEXT_COLOR)
    return ax


def add_vertical_colorbar(ax, cmap, vmin, vmax, *, label=None, width=0.015, pad=0.01):
    """House colorbar: a thin, frameless, full-height **vertical** bar pinned to
    the right edge of ``ax`` (same pattern as OncoPlot's NumericMatrixTrack), so
    every continuous scale looks identical across the library.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to attach the colorbar to (the bar matches its height).
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float
        Value range.
    label : str, optional
        Colorbar label.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize

    cax = ax.inset_axes([1.0 + pad, 0.0, width, 1.0])
    cbar = ax.figure.colorbar(
        mcm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap),
        cax=cax,
        ticks=[vmin, vmax],
    )
    cbar.ax.tick_params(
        labelsize=7, length=2, width=0.5, color=SPINE_COLOR, labelcolor=TEXT_COLOR
    )
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    if label:
        cbar.set_label(label, fontsize=8, color=TEXT_COLOR)
    return cbar


def fig_with_legend(figsize, *, legend_width: float = 0.2):
    """Create ``(fig, main_ax, legend_ax)`` — a main plot plus a right-hand
    legend panel, matching OncoPlot's card-style legends.

    The legend axis has its frame hidden; draw titled cards onto it with
    :func:`draw_legend_cards`.
    """
    import matplotlib.pyplot as plt

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0 - legend_width, legend_width]},
    )
    legend_ax.axis("off")
    return fig, ax, legend_ax


def draw_legend_cards(
    legend_ax,
    categorical: dict[str, dict[str, str]] | None = None,
    numeric: dict[str, dict] | None = None,
) -> None:
    """Draw the house legend panel on ``legend_ax`` via ``LegendManager``.

    Both categorical swatch cards and continuous colorbars live in the same
    right-hand panel (frameless, consistent width), the way OncoPlot does it —
    so every standalone plot's legend matches.

    Parameters
    ----------
    categorical : dict, optional
        ``{card_title: {category: colour}}`` swatch legends.
    numeric : dict, optional
        ``{card_title: {"colormap": str, "vmin": float, "vmax": float,
        "label": str}}`` colorbar legends.
    """
    from .LegendManager import LegendManager

    manager = LegendManager(legend_ax)
    for title, color_dict in (categorical or {}).items():
        manager.add_legend(title, color_dict)
    for title, info in (numeric or {}).items():
        manager.add_numeric_legend(
            title,
            colormap=info["colormap"],
            vmin=info["vmin"],
            vmax=info["vmax"],
            label=info.get("label"),
        )
    # Roomier spacing than the OncoPlot defaults: a standalone figure's legend
    # axis is short/wide, so the tight OncoPlot values overlap.
    manager.plot_legends(
        ax=legend_ax,
        fontsize=10,
        title_fontsize=12,
        start_y=0.98,
        rect_height=0.04,
        item_offset_y=0.07,
        title_offset_y=0.09,
        legend_gap=0.06,
    )
