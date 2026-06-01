"""Track abstractions for declarative OncoPlot composition.

A ``Track`` owns a strip of the oncoplot aligned to one axis of the main matrix.
Each track knows how to draw itself onto a matplotlib ``Axes`` (:meth:`render`)
and what it contributes to the shared legend (:meth:`legend_entries`).

See ``PLOTTING_REVIEW.md`` ("目標架構") for the full design. Stage 1 introduces
only the abstract :class:`Track` and the :class:`MainMatrixTrack`; the
sample/feature annotation tracks come in later stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def draw_categorical_heatmap(
    table,
    category_cmap,
    ax=None,
    fig_size=(10, 6),
    unknown_color="white",
    linecolor="white",
    **kwargs,
):
    """Draw a categorical heatmap and return ``(fig, ax, legend_info)``.

    Implementation shared by :class:`MainMatrixTrack` and (from Stage 2) the
    annotation tracks. Moved verbatim from ``OncoPlot.categorical_heatmap``,
    which now delegates here so external callers keep working.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.get_figure()
    category_to_index = {k: i for i, k in enumerate(category_cmap.keys())}
    table_mapped = table.map(lambda x: category_to_index.get(x, len(category_cmap)))
    has_unknown = table.map(lambda x: x not in category_cmap).any().any()

    color_list = list(category_cmap.values())
    if has_unknown:
        color_list.append(unknown_color)
    cmap = ListedColormap(color_list)

    # plot heatmap
    sns.heatmap(
        table_mapped, cmap=cmap, cbar=False, ax=ax, linecolor=linecolor, **kwargs
    )

    # prepare legend info
    legend_info = list(category_cmap.items())
    if has_unknown:
        legend_info.append(("Unknown", unknown_color))

    return fig, ax, legend_info


class Track(ABC):
    """Abstract strip of an oncoplot aligned to one axis of the main matrix.

    Attributes
    ----------
    side : str
        ``"main"`` for the central matrix; ``"top"``/``"bottom"`` align to
        samples (columns); ``"left"``/``"right"`` align to features (rows).
    size : float
        Relative thickness of the track's slot in the derived layout.
    """

    side: str = "main"
    size: float = 1.0

    @abstractmethod
    def render(self, ax: Axes) -> Axes:
        """Draw this track onto ``ax``."""

    def legend_entries(self) -> dict[str, dict] | None:
        """Return ``{legend_name: {category: color}}`` contributed to the
        shared legend, or ``None`` if this track has no legend."""
        return None


class MainMatrixTrack(Track):
    """Main mutation heatmap (categorical variant matrix).

    Owns every drawing parameter that ``OncoPlot.mutation_heatmap`` used to
    apply inline, so that method can stay a thin wrapper around this track.
    """

    side = "main"

    def __init__(
        self,
        table: pd.DataFrame,
        cmap_dict: dict,
        *,
        linecolor: str = "white",
        linewidths: float = 1,
        show_frame: bool = False,
        n: int = 3,
        yticklabels: bool = True,
        ytick_fontsize: int = 10,
        show_ylabel: bool = False,
    ) -> None:
        self.table = table
        self.cmap_dict = cmap_dict
        self.linecolor = linecolor
        self.linewidths = linewidths
        self.show_frame = show_frame
        self.n = n
        self.yticklabels = yticklabels
        self.ytick_fontsize = ytick_fontsize
        self.show_ylabel = show_ylabel

    def render(self, ax: Axes) -> Axes:
        _fig, ax, _legend_info = draw_categorical_heatmap(
            table=self.table,
            category_cmap=self.cmap_dict,
            ax=ax,
            linecolor=self.linecolor,
            linewidths=self.linewidths,
            vmin=0,  # ensure mapping uses full range
            vmax=len(self.cmap_dict),
        )

        ax.set_xticks([])
        ax.set_xlabel("")
        if not self.show_ylabel:
            ax.set_ylabel("")
        if self.yticklabels:
            ax.set_yticks([i + 0.5 for i in range(len(self.table.index))])
            ax.set_yticklabels(
                self.table.index, rotation=0, fontsize=self.ytick_fontsize
            )
        else:
            ax.set_yticks([])

        # Show frame every `n` columns
        if self.show_frame:
            for i in range(0, len(self.table.columns), self.n):
                rect = Rectangle(
                    (i, -0.5),  # x, y
                    self.n,  # width
                    len(self.table) + 1,  # height
                    linewidth=1,
                    edgecolor="lightgray",
                    facecolor="none",
                )
                ax.add_patch(rect)
        return ax

    def legend_entries(self) -> dict[str, dict]:
        mutation_legend = {
            key: self.cmap_dict[key]
            for key in self.cmap_dict.keys()
            if key != "Unknown"
        }
        return {"Mutation": mutation_legend}
