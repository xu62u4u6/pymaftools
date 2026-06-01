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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Normalize
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
        show_all_categories: bool = False,
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
        self.show_all_categories = show_all_categories

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

    # wild-type / no-mutation sentinels never worth a legend entry. Note the
    # matrix stores boolean ``False`` for wild-type while the cmap keys it as the
    # string ``"False"`` — exclude both. (PLOTTING_REVIEW P1#4)
    _WILDTYPE = frozenset({False, "False", "", None})

    def legend_entries(self) -> dict[str, dict]:
        if self.show_all_categories:
            legend = {k: v for k, v in self.cmap_dict.items() if k != "Unknown"}
            return {"Mutation": legend}

        # Default: only categories that actually occur in the matrix and are not
        # wild-type. Because the matrix uses boolean False (not the string
        # "False"), "key present in data" already drops wild-type and every cmap
        # category absent from this cohort; the sentinel set is a belt-and-braces.
        present = set(pd.unique(self.table.values.ravel()))
        legend = {
            k: v
            for k, v in self.cmap_dict.items()
            if k != "Unknown" and k in present and k not in self._WILDTYPE
        }
        return {"Mutation": legend}


class BarTrack(Track):
    """Per-sample bar strip (e.g. TMB) drawn above the matrix."""

    side = "top"

    def __init__(
        self,
        values,
        label: str,
        *,
        bar_value: bool = False,
        fontsize: int = 6,
        ylabel_size: int = 8,
    ) -> None:
        self.values = np.asarray(values)
        self.label = label
        self.bar_value = bar_value
        self.fontsize = fontsize
        self.ylabel_size = ylabel_size

    def render(self, ax: Axes) -> Axes:
        x = np.arange(len(self.values))
        ax.bar(x, self.values, width=0.95, color="gray", edgecolor="white")
        ax.set_xlim(-0.5, len(self.values) - 0.5)
        if self.bar_value:
            for i, value in enumerate(self.values):
                # NOTE: the original OncoPlot.plot_bar formatted the whole array
                # here (``f"{bar_values:.1f}"``), which raised on an ndarray;
                # bar_value defaults False so it was never hit. Format the scalar.
                ax.text(
                    i, value + 2, f"{value:.1f}", ha="center", fontsize=self.fontsize
                )

        ax.spines["left"].set_visible(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_ylabel(self.label, fontsize=self.ylabel_size)
        return ax


class FreqTrack(Track):
    """Per-feature frequency annotation column drawn right of the matrix."""

    side = "right"

    def __init__(
        self,
        freq_data: pd.DataFrame,
        *,
        line_color: str = "white",
        annot_fontsize: int = 9,
        linewidths: float = 1,
        xtick_fontsize: int = 9,
    ) -> None:
        self.freq_data = freq_data
        self.line_color = line_color
        self.annot_fontsize = annot_fontsize
        self.linewidths = linewidths
        self.xtick_fontsize = xtick_fontsize

    def render(self, ax: Axes) -> Axes:
        sns.heatmap(
            self.freq_data,
            cbar=False,
            linewidths=self.linewidths,
            linecolor=self.line_color,
            ax=ax,
            xticklabels=self.freq_data.columns,
            yticklabels=False,
            annot=True,
            fmt=".2f",
            annot_kws={"size": self.annot_fontsize},
            cmap="Blues",
        )
        ax.tick_params(axis="x", labelsize=self.xtick_fontsize)
        ax.set_ylabel("")
        ax.set_yticks([])  # hide y-axis
        return ax


class CategoricalTrack(Track):
    """Single categorical annotation strip.

    Bound to either axis of the main matrix via ``side`` (S3 — feature and
    sample annotations are the same kind of track):

    - ``side`` in ``("top", "bottom")`` → sample-aligned, horizontal strip;
      ``data`` is 1xN (the metadata field as a row), its name labels the y-axis.
    - ``side`` in ``("left", "right")`` → feature-aligned, vertical strip;
      ``data`` is Nx1 (the metadata field as a column), its name labels the
      x-axis; feature ticks are left to the main heatmap.
    """

    side = "bottom"

    def __init__(
        self,
        data: pd.DataFrame,
        column_cmap: dict,
        name: str,
        *,
        side: str = "bottom",
        line_color: str = "white",
        linewidths: float = 1,
        alpha: float = 1.0,
        annotate: bool = False,
        annotation_font_size: int = 10,
        annotate_text_color: str = "black",
        ytick_fontsize: int = 10,
    ) -> None:
        self.data = data  # 1xN (sample-aligned) or Nx1 (feature-aligned)
        self.column_cmap = column_cmap
        self.name = name
        self.side = side
        self.line_color = line_color
        self.linewidths = linewidths
        self.alpha = alpha
        self.annotate = annotate
        self.annotation_font_size = annotation_font_size
        self.annotate_text_color = annotate_text_color
        self.ytick_fontsize = ytick_fontsize

    def render(self, ax: Axes) -> Axes:
        horizontal = self.side in ("top", "bottom")
        _fig, ax, _info = draw_categorical_heatmap(
            table=self.data,
            category_cmap=self.column_cmap,
            ax=ax,
            linecolor=self.line_color,
            linewidths=self.linewidths,
            xticklabels=False,
            yticklabels=list(self.data.index) if horizontal else False,
            alpha=self.alpha,
        )

        if self.annotate:
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    ax.text(
                        j + 0.5,
                        i + 0.5,  # center the text
                        f"{self.data.iloc[i, j]}",
                        ha="center",
                        va="center",
                        fontsize=self.annotation_font_size,
                        color=self.annotate_text_color,
                    )

        if horizontal:
            ax.set_xticks([])
            ax.set_yticks([i + 0.5 for i in range(len(self.data.index))])
            ax.set_yticklabels(
                self.data.index, rotation=0, fontsize=self.ytick_fontsize
            )
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, top=False)
        else:
            # feature-aligned: name on x, leave feature ticks to the heatmap
            ax.set_yticks([])
            ax.set_xticks([j + 0.5 for j in range(len(self.data.columns))])
            ax.set_xticklabels(
                self.data.columns, rotation=90, fontsize=self.ytick_fontsize
            )
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, right=False)
        return ax

    def legend_entries(self) -> dict[str, dict]:
        return {self.name: self.column_cmap}


class NumericTrack(Track):
    """Single numeric sample-annotation strip drawn below the matrix.

    Unlike the legacy ``plot_numeric_metadata`` it carries its own colorbar
    (an inset axis), so the value scale is interpretable (PLOTTING_REVIEW P1#3).
    """

    side = "bottom"

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        side: str = "bottom",
        cmap: str = "Blues",
        line_color: str = "white",
        linewidths: float = 1,
        alpha: float = 1,
        annotate: bool = False,
        annotation_font_size: int = 10,
        fmt: str = ".2f",
        ytick_fontsize: int = 10,
        cbar: bool = True,
    ) -> None:
        self.data = data  # 1xN (sample-aligned) or Nx1 (feature-aligned)
        self.side = side
        self.cmap = cmap
        self.line_color = line_color
        self.linewidths = linewidths
        self.alpha = alpha
        self.annotate = annotate
        self.annotation_font_size = annotation_font_size
        self.fmt = fmt
        self.ytick_fontsize = ytick_fontsize
        self.cbar = cbar

    def render(self, ax: Axes) -> Axes:
        horizontal = self.side in ("top", "bottom")
        # Color range: symmetric around 0 for the diverging "coolwarm" map,
        # otherwise span the data (matches legacy plot_numeric_metadata).
        if self.cmap == "coolwarm":
            vextreme = max(abs(self.data.min().min()), abs(self.data.max().max()))
            vmin, vmax = -vextreme, vextreme
        else:
            vmin = float(self.data.min().min())
            vmax = float(self.data.max().max())

        sns.heatmap(
            self.data,
            cmap=self.cmap,
            cbar=False,
            linewidths=self.linewidths,
            linecolor=self.line_color,
            ax=ax,
            xticklabels=False,
            yticklabels=list(self.data.index) if horizontal else False,
            annot=self.annotate,
            fmt=self.fmt if self.annotate else "",
            annot_kws={"size": self.annotation_font_size} if self.annotate else None,
            alpha=self.alpha,
            vmin=vmin,
            vmax=vmax,
        )
        if horizontal:
            ax.set_yticks([i + 0.5 for i in range(len(self.data.index))])
            ax.set_yticklabels(
                self.data.index, rotation=0, fontsize=self.ytick_fontsize
            )
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_yticks([])
            ax.set_xticks([j + 0.5 for j in range(len(self.data.columns))])
            ax.set_xticklabels(
                self.data.columns, rotation=90, fontsize=self.ytick_fontsize
            )
            ax.set_ylabel("")

        if self.cbar:
            self._draw_colorbar(ax, vmin, vmax)
        return ax

    def _draw_colorbar(self, ax: Axes, vmin: float, vmax: float) -> None:
        """Draw a compact colorbar in an inset axis just right of the strip."""
        cax = ax.inset_axes([1.01, 0.0, 0.01, 1.0])
        cbar = ax.figure.colorbar(
            cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=self.cmap),
            cax=cax,
            ticks=[vmin, vmax],
        )
        cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)
