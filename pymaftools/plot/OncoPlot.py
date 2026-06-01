from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .BasePlot import BasePlot
from .Track import (
    Track,
    MainMatrixTrack,
    NumericMatrixTrack,
    BarTrack,
    FreqTrack,
    CategoricalTrack,
    NumericTrack,
    draw_categorical_heatmap,
)

# Type checking imports to avoid circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.PivotTable import PivotTable


class OncoPlot(BasePlot):
    """
    OncoPlot class for creating oncoplots (mutation heatmaps) with metadata annotations.

    This class provides comprehensive functionality for visualizing mutation data along with
    sample and feature metadata, including TMB (Tumor Mutation Burden) plots, frequency plots,
    and categorical/numeric metadata annotations.

    Inherits from BasePlot to utilize unified legend management and save functionality.
    """

    def __init__(self, pivot_table, **kwargs):
        """
        Initialize OncoPlot with a PivotTable and configuration options.

        Parameters
        ----------
        pivot_table : PivotTable
            The PivotTable instance containing mutation data and metadata
        **kwargs : dict
            Configuration options passed to set_config()
        """
        # Initialize BasePlot
        super().__init__()

        # Load PivotTable
        self.pivot_table = pivot_table
        self.feature_metadata = pivot_table.feature_metadata
        self.sample_metadata = pivot_table.sample_metadata

        # Registered tracks (declarative path).
        self.tracks: list[Track] = []

        # Optional axis grouping (drawn by render() as separators + titles).
        self._feature_groups: dict | None = None
        self._sample_groups: dict | None = None

        self.set_config(**kwargs)

    def set_config(
        self,
        line_color: str = "white",
        cmap: str | dict = "nonsynonymous",
        figsize: tuple = (20, 15),
        width_ratios: list[int] = [25, 1, 1, 2],
        height_ratios: list[int] = [1, 20],
        wspace: float = 0.015,
        hspace: float = 0.02,
        categorical_columns: list[str] = [],
        numeric_columns: list[str] = [],
        ytick_fontsize: int = 10,
    ) -> OncoPlot:
        """
        Configure OncoPlot appearance and layout settings.

        Parameters
        ----------
        line_color : str, default "white"
            Color of lines between heatmap cells
        cmap : str or dict, default "nonsynonymous"
            Color map for mutation types
        figsize : tuple, default (20, 15)
            Figure size (width, height)
        width_ratios : list, default [25, 1, 1, 2]
            Layout hint; only the main-matrix width (``width_ratios[0]``) is used
            by ``render()`` to size the heatmap column relative to side tracks.
        height_ratios : list, default [1, 20]
            Layout hint; only the main-matrix height (``height_ratios[-1]``) is
            used by ``render()``.
        wspace : float, default 0.015
            Default inter-axis width spacing for ``render()``.
        hspace : float, default 0.02
            Default inter-axis height spacing for ``render()``.
        categorical_columns : list, default []
            Categorical metadata columns drawn by ``plot_categorical_metadata()``.
        numeric_columns : list, default []
            Numeric metadata columns drawn by ``plot_numeric_metadata()``.
        ytick_fontsize : int, default 10
            Font size for y-axis tick labels

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """

        self.line_color = line_color
        self.cmap = self.color_manager.get_cmap(cmap) if isinstance(cmap, str) else cmap
        self.figsize = figsize
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.wspace = wspace
        self.hspace = hspace
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.ytick_fontsize = ytick_fontsize
        return self

    def plot_numeric_metadata(
        self,
        annotate: bool = False,
        annotation_font_size: int = 10,
        fmt: str = ".2f",
        cmap: str = "Blues",
        cmap_dict: dict | None = None,
        alpha: float = 1,
        linewidths: float = 1,
        colorbar: str | bool = "legend",
    ) -> OncoPlot:
        """
        Register one numeric sample-annotation track per ``numeric_columns``.

        Backward-compatible convenience over :meth:`add_sample_annotation`:
        registers the tracks (drawn on the next :meth:`render`). The columns come
        from ``numeric_columns`` configured in :meth:`set_config`.

        Parameters
        ----------
        annotate : bool, default False
            Whether to display numeric values on the heatmap
        annotation_font_size : int, default 10
            Font size for annotations
        fmt : str, default ".2f"
            Format string for numeric annotations
        cmap : str, default "Blues"
            Default colormap for numeric data
        cmap_dict : dict, optional
            Dictionary mapping column names to specific colormaps
        alpha : float, default 1
            Transparency level (0-1)
        linewidths : float, default 1
            Width of lines between cells
        colorbar : {"legend", "inset", "off"}, default "legend"
            Where to draw each column's colorbar: ``"legend"`` stacks them in the
            shared legend area (best with several numeric columns), ``"inset"``
            puts a small one beside each strip, ``"off"`` none. Accepts a bool
            (True→inset, False→off). (PLOTTING_REVIEW P1#3.)
        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        for col in self.numeric_columns:
            col_cmap = cmap_dict.get(col, "Blues") if cmap_dict else "Blues"
            self.tracks.append(
                NumericTrack(
                    self.sample_metadata[[col]].T,
                    side="bottom",
                    cmap=col_cmap,
                    name=col,
                    line_color=self.line_color,
                    linewidths=linewidths,
                    alpha=alpha,
                    annotate=annotate,
                    annotation_font_size=annotation_font_size,
                    fmt=fmt,
                    ytick_fontsize=self.ytick_fontsize,
                    colorbar=colorbar,
                )
            )
        return self

    @staticmethod
    def categorical_heatmap(
        table,
        category_cmap,
        ax=None,
        fig_size=(10, 6),
        unknown_color="white",
        linecolor="white",
        **kwargs,
    ):
        """Backward-compatible delegate to :func:`Track.draw_categorical_heatmap`."""
        return draw_categorical_heatmap(
            table,
            category_cmap,
            ax=ax,
            fig_size=fig_size,
            unknown_color=unknown_color,
            linecolor=linecolor,
            **kwargs,
        )

    def mutation_heatmap(
        self,
        cmap_dict: dict | None = None,
        linecolor: str = "white",
        linewidths: float = 1,
        show_frame: bool = False,
        n: int = 3,
        yticklabels: bool = True,
        ytick_fontsize: int | None = None,
        show_ylabel: bool = False,
        show_all_categories: bool = False,
    ) -> OncoPlot:
        """
        Plot the main mutation heatmap using categorical color coding.

        Parameters
        ----------
        cmap_dict : dict, optional
            Color mapping for mutation types
        linecolor : str, default "white"
            Color of lines between cells
        linewidths : int, default 1
            Width of lines between cells
        show_frame : bool, default False
            Whether to show frames around groups of columns
        n : int, default 3
            Number of columns per frame group
        yticklabels : bool, default True
            Whether to show y-tick labels (feature names)
        ytick_fontsize : int, optional
            Font size for y-axis tick labels (defaults to self.ytick_fontsize)
        show_ylabel : bool, default False
            Whether to show y-axis label
        show_all_categories : bool, default False
            If False (default), the legend lists only mutation categories that
            occur in the data and are not wild-type; set True to force the full
            colormap (PLOTTING_REVIEW P1#4).
        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        if cmap_dict is None:
            cmap_dict = self.cmap
        if ytick_fontsize is None:
            ytick_fontsize = self.ytick_fontsize

        # Register-only convenience over main(kind="mutation"); drawn on render().
        self.tracks.append(
            MainMatrixTrack(
                self.pivot_table,
                cmap_dict,
                linecolor=linecolor,
                linewidths=linewidths,
                show_frame=show_frame,
                n=n,
                yticklabels=yticklabels,
                ytick_fontsize=ytick_fontsize,
                show_ylabel=show_ylabel,
                show_all_categories=show_all_categories,
            )
        )
        return self

    def main(self, kind: str = "mutation", cmap_dict: dict | None = None, **kwargs) -> OncoPlot:
        """Register the main matrix track for the declarative ``render()`` path.

        Parameters
        ----------
        kind : {"mutation", "cnv"}, default "mutation"
            ``"mutation"`` registers a categorical :class:`MainMatrixTrack`;
            ``"cnv"`` (alias ``"numeric"``) registers a continuous
            :class:`NumericMatrixTrack` for log2-ratio data.
        cmap_dict : dict, optional
            Categorical colormap (mutation only); defaults to ``self.cmap``.
        **kwargs
            Forwarded to the matrix track (e.g. ``show_frame``, ``yticklabels``
            for mutation; ``cmap``, ``symmetric`` for cnv).

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining.
        """
        kwargs.setdefault("ytick_fontsize", self.ytick_fontsize)
        if kind == "mutation":
            if cmap_dict is None:
                cmap_dict = self.cmap
            self.tracks.append(MainMatrixTrack(self.pivot_table, cmap_dict, **kwargs))
        elif kind in ("cnv", "numeric"):
            self.tracks.append(NumericMatrixTrack(self.pivot_table, **kwargs))
        else:
            raise ValueError(
                f"Unknown kind {kind!r}; expected 'mutation' or 'cnv'."
            )
        return self

    def add_bar(
        self, bar_col: str = "TMB", side: str = "top", **kwargs
    ) -> OncoPlot:
        """Register a per-sample bar track (e.g. TMB) for ``render()``."""
        if bar_col not in self.sample_metadata.columns:
            hint = " Please do table.calculate_tmb() first." if bar_col == "TMB" else ""
            raise ValueError(f"Column '{bar_col}' not found in sample metadata.{hint}")
        track = BarTrack(self.sample_metadata[bar_col].values, bar_col, **kwargs)
        track.side = side
        self.tracks.append(track)
        return self

    def add_freq(
        self, freq_columns: list[str] = ["freq"], side: str = "right", **kwargs
    ) -> OncoPlot:
        """Register a per-feature frequency track for ``render()``."""
        track = FreqTrack(
            self.feature_metadata[freq_columns], line_color=self.line_color, **kwargs
        )
        track.side = side
        self.tracks.append(track)
        return self

    def _add_annotation(self, metadata, columns, side, cmap_dict, default_cmap, kwargs):
        """Register one track per column, dtype-inferred (numeric vs categorical),
        aligned to whichever axis ``metadata`` belongs to. ``transpose`` is keyed
        off ``side``: sample annotations (top/bottom) are 1xN rows, feature
        annotations (left/right) are Nx1 columns."""
        horizontal = side in ("top", "bottom")
        for col in columns:
            series = metadata[col]
            data = series.to_frame().T if horizontal else series.to_frame()
            if pd.api.types.is_numeric_dtype(series):
                track = NumericTrack(
                    data,
                    side=side,
                    name=col,
                    line_color=self.line_color,
                    ytick_fontsize=self.ytick_fontsize,
                    **kwargs,
                )
            else:
                column_cmap = (cmap_dict or {}).get(col)
                if not column_cmap:
                    column_cmap = self.color_manager.generate_categorical_cmap(
                        series, default_palette=default_cmap
                    )
                track = CategoricalTrack(
                    data,
                    column_cmap,
                    col,
                    side=side,
                    line_color=self.line_color,
                    ytick_fontsize=self.ytick_fontsize,
                    **kwargs,
                )
            self.tracks.append(track)
        return self

    def add_sample_annotation(
        self,
        columns: list[str],
        side: str = "bottom",
        *,
        cmap_dict: dict | None = None,
        default_cmap: str = "pastel",
        **kwargs,
    ) -> OncoPlot:
        """Register sample-aligned annotation tracks (one per column) for
        ``render()``. Numeric columns become NumericTrack (with colorbar),
        others CategoricalTrack."""
        return self._add_annotation(
            self.sample_metadata, columns, side, cmap_dict, default_cmap, kwargs
        )

    def add_feature_annotation(
        self,
        columns: list[str],
        side: str = "right",
        *,
        cmap_dict: dict | None = None,
        default_cmap: str = "pastel",
        **kwargs,
    ) -> OncoPlot:
        """Register feature-aligned annotation tracks (one per column) for
        ``render()`` — e.g. ``pathway`` / ``is_driver`` from feature_metadata
        drawn as row-side strips. This is the feature-dimension gap (S3 / P1#2)
        that the eager layout never had a slot for."""
        return self._add_annotation(
            self.feature_metadata, columns, side, cmap_dict, default_cmap, kwargs
        )

    def group_features(
        self,
        by: str,
        *,
        show_titles: bool = True,
        title_fontsize: int = 11,
        line_color: str = "black",
        line_width: float = 1.5,
    ) -> OncoPlot:
        """Mark row (feature) groups from a ``feature_metadata`` column.

        ``render()`` then draws a separator line between consecutive groups (on
        the matrix and every feature-aligned track) and, if ``show_titles``, a
        rotated group label on the left. Features must already be contiguous by
        group — sort first, e.g. ``table.sort_features(by="pathway")``.
        """
        self._feature_groups = {
            "by": by,
            "show_titles": show_titles,
            "title_fontsize": title_fontsize,
            "line_color": line_color,
            "line_width": line_width,
        }
        return self

    def group_samples(
        self,
        by: str,
        *,
        show_titles: bool = True,
        title_fontsize: int = 11,
        line_color: str = "black",
        line_width: float = 1.5,
    ) -> OncoPlot:
        """Mark column (sample) groups from a ``sample_metadata`` column.

        ``render()`` draws a separator line between consecutive groups (on the
        matrix and every sample-aligned track) and, if ``show_titles``, a group
        label on top. Samples must already be contiguous by group — sort first,
        e.g. ``table.sort_samples_by_group(group_col="subtype", ...)``.
        """
        self._sample_groups = {
            "by": by,
            "show_titles": show_titles,
            "title_fontsize": title_fontsize,
            "line_color": line_color,
            "line_width": line_width,
        }
        return self

    @staticmethod
    def _group_runs(labels: list) -> list[tuple]:
        """Contiguous runs of equal labels → list of (label, start, end)."""
        runs = []
        start = 0
        n = len(labels)
        for i in range(1, n + 1):
            if i == n or labels[i] != labels[start]:
                runs.append((labels[start], start, i))
                start = i
        return runs

    def _draw_axis_groups(self, axis, cfg, aligned_axes, title_axes) -> None:
        """Draw group separator lines (across the matrix + aligned tracks) and
        group titles for one axis. ``axis`` is ``"sample"`` or ``"feature"``."""
        by = cfg["by"]
        if axis == "sample":
            labels = self.sample_metadata.loc[list(self.pivot_table.columns), by].tolist()
        else:
            labels = self.feature_metadata.loc[list(self.pivot_table.index), by].tolist()

        runs = self._group_runs(labels)
        boundaries = [end for _label, _start, end in runs[:-1]]  # internal only

        for ax in aligned_axes:
            for b in boundaries:
                if axis == "sample":
                    ax.axvline(b, color=cfg["line_color"], lw=cfg["line_width"])
                else:
                    ax.axhline(b, color=cfg["line_color"], lw=cfg["line_width"])

        if not cfg["show_titles"]:
            return
        title_ax = title_axes[0] if axis == "sample" else title_axes[-1]
        for label, start, end in runs:
            center = (start + end) / 2
            if axis == "sample":
                title_ax.annotate(
                    str(label),
                    xy=(center, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=cfg["title_fontsize"],
                    fontweight="bold",
                    annotation_clip=False,
                )
            else:
                title_ax.annotate(
                    str(label),
                    xy=(0.0, center),
                    xycoords=("axes fraction", "data"),
                    xytext=(-40, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=cfg["title_fontsize"],
                    fontweight="bold",
                    annotation_clip=False,
                )

    def render(
        self,
        fig=None,
        *,
        main_width: float | None = None,
        main_height: float | None = None,
        legend_width: float = 3,
        legend_pad: float = 0,
        colorbar_width: float = 0.55,
        wspace: float | None = None,
        hspace: float | None = None,
    ) -> OncoPlot:
        """Derive the layout from registered tracks and draw them in one pass.

        The sole draw path. Groups registered tracks by ``side`` and builds a
        GridSpec whose rows are ``[top..] + main + [bottom..]`` and columns are
        ``[left..] + main + [right..] + legend``; sample-aligned tracks (top/
        bottom) share the matrix width, feature-aligned tracks (left/right) share
        its height. The convenience methods (``mutation_heatmap`` / ``plot_freq``
        / ``plot_bar`` / ``plot_categorical_metadata`` / ``plot_numeric_metadata``
        / ``numeric_heatmap``) only register tracks; this method draws them.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Existing figure to draw into. If *None*, a new figure is created via
            ``plt.figure`` (the only remaining global-pyplot touch; pass ``fig``
            to embed the oncoplot in a larger figure).
        main_width, main_height : float, optional
            Relative size of the main matrix column / row. Default to
            ``width_ratios[0]`` / ``height_ratios[-1]`` from ``set_config``.
        legend_width : float, default 3
            Relative width of the legend column.
        legend_pad : float, default 0
            Relative width of an explicit empty spacer column inserted between the
            right-side tracks and the legend. This is the named, opt-in
            replacement for the old hardcoded "phantom column"; 0 means none.
        colorbar_width : float, default 0.55
            Width (axes fraction) of the stacked numeric colorbars drawn in the
            legend area for tracks with ``colorbar="legend"``.
        wspace, hspace : float, optional
            Inter-axis spacing; default to the configured ``self.wspace`` /
            ``self.hspace``.

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining.
        """
        main_tracks = [t for t in self.tracks if t.side == "main"]
        if not main_tracks:
            raise ValueError("No main track registered; call .main() before render().")

        top = [t for t in self.tracks if t.side == "top"]
        bottom = [t for t in self.tracks if t.side == "bottom"]
        left = [t for t in self.tracks if t.side == "left"]
        right = [t for t in self.tracks if t.side == "right"]

        main_row = len(top)
        main_col = len(left)
        nrows = len(top) + 1 + len(bottom)
        has_pad = legend_pad > 0
        # cols: left.. + main + right.. + [optional pad spacer] + legend
        ncols = len(left) + 1 + len(right) + (1 if has_pad else 0) + 1
        legend_col = ncols - 1

        main_w = self.width_ratios[0] if main_width is None else main_width
        main_h = self.height_ratios[-1] if main_height is None else main_height
        height_ratios = (
            [t.size for t in top] + [main_h] + [t.size for t in bottom]
        )
        width_ratios = (
            [t.size for t in left]
            + [main_w]
            + [t.size for t in right]
            + ([legend_pad] if has_pad else [])
            + [legend_width]
        )

        if fig is None:
            fig = plt.figure(figsize=self.figsize)
        self.fig = fig
        self.gs = GridSpec(
            nrows,
            ncols,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=self.wspace if wspace is None else wspace,
            hspace=self.hspace if hspace is None else hspace,
            figure=fig,
        )

        self.ax_heatmap = fig.add_subplot(self.gs[main_row, main_col])
        self.ax_legend = fig.add_subplot(self.gs[main_row, legend_col])

        main_track = main_tracks[0]
        # A continuous main matrix has a colorbar, not categorical swatches: put
        # it in the (otherwise empty) legend column instead of an inset.
        numeric_main = isinstance(main_track, NumericMatrixTrack)
        if numeric_main:
            main_track.cbar = False
        main_track.render(self.ax_heatmap)

        # sample-aligned tracks share the matrix column; collect axes so group
        # separators can be drawn across the matrix and every aligned track.
        top_axes, bottom_axes, left_axes, right_axes = [], [], [], []
        for i, track in enumerate(top):
            ax = fig.add_subplot(self.gs[i, main_col])
            track.render(ax)
            top_axes.append(ax)
        for i, track in enumerate(bottom):
            ax = fig.add_subplot(self.gs[main_row + 1 + i, main_col])
            track.render(ax)
            bottom_axes.append(ax)
        # feature-aligned tracks share the matrix row; registered outward from it
        for i, track in enumerate(left):
            ax = fig.add_subplot(self.gs[main_row, main_col - 1 - i])
            track.render(ax)
            left_axes.append(ax)
        for i, track in enumerate(right):
            ax = fig.add_subplot(self.gs[main_row, main_col + 1 + i])
            track.render(ax)
            right_axes.append(ax)

        # Group separators + titles (light: overlay lines on aligned axes; no
        # physical gaps / no per-track sectioning).
        if self._sample_groups:
            self._draw_axis_groups(
                "sample",
                self._sample_groups,
                aligned_axes=[self.ax_heatmap, *top_axes, *bottom_axes],
                title_axes=top_axes or [self.ax_heatmap],
            )
        if self._feature_groups:
            self._draw_axis_groups(
                "feature",
                self._feature_groups,
                aligned_axes=[self.ax_heatmap, *left_axes, *right_axes],
                title_axes=left_axes or [self.ax_heatmap],
            )

        # legends: single source of truth, gathered from every track —
        # categorical swatches plus any numeric tracks set to colorbar="legend".
        self.clear_legends()
        for track in self.tracks:
            for name, color_dict in (track.legend_entries() or {}).items():
                self.add_legend(name, color_dict)
            spec = track.colorbar_legend() if hasattr(track, "colorbar_legend") else None
            if spec:
                self.legend_manager.add_numeric_legend(
                    spec["label"], spec["colormap"], spec["vmin"], spec["vmax"]
                )

        lm = self.legend_manager
        if numeric_main and not lm.legend_dict and not lm.numeric_legends:
            # CNV-only oncoplot: a full colorbar fills the empty legend column,
            # anchored west so the thin (aspect-constrained) bar hugs the matrix.
            cbar = main_track.draw_colorbar(self.ax_legend)
            cbar.ax.set_anchor("W")
        else:
            if numeric_main:
                # combined plot: the matrix scale joins the stacked legend colorbars
                lm.add_numeric_legend(
                    "value", main_track.cmap, main_track._vmin, main_track._vmax
                )
            self.plot_all_legends(colorbar_width=colorbar_width)
        return self

    def plot_bar(
        self,
        fontsize: int = 6,
        bar_value: bool = False,
        bar_col: str = "TMB",
        ylabel_size: int = 8,
    ) -> OncoPlot:
        """
        Plot bar chart showing values (typically TMB) for each sample.

        Parameters
        ----------
        fontsize : int, default 6
            Font size for bar value annotations
        bar_value : bool, default False
            Whether to show values on top of bars
        bar_col : str, default "TMB"
            Column name in sample_metadata to use for bar values
        ylabel_size : int, default 8
            Font size for y-axis label

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        if bar_col == "TMB" and bar_col not in self.sample_metadata.columns:
            raise ValueError(
                f"Column '{bar_col}' not found in sample metadata. Please do table.calculate_tmb() first."
            )
        if bar_col not in self.sample_metadata.columns:
            raise ValueError(f"Column '{bar_col}' not found in sample metadata.")

        self.tracks.append(
            BarTrack(
                self.sample_metadata[bar_col].values,
                bar_col,
                bar_value=bar_value,
                fontsize=fontsize,
                ylabel_size=ylabel_size,
            )
        )
        return self

    def plot_freq(
        self,
        freq_columns: list[str] = ["freq"],
        annot_fontsize: int = 9,
        linewidths: float = 1,
        xtick_fontsize: int = 9,
    ) -> OncoPlot:
        """
        Plot frequency heatmap showing mutation frequencies for each gene.

        Parameters
        ----------
        freq_columns : list, default ["freq"]
            List of frequency columns to display
        annot_fontsize : int, default 9
            Font size for annotations
        linewidths: float, default 1
            Width of lines between cells
        xtick_fontsize
            Font size for x-axis tick labels (defaults 9)

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        self.tracks.append(
            FreqTrack(
                self.feature_metadata[freq_columns],
                line_color=self.line_color,
                annot_fontsize=annot_fontsize,
                linewidths=linewidths,
                xtick_fontsize=xtick_fontsize,
            )
        )
        return self

    def plot_categorical_metadata(
        self,
        annotate: bool = False,
        cmap_dict: dict | None = None,
        alpha: float = 1.0,
        default_cmap: str = "pastel",
        annotation_font_size: int = 10,
        annotate_text_color: str = "black",
        linewidths: float = 1,
    ) -> OncoPlot:
        """
        Plot categorical metadata as color-coded heatmaps below the main mutation heatmap.

        Parameters
        ----------
        annotate : bool, default False
            Whether to display category labels on the heatmap
        cmap_dict : dict, optional
            Dictionary mapping columns to color mappings
            Example: {
                "subtype": {
                    "LUAD": "orange",
                    "LUSC": "blue",
                    "ASC": "green"
                },
                "smoke": {
                    "is_smoke": "gray",
                    "no_smoke": "white"
                }
            }
        alpha : float, default 1.0
            Transparency level (0-1)
        default_cmap : str, default "pastel"
            Default color palette for categories without specified colors
        annotation_font_size : int, default 10
            Font size for annotations
        annotate_text_color : str, default "black"
            Color of annotation text
        linewidths : float, default 0.1
            Width of lines between cells

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        for col in self.categorical_columns:
            data = self.sample_metadata[[col]].T  # Ensure you pass a DataFrame

            # Use ColorManager to generate color mapping
            column_cmap = cmap_dict.get(col, {}) if cmap_dict else {}
            if not column_cmap:
                column_cmap = self.color_manager.generate_categorical_cmap(
                    data.iloc[0], default_palette=default_cmap
                )

            self.tracks.append(
                CategoricalTrack(
                    data,
                    column_cmap,
                    col,
                    side="bottom",
                    line_color=self.line_color,
                    linewidths=linewidths,
                    alpha=alpha,
                    annotate=annotate,
                    annotation_font_size=annotation_font_size,
                    annotate_text_color=annotate_text_color,
                    ytick_fontsize=self.ytick_fontsize,
                )
            )

        return self

    def add_xticklabel(
        self, fontsize: int | None = None, rotation: float = 90
    ) -> OncoPlot:
        """
        Add x-axis tick labels (sample names) to the bottom-most subplot.

        Finds the subplot in the bottom row and first column, then adds
        sample names as x-axis tick labels.

        Parameters
        ----------
        fontsize : int, optional
            Font size for the sample labels (matplotlib default if None).
        rotation : float, default 90
            Rotation angle of the sample labels.

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        # Get the maximum row number
        max_row = max([spec.rowspan.stop for spec in self.gs]) - 1

        # Find target axis
        target_ax = None
        for ax in self.fig.axes:
            try:
                subplotspec = ax.get_subplotspec()
                if (
                    subplotspec.rowspan.start == max_row
                    and subplotspec.colspan.start == 0
                ):
                    target_ax = ax
                    break
            except AttributeError:
                # Handle cases where get_subplotspec is not available
                continue

        # Add xtick labels and xticks
        if target_ax:
            target_ax.set_xticks([i + 0.5 for i in range(len(self.sample_metadata))])
            target_ax.set_xticklabels(
                self.sample_metadata.index, rotation=rotation, fontsize=fontsize
            )
        return self

    def numeric_heatmap(
        self,
        cmap: str = "Blues",
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = False,
        yticklabels: bool = True,
        annot: bool = False,
        fmt: str = ".2f",
        ytick_fontsize: int | None = None,
        linewidths: float = 1,
        show_ylabel: bool = False,
    ) -> OncoPlot:
        """
        Plot numeric heatmap with customizable y-axis tick label font size.

        Parameters
        ----------
        cmap : str, default "Blues"
            Colormap for the heatmap
        vmin : float, optional
            Minimum value for color scale
        vmax : float, optional
            Maximum value for color scale
        symmetric : bool, default False
            Whether to use symmetric color scale
        yticklabels : bool, default True
            Whether to show y-axis labels (gene names)
        annot : bool, default False
            Whether to annotate cells with values
        fmt : str, default ".2f"
            Format string for annotations
        ytick_fontsize : int, optional
            Font size for y-axis tick labels (defaults to self.ytick_fontsize)
        linewidths : int, default 1
            Width of lines between cells
        show_ylabel : bool, default False
        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        if ytick_fontsize is None:
            ytick_fontsize = self.ytick_fontsize

        # Register-only convenience over main(kind="cnv"); drawn on render(),
        # where the NumericMatrixTrack renders its own inset colorbar.
        self.tracks.append(
            NumericMatrixTrack(
                self.pivot_table,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                symmetric=symmetric,
                yticklabels=yticklabels,
                annot=annot,
                fmt=fmt,
                ytick_fontsize=ytick_fontsize,
                linewidths=linewidths,
                show_ylabel=show_ylabel,
            )
        )
        return self

    @staticmethod
    def default_oncoplot(
        pivot_table: "PivotTable",
        figsize: tuple[int, int] = (30, 15),
    ) -> OncoPlot:
        """
        Create a default oncoplot with standard components.

        Convenience constructor that registers the main mutation heatmap, the
        frequency column and the TMB bar, then renders.

        Parameters
        ----------
        pivot_table : PivotTable
            The PivotTable instance containing mutation data
        figsize : tuple, default (30, 15)
            Figure size (width, height)

        Returns
        -------
        oncoplot : OncoPlot
            Configured and rendered OncoPlot instance
        """
        return (
            OncoPlot(pivot_table=pivot_table, figsize=figsize)
            .main()
            .add_bar("TMB", side="top")
            .add_freq(side="right")
            .render()
            .add_xticklabel()
        )
