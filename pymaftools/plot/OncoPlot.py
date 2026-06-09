from __future__ import annotations

import numpy as np
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

    # Above this many samples, render() hides the per-column tick labels by
    # default (they overprint into an unreadable smear); override per call with
    # render(show_sample_labels=True/False).
    _SAMPLE_LABEL_LIMIT = 50

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

        Legacy shorthand for the canonical ``main(kind="mutation")``; prefer
        ``main()`` in new code.

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

    def add_track(self, track: Track) -> OncoPlot:
        """Register an arbitrary pre-built :class:`Track` for ``render()``.

        The escape hatch behind the curated ``add_*`` helpers: any ``Track``
        subclass (e.g. a :class:`BarTrack` placed on any side) can be attached
        here. Position and order follow ``(side, registration order)`` — the
        track's ``side`` places it on top/bottom/left/right, and the order in
        which tracks are added sets distance from the matrix (first added =
        outermost). No separate ordering parameter is needed.

        Examples
        --------
        >>> op.add_track(BarTrack(
        ...     pt.sample_metadata[[f"tmb_{g}" for g in ColorManager.FUNCTIONAL_ORDER]],
        ...     side="top", label="TMB", cmap=ColorManager.FUNCTIONAL_CMAP))
        """
        self.tracks.append(track)
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
        """Register a per-feature frequency track for ``render()``.

        Parameters
        ----------
        freq_columns : list of str, default ["freq"]
            ``feature_metadata`` columns to draw, one strip each (e.g.
            ``["LUAD_freq", "LUSC_freq"]``). Populate group-specific columns with
            :meth:`PivotTable.add_freq` (``groups=`` or ``group_col=``).
        side : {"right", "left"}, default "right"
            Which side of the matrix to place the strip(s) on.
        **kwargs
            Forwarded to :class:`Track.FreqTrack` (e.g. ``annot``, ``linewidths``,
            ``vmin``/``vmax`` — defaults pin the scale to 0–1).

        Notes
        -----
        For a *signed* per-feature numeric column (e.g. ``delta_freq``) use
        :meth:`add_feature_annotation` with a diverging cmap instead; ``add_freq``
        is for 0–1 proportions on a shared scale.
        """
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
        annotations (left/right) are Nx1 columns.

        ``cmap_dict`` is a per-column override for BOTH dtypes:

        - numeric column -> the entry is the matplotlib cmap name for that
          :class:`Track.NumericTrack`, taking precedence over a global ``cmap``
          passed in ``kwargs``. A ``"coolwarm"`` (diverging) cmap additionally
          triggers a symmetric-around-0 value scale (``[-|max|, +|max|]``), so a
          signed column like ``delta_freq`` centres its midpoint at 0.
        - categorical column -> the entry is a ``{category: color}`` mapping;
          categories without a colour fall back to a ``default_cmap`` palette.

        Columns absent from ``cmap_dict`` use the global ``cmap`` in ``kwargs``
        (numeric, default ``"Blues"`` in NumericTrack) or an auto-generated
        ``default_cmap`` palette (categorical)."""
        horizontal = side in ("top", "bottom")
        for col in columns:
            series = metadata[col]
            data = series.to_frame().T if horizontal else series.to_frame()
            if pd.api.types.is_numeric_dtype(series):
                # per-column cmap from cmap_dict (e.g. "coolwarm" for a signed
                # delta column) takes precedence over a global cmap in kwargs.
                numeric_kwargs = dict(kwargs)
                column_cmap = (cmap_dict or {}).get(col)
                if column_cmap:
                    numeric_kwargs["cmap"] = column_cmap
                track = NumericTrack(
                    data,
                    side=side,
                    name=col,
                    line_color=self.line_color,
                    ytick_fontsize=self.ytick_fontsize,
                    **numeric_kwargs,
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
        others CategoricalTrack.

        Parameters
        ----------
        columns : list of str
            ``sample_metadata`` columns to draw, one track each.
        side : {"bottom", "top"}, default "bottom"
            Which side of the matrix to place the strips on.
        cmap_dict : dict, optional
            Per-column colour override (numeric cmap name or categorical
            ``{category: color}`` mapping); see :meth:`_add_annotation`.
        default_cmap : str, default "pastel"
            Palette for categorical categories without an explicit colour.
        **kwargs
            Forwarded to the per-column track (``NumericTrack`` /
            ``CategoricalTrack``), e.g. ``annotate``, ``linewidths``, ``size``.

        See Also
        --------
        add_feature_annotation : same, on the feature (row) axis.
        """
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
        that the eager layout never had a slot for.

        Parameters
        ----------
        columns : list of str
            ``feature_metadata`` columns to draw, one track each.
        side : {"right", "left"}, default "right"
            Which side of the matrix to place the strips on.
        cmap_dict : dict, optional
            Per-column colour override. For a numeric column the entry is a cmap
            name and takes precedence over a global ``cmap`` in ``kwargs``; a
            ``"coolwarm"`` cmap centres a signed column (e.g. ``delta_freq``)
            symmetrically around 0. For a categorical column it is a
            ``{category: color}`` mapping. See :meth:`_add_annotation`.
        default_cmap : str, default "pastel"
            Palette for categorical categories without an explicit colour.
        **kwargs
            Forwarded to the per-column track (``NumericTrack`` /
            ``CategoricalTrack``), e.g. ``annotate``, ``linewidths``.

        Notes
        -----
        Use this (not :meth:`add_freq`) for signed/continuous feature columns:
        ``add_freq`` is for 0–1 frequency proportions on a shared 0–1 scale,
        while ``add_feature_annotation`` infers the scale per column (and goes
        symmetric for ``coolwarm``).

        Examples
        --------
        >>> # diverging delta_freq strip, symmetric around 0
        >>> op.add_feature_annotation(
        ...     ["delta_freq"], side="right",
        ...     cmap_dict={"delta_freq": "coolwarm"}, annotate=True,
        ... )
        """
        return self._add_annotation(
            self.feature_metadata, columns, side, cmap_dict, default_cmap, kwargs
        )

    def group_features(
        self,
        by: str,
        *,
        gap: float = 0.5,
        show_titles: bool = True,
        title_fontsize: int = 11,
        title_align: str = "start",
    ) -> OncoPlot:
        """Split rows (features) into groups from a ``feature_metadata`` column.

        ``render()`` draws each group as its own section with a whitespace
        ``gap`` (in matrix-cell units) between them and, if ``show_titles``, a
        rotated group label on the left (``title_align`` = ``"start"`` top /
        ``"center"`` / ``"end"`` bottom). Features must already be contiguous by
        group — sort first, e.g. ``table.sort_features(by="pathway")``.
        """
        self._feature_groups = {
            "by": by,
            "gap": gap,
            "show_titles": show_titles,
            "title_fontsize": title_fontsize,
            "title_align": title_align,
        }
        return self

    def group_samples(
        self,
        by: str,
        *,
        gap: float = 0.5,
        show_titles: bool = True,
        title_fontsize: int = 11,
        title_align: str = "start",
        freq: bool = False,
        freq_suffix: str = "_freq",
        freq_size: float = 1.0,
        freq_annot: bool = True,
    ) -> OncoPlot:
        """Split columns (samples) into groups from a ``sample_metadata`` column.

        ``render()`` draws each group as its own section with a whitespace
        ``gap`` (in matrix-cell units) between them and, if ``show_titles``, a
        group label on top (``title_align`` = ``"start"`` left / ``"center"`` /
        ``"end"`` right). Samples must already be contiguous by group — sort
        first, e.g. ``table.sort_samples_by_group(group_col="subtype", ...)``.

        Per-section frequency
        ---------------------
        With ``freq=True`` each section gets its own frequency strip immediately
        to its right, reading ``feature_metadata[f"{label}{freq_suffix}"]`` (e.g.
        the ``LUAD_freq`` / ``LUSC_freq`` columns produced by
        ``PivotTable.add_freq(groups=...)``). The strips share a 0–1 colour scale
        so groups are comparable. An overall frequency bar is independent — add it
        the usual way with ``.add_freq(freq_columns=["freq"])``.

        Parameters
        ----------
        freq : bool, default False
            Draw a per-section frequency strip right of each sample section.
        freq_suffix : str, default "_freq"
            Suffix appended to each section label to find its frequency column.
        freq_size : float, default 1.0
            Width of each per-section frequency strip (matrix-cell units).
        freq_annot : bool, default True
            Annotate each strip cell with its frequency value.
        """
        self._sample_groups = {
            "by": by,
            "gap": gap,
            "show_titles": show_titles,
            "title_fontsize": title_fontsize,
            "title_align": title_align,
            "freq": freq,
            "freq_suffix": freq_suffix,
            "freq_size": freq_size,
            "freq_annot": freq_annot,
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

    def _axis_sections(self, axis: str) -> list[tuple]:
        """Contiguous group sections along an axis → list of (label, positions).

        ``positions`` are integer positions into the matrix rows (feature) or
        columns (sample) in their current order. A single ``(None, all)`` section
        when that axis is not grouped."""
        if axis == "sample":
            cfg = self._sample_groups
            order = list(self.pivot_table.columns)
            n = len(order)
            if not cfg:
                return [(None, np.arange(n))]
            labels = self.sample_metadata.loc[order, cfg["by"]].tolist()
        else:
            cfg = self._feature_groups
            order = list(self.pivot_table.index)
            n = len(order)
            if not cfg:
                return [(None, np.arange(n))]
            labels = self.feature_metadata.loc[order, cfg["by"]].tolist()
        return [
            (label, np.arange(start, end))
            for label, start, end in self._group_runs(labels)
        ]

    def render(
        self,
        fig=None,
        *,
        main_width: float | None = None,
        main_height: float | None = None,
        legend_width: float = 3,
        legend_pad: float = 1,
        colorbar_width: float = 0.55,
        wspace: float | None = None,
        hspace: float | None = None,
        show_sample_labels: bool | None = None,
        feature_labels=None,
        sample_labels=None,
        feature_gutter_size: float = 3.0,
        sample_gutter_size: float = 3.0,
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
        legend_pad : float, default 1
            Relative width of an explicit empty spacer column inserted between the
            right-side tracks and the legend. Use 0 to remove the spacer.
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

        main_track = main_tracks[0]
        # A continuous main matrix has a colorbar, not categorical swatches: put
        # it in the (otherwise empty) legend column instead of an inset.
        numeric_main = isinstance(main_track, NumericMatrixTrack)
        if numeric_main:
            main_track.cbar = False

        main_w = self.width_ratios[0] if main_width is None else main_width
        main_h = self.height_ratios[-1] if main_height is None else main_height
        if fig is None:
            fig = plt.figure(figsize=self.figsize)
        self.fig = fig
        wspace = self.wspace if wspace is None else wspace
        hspace = self.hspace if hspace is None else hspace

        # Sample (column) tick labels are unreadable for large cohorts; auto-hide
        # them past _SAMPLE_LABEL_LIMIT unless the caller forces show_sample_labels.
        if show_sample_labels is None:
            show_sample_labels = self.pivot_table.shape[1] <= self._SAMPLE_LABEL_LIMIT

        # Gene / sample names live in dedicated gutter blocks (default = matrix
        # index, or a positional list-like override). The matrix hands off its
        # own gene labels so they can't overflow into a left track.
        feature_labels = self._resolve_labels(
            feature_labels, self.pivot_table.index, "feature"
        )
        sample_labels = self._resolve_labels(
            sample_labels, self.pivot_table.columns, "sample"
        )
        show_feature_labels = bool(getattr(main_track, "yticklabels", False))
        main_track.yticklabels = False
        self._sample_gutter_axes = []  # collected by the layout, restyled by add_xticklabel

        layout = self._render_sectioned if (
            self._feature_groups or self._sample_groups
        ) else self._render_simple
        layout(
            fig, main_track, top, bottom, left, right,
            main_w=main_w, main_h=main_h, legend_width=legend_width,
            legend_pad=legend_pad, wspace=wspace, hspace=hspace,
            show_sample_labels=show_sample_labels,
            show_feature_labels=show_feature_labels,
            feature_labels=feature_labels, sample_labels=sample_labels,
            feature_gutter_size=feature_gutter_size,
            sample_gutter_size=sample_gutter_size,
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

    @staticmethod
    def _resolve_labels(override, default, kind: str) -> list:
        """Default to the matrix axis index, or a positional list-like override.

        The override is applied position-for-position in matrix order, so its
        length must match the axis (fail loud otherwise)."""
        if override is None:
            return list(default)
        override = list(override)
        if len(override) != len(default):
            raise ValueError(
                f"{kind}_labels has length {len(override)}, "
                f"expected {len(default)} ({kind}s in the matrix)."
            )
        return override

    @staticmethod
    def _draw_label_gutter(ax, labels, *, axis: str, n: int, fontsize: int = 8,
                           rotation: float = 90) -> None:
        """Draw axis labels in a dedicated (outermost) cell — the "block".

        One mechanism for both axes: ``axis="y"`` draws feature (row) labels
        hugging the cell's right edge; ``axis="x"`` draws sample (column) labels
        below. Positions match the matrix (cell centres at ``i + 0.5``). Because
        the gutter is the outermost cell on its side, any label overflow spills
        into the empty figure margin rather than over a neighbouring track."""
        if axis == "y":
            ax.set_ylim(n, 0)  # inverted to match sns.heatmap row order
            ax.set_yticks([i + 0.5 for i in range(len(labels))])
            ax.set_yticklabels(labels, fontsize=fontsize)
            ax.yaxis.tick_right()  # names sit against the matrix side
            ax.tick_params(axis="y", length=0)
            ax.set_xticks([])
        else:
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_xticks([i + 0.5 for i in range(len(labels))])
            ax.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)
            ax.tick_params(axis="x", length=0)
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _render_simple(
        self, fig, main_track, top, bottom, left, right, *,
        main_w, main_h, legend_width, legend_pad, wspace, hspace,
        show_sample_labels=True, show_feature_labels=True,
        feature_labels=None, sample_labels=None,
        feature_gutter_size=3.0, sample_gutter_size=3.0,
    ) -> None:
        """Ungrouped layout: one cell per track along each axis.

        (``show_sample_labels`` is accepted for a uniform layout interface; the
        ungrouped path draws sample tick labels only via ``add_xticklabel()``.)
        """
        n_feat, n_samp = self.pivot_table.shape
        has_pad = legend_pad > 0
        fgut = 1 if show_feature_labels else 0  # feature-label gutter column
        sgut = 1 if show_sample_labels else 0   # sample-label gutter row

        # Columns: [left tracks] [feature gutter?] [main] [right tracks] [pad?] [legend]
        gut_col = len(left) if fgut else None
        main_col = len(left) + fgut
        right_cols = [main_col + 1 + i for i in range(len(right))]
        legend_col = main_col + 1 + len(right) + (1 if has_pad else 0)
        width_ratios = (
            [t.size for t in left]
            + ([feature_gutter_size] if fgut else [])
            + [main_w] + [t.size for t in right]
            + ([legend_pad] if has_pad else []) + [legend_width]
        )

        # Rows: [top tracks] [main] [bottom tracks] [sample gutter?]
        main_row = len(top)
        sgut_row = (main_row + 1 + len(bottom)) if sgut else None
        height_ratios = (
            [t.size for t in top] + [main_h] + [t.size for t in bottom]
            + ([sample_gutter_size] if sgut else [])
        )

        self.gs = GridSpec(
            len(height_ratios), len(width_ratios), width_ratios=width_ratios,
            height_ratios=height_ratios, wspace=wspace, hspace=hspace, figure=fig,
        )
        self.ax_heatmap = fig.add_subplot(self.gs[main_row, main_col])
        self.ax_legend = fig.add_subplot(self.gs[main_row, legend_col])
        main_track.render(self.ax_heatmap)
        for i, t in enumerate(top):
            t.render(fig.add_subplot(self.gs[i, main_col]))
        for i, t in enumerate(bottom):
            t.render(fig.add_subplot(self.gs[main_row + 1 + i, main_col]))
        # left[0] sits adjacent to the matrix (just left of the feature gutter)
        for i, t in enumerate(left):
            t.render(fig.add_subplot(self.gs[main_row, len(left) - 1 - i]))
        for i, t in enumerate(right):
            t.render(fig.add_subplot(self.gs[main_row, right_cols[i]]))

        if fgut:
            self._draw_label_gutter(
                fig.add_subplot(self.gs[main_row, gut_col]),
                feature_labels, axis="y", n=n_feat,
            )
        if sgut:
            ax = fig.add_subplot(self.gs[sgut_row, main_col])
            self._draw_label_gutter(ax, sample_labels, axis="x", n=n_samp)
            self._sample_gutter_axes.append(ax)

    @staticmethod
    def _fix_shared_scales(main_track, top, bottom, left, right) -> None:
        """Pin each non-categorical track's value range to its full data so the
        sections (which render sliced copies) stay on one consistent scale."""
        if isinstance(main_track, NumericMatrixTrack) and main_track.vmin is None:
            tbl = main_track.table
            if main_track.symmetric:
                ext = max(abs(tbl.min().min()), abs(tbl.max().max()))
                main_track.vmin, main_track.vmax = -ext, ext
            else:
                main_track.vmin = float(tbl.min().min())
                main_track.vmax = float(tbl.max().max())
            main_track._vmin, main_track._vmax = main_track.vmin, main_track.vmax
        for t in [*top, *bottom, *left, *right]:
            if isinstance(t, NumericTrack) and t.vmin is None:
                d = t.data
                if t.cmap == "coolwarm":
                    ext = max(abs(d.min().min()), abs(d.max().max()))
                    t.vmin, t.vmax = -ext, ext
                else:
                    t.vmin, t.vmax = float(d.min().min()), float(d.max().max())
            elif isinstance(t, FreqTrack) and t.vmin is None:
                d = t.freq_data
                t.vmin, t.vmax = float(d.min().min()), float(d.max().max())
            elif isinstance(t, BarTrack):
                totals = t.frame.sum(axis=1)
                t._shared_max = float(totals.max()) if len(totals) else 1.0

    def _render_sectioned(
        self, fig, main_track, top, bottom, left, right, *,
        main_w, main_h, legend_width, legend_pad, wspace, hspace,
        show_sample_labels=True, show_feature_labels=True,
        feature_labels=None, sample_labels=None,
        feature_gutter_size=3.0, sample_gutter_size=3.0,
    ) -> None:
        """Grouped layout: split the matrix into sections with whitespace gaps,
        rendering a sliced copy of each track per section, plus group titles."""
        fsecs = self._axis_sections("feature")
        ssecs = self._axis_sections("sample")
        gf, gs_ = len(fsecs), len(ssecs)
        fcfg, scfg = self._feature_groups, self._sample_groups
        feat_gap = fcfg["gap"] if fcfg else 0.0
        samp_gap = scfg["gap"] if scfg else 0.0
        show_ft = bool(fcfg and fcfg["show_titles"])
        show_st = bool(scfg and scfg["show_titles"])
        n_feat, n_samp = self.pivot_table.shape

        self._fix_shared_scales(main_track, top, bottom, left, right)

        height_ratios: list[float] = []
        def add_row(h):
            height_ratios.append(h)
            return len(height_ratios) - 1

        stitle_row = add_row(1.0) if show_st else None
        top_rows = [add_row(t.size) for t in top]
        feat_rows = []
        for fi, (_lbl, pos) in enumerate(fsecs):
            feat_rows.append(add_row(main_h * len(pos) / n_feat))
            if fi < gf - 1:
                add_row(feat_gap)
        bottom_rows = [add_row(t.size) for t in bottom]
        sgut_row = add_row(sample_gutter_size) if show_sample_labels else None

        width_ratios: list[float] = []
        def add_col(w):
            width_ratios.append(w)
            return len(width_ratios) - 1

        # Per-section sample frequency strips (group_samples(freq=True)): one
        # narrow column right after each sample section, before its gap.
        sfreq = bool(scfg and scfg.get("freq"))
        sfreq_suffix = scfg.get("freq_suffix", "_freq") if scfg else "_freq"
        if sfreq:
            missing = [
                f"{lbl}{sfreq_suffix}"
                for lbl, _ in ssecs
                if f"{lbl}{sfreq_suffix}" not in self.feature_metadata.columns
            ]
            if missing:
                raise ValueError(
                    f"group_samples(freq=True) needs feature_metadata columns "
                    f"{missing}. Compute per-group frequencies first, e.g. "
                    "pt.add_freq(groups={'LUAD': luad_subset, ...})."
                )

        # Wider title column so multi-line rotated group titles (e.g.
        # "Large\n(>20kb)") sit clear of the heatmap's gene tick labels.
        ftitle_col = add_col(5.0) if show_ft else None
        left_cols = [add_col(t.size) for t in left]
        # feature-label gutter: between the left tracks and the matrix, so gene
        # names hug the matrix instead of overflowing a left track.
        fgut_col = add_col(feature_gutter_size) if show_feature_labels else None
        samp_cols = []
        sfreq_cols: list[int | None] = []
        for si, (_lbl, pos) in enumerate(ssecs):
            samp_cols.append(add_col(main_w * len(pos) / n_samp))
            sfreq_cols.append(add_col(scfg["freq_size"]) if sfreq else None)
            if si < gs_ - 1:
                add_col(samp_gap)
        # keep the overall freq bar (a right track) clear of the per-section strips
        if sfreq and right:
            add_col(samp_gap)
        right_cols = [add_col(t.size) for t in right]
        if legend_pad > 0:
            add_col(legend_pad)
        legend_col = add_col(legend_width)

        self.gs = GridSpec(
            len(height_ratios), len(width_ratios), width_ratios=width_ratios,
            height_ratios=height_ratios, wspace=wspace, hspace=hspace, figure=fig,
        )

        # main matrix sub-axes (gf x gs)
        first_ax = None
        for fi, (_flbl, fpos) in enumerate(fsecs):
            for si, (_slbl, spos) in enumerate(ssecs):
                ax = fig.add_subplot(self.gs[feat_rows[fi], samp_cols[si]])
                # gene names are drawn in the feature gutter, not the matrix
                sub = main_track.subset(feat=fpos, samp=spos)
                sub.render(ax)
                first_ax = first_ax or ax
                ax.set_xticks([])  # sample names live in the bottom gutter
        self.ax_heatmap = first_ax

        # top / bottom tracks (sample sections)
        for ti, t in enumerate(top):
            for si, (_slbl, spos) in enumerate(ssecs):
                ax = fig.add_subplot(self.gs[top_rows[ti], samp_cols[si]])
                # render owns the value-axis limits (incl. BarTrack grow/shared
                # scale); only the first section keeps the value axis visible.
                t.subset(samp=spos).render(ax)
                if si > 0:
                    ax.set_ylabel("")
                    ax.set_yticks([])
                    ax.set_title("")  # keep the track title on the first section only
                    ax.spines["left"].set_visible(False)
        for ti, t in enumerate(bottom):
            for si, (_slbl, spos) in enumerate(ssecs):
                ax = fig.add_subplot(self.gs[bottom_rows[ti], samp_cols[si]])
                t.subset(samp=spos).render(ax)
                if si > 0:
                    ax.set_yticks([])
                    ax.set_ylabel("")
                    ax.set_title("")  # keep the track title on the first section only
                ax.set_xticks([])  # sample names live in the bottom gutter

        # left / right tracks (feature sections)
        for ti, t in enumerate(left):
            for fi, (_flbl, fpos) in enumerate(fsecs):
                fig_ax = fig.add_subplot(self.gs[feat_rows[fi], left_cols[ti]])
                t.subset(feat=fpos).render(fig_ax)
                if fi < gf - 1:
                    fig_ax.set_xticks([])
        for ti, t in enumerate(right):
            for fi, (_flbl, fpos) in enumerate(fsecs):
                ax = fig.add_subplot(self.gs[feat_rows[fi], right_cols[ti]])
                t.subset(feat=fpos).render(ax)
                if fi < gf - 1:
                    ax.set_xticks([])

        # label gutters: one per section, labels sliced to the section
        if fgut_col is not None:
            for fi, (_flbl, fpos) in enumerate(fsecs):
                ax = fig.add_subplot(self.gs[feat_rows[fi], fgut_col])
                self._draw_label_gutter(
                    ax, [feature_labels[i] for i in fpos], axis="y", n=len(fpos)
                )
        if sgut_row is not None:
            for si, (_slbl, spos) in enumerate(ssecs):
                ax = fig.add_subplot(self.gs[sgut_row, samp_cols[si]])
                self._draw_label_gutter(
                    ax, [sample_labels[i] for i in spos], axis="x", n=len(spos)
                )
                self._sample_gutter_axes.append(ax)

        # per-section sample frequency strips (group_samples(freq=True)):
        # each section's own freq column, on a shared 0-1 scale for comparison.
        if sfreq:
            for si, (slbl, _spos) in enumerate(ssecs):
                strip = FreqTrack(
                    self.feature_metadata[[f"{slbl}{sfreq_suffix}"]],
                    line_color=self.line_color,
                    annot=scfg["freq_annot"],
                    vmin=0.0,
                    vmax=1.0,
                )
                for fi, (_flbl, fpos) in enumerate(fsecs):
                    ax = fig.add_subplot(self.gs[feat_rows[fi], sfreq_cols[si]])
                    strip.subset(feat=fpos).render(ax)
                    if fi < gf - 1:
                        ax.set_xticks([])

        # group titles
        if show_st:
            x, ha = {
                "start": (0.0, "left"), "center": (0.5, "center"), "end": (1.0, "right"),
            }[scfg["title_align"]]
            for si, (slbl, _spos) in enumerate(ssecs):
                ax = fig.add_subplot(self.gs[stitle_row, samp_cols[si]])
                ax.axis("off")
                ax.text(
                    x, 0.0, str(slbl), ha=ha, va="bottom",
                    fontsize=scfg["title_fontsize"], fontweight="bold",
                )
        if show_ft:
            y, va = {
                "start": (1.0, "top"), "center": (0.5, "center"), "end": (0.0, "bottom"),
            }[fcfg["title_align"]]
            for fi, (flbl, _fpos) in enumerate(fsecs):
                ax = fig.add_subplot(self.gs[feat_rows[fi], ftitle_col])
                ax.axis("off")
                ax.text(
                    0.25, y, str(flbl), rotation=90, ha="center", va=va,
                    fontsize=fcfg["title_fontsize"], fontweight="bold",
                )

        # legend spans the matrix rows
        self.ax_legend = fig.add_subplot(
            self.gs[feat_rows[0]:feat_rows[-1] + 1, legend_col]
        )

    def plot_bar(
        self,
        fontsize: int = 6,
        bar_value: bool = False,
        bar_col: str = "TMB",
        ylabel_size: int = 8,
    ) -> OncoPlot:
        """
        Plot bar chart showing values (typically TMB) for each sample.

        Legacy shorthand for the canonical ``add_bar()``; prefer ``add_bar()``
        in new code.

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
        annot_fontsize: int = 8,
        linewidths: float = 1,
        xtick_fontsize: int = 9,
    ) -> OncoPlot:
        """
        Plot frequency heatmap showing mutation frequencies for each gene.

        Legacy shorthand for the canonical ``add_freq()``; prefer ``add_freq()``
        in new code.

        Parameters
        ----------
        freq_columns : list, default ["freq"]
            List of frequency columns to display
        annot_fontsize : int, default 8
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
            Dictionary mapping columns to ``{category: color}`` mappings, e.g.::

                {
                    "subtype": {"LUAD": "orange", "LUSC": "blue", "ASC": "green"},
                    "smoke": {"is_smoke": "gray", "no_smoke": "white"},
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
        """Legacy: restyle the auto-drawn sample-name gutter labels.

        Sample names are now rendered in a dedicated bottom gutter block by
        ``render()`` (pass ``sample_labels=`` to override them, or
        ``show_sample_labels=`` to force on/off). This shim is kept for
        backward compatibility: it re-applies ``rotation`` / ``fontsize`` to the
        already-drawn gutter labels. Call it after ``render()``.

        Returns
        -------
        self : OncoPlot
            Returns self for method chaining
        """
        for ax in getattr(self, "_sample_gutter_axes", []):
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)
                if fontsize is not None:
                    label.set_fontsize(fontsize)
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

        Legacy shorthand for the canonical ``main(kind="cnv")``; prefer
        ``main(kind="cnv")`` in new code.

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
