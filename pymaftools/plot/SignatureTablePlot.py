"""
SignatureTablePlot Module

Plotting functionality specific to SignatureTable objects.
Reachable via ``signature_table.plot``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .PivotStatsPlot import PivotStatsPlot


class SignatureTablePlot(PivotStatsPlot):
    """
    Plotting interface for SignatureTable objects.

    Inherits the shared plotters from :class:`PivotStatsPlot` (``oncoplot``,
    ``plot_pca_samples``, ``plot_boxplot_with_annot``) and adds the
    signature-specific exposure landscape (:meth:`stacked_bar`).

    A :class:`~pymaftools.core.SignatureTable.SignatureTable` is oriented
    signatures (rows) × samples (columns), with each value the activity /
    exposure of that signature in that sample.
    """

    def stacked_bar(
        self,
        *,
        normalize: bool = True,
        ax=None,
        figsize: tuple[float, float] = (20, 6),
        colormap: str = "tab20",
        legend: bool = True,
        width: float = 0.9,
    ):
        """Per-sample signature exposure as a stacked bar chart.

        Samples run along the x-axis; each bar stacks its signature
        contributions (one colour per signature). This is the canonical
        "signature landscape" view of an exposure matrix.

        Parameters
        ----------
        normalize : bool, default True
            Scale each sample's bar to sum to 1 (relative contributions). When
            False, bars use the raw exposure totals (variable bar heights).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure/axes is created when None.
        figsize : tuple, default (20, 6)
            Figure size, used only when ``ax`` is None.
        colormap : str, default "tab20"
            Matplotlib colormap mapping signatures to colours.
        legend : bool, default True
            Draw the signature legend (placed outside, upper-right).
        width : float, default 0.9
            Bar width passed to the pandas bar plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the bars were drawn on.

        Examples
        --------
        >>> signature_table.plot.stacked_bar()                 # relative
        >>> signature_table.plot.stacked_bar(normalize=False)  # raw exposures
        """
        # SignatureTable is signatures × samples; transpose so each row is a
        # sample (a bar) and each column a signature (a stacked segment). Wrap
        # in a plain DataFrame first: PivotTable overrides ``.plot`` with the
        # accessor, which would otherwise shadow pandas' bar plotting.
        df = pd.DataFrame(self.pivot_table).T.astype(float)
        if normalize:
            totals = df.sum(axis=1)
            df = df.div(totals.where(totals != 0, 1), axis=0)

        from matplotlib.colors import to_hex

        from . import style

        # Explicit colour per signature, so the stacked bars and the legend card
        # share one mapping.
        cmap = plt.get_cmap(colormap)
        sig_colors = {
            sig: to_hex(cmap(i % cmap.N)) for i, sig in enumerate(df.columns)
        }

        legend_ax = None
        if ax is None:
            self.fig, ax, legend_ax = style.fig_with_legend(figsize, legend_width=0.12)
        else:
            self.fig = ax.figure

        df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[sig_colors[s] for s in df.columns],
            width=width,
            legend=False,
        )
        ax.set_xlabel("sample")
        ax.set_ylabel("relative exposure" if normalize else "exposure")
        ax.margins(x=0)
        style.style_axes(ax)
        if legend and legend_ax is not None:
            style.draw_legend_cards(legend_ax, {"signature": sig_colors})
        elif legend:
            ax.legend(
                title="signature",
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                frameon=False,
            )
        return ax
