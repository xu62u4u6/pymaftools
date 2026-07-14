"""
ExpressionTablePlot Module

Plotting functionality specific to ExpressionTable objects,
including volcano plots for differential expression results.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from .PivotStatsPlot import PivotStatsPlot


class ExpressionTablePlot(PivotStatsPlot):
    """
    Plotting functionality for ExpressionTable objects.

    Inherits all PivotStatsPlot methods (PCA, boxplot, etc.) and adds
    expression-specific visualisations such as volcano plots.

    Parameters
    ----------
    expression_table : ExpressionTable
        The ExpressionTable instance to create plots for.
    """

    def volcano(
        self,
        padj: float = 0.05,
        log2fc: float = 1.0,
        top_n: int = 10,
        label_col: str = "gene_name",
        ax: Optional[Axes] = None,
        figsize: tuple = (8, 6),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> Axes:
        """
        Draw a volcano plot from DESeq2 results in ``feature_metadata``.

        Parameters
        ----------
        padj : float, default 0.05
            Adjusted p-value threshold for significance.
        log2fc : float, default 1.0
            Absolute log2 fold-change threshold.
        top_n : int, default 10
            Number of top DEGs to label.
        label_col : str, default "gene_name"
            Column in ``feature_metadata`` used for gene labels.
        ax : Axes, optional
            Matplotlib axes to draw on. Created if *None*.
        figsize : tuple, default (8, 6)
            Figure size when creating a new figure.
        save_path : str, optional
            Path to save the figure.
        dpi : int, default 150
            Resolution for saved figure.

        Returns
        -------
        Axes
        """
        fm = self.pivot_table.feature_metadata.copy()
        if "padj" not in fm.columns or "log2FoldChange" not in fm.columns:
            raise ValueError("Run .deseq2() before plotting a volcano.")

        fm = fm.dropna(subset=["padj", "log2FoldChange"])
        fm["-log10padj"] = -np.log10(fm["padj"].clip(lower=1e-300))

        # Classify genes
        fm["_deg"] = "NS"
        up = (fm["padj"] < padj) & (fm["log2FoldChange"] > log2fc)
        down = (fm["padj"] < padj) & (fm["log2FoldChange"] < -log2fc)
        fm.loc[up, "_deg"] = "Up"
        fm.loc[down, "_deg"] = "Down"

        from . import style
        from .ColorManager import ColorManager

        palette = {
            "NS": style.MUTED,
            "Up": ColorManager.FUNCTIONAL_CMAP["Truncating"],  # house red
            "Down": style.ACCENT,  # house blue
        }

        if ax is None:
            self.fig, ax = plt.subplots(figsize=figsize)
        else:
            self.fig = ax.figure

        sns.scatterplot(
            data=fm,
            x="log2FoldChange",
            y="-log10padj",
            hue="_deg",
            hue_order=["Down", "NS", "Up"],
            palette=palette,
            s=8,
            alpha=0.6,
            edgecolor="none",
            ax=ax,
        )

        # Threshold lines
        ax.axhline(-np.log10(padj), ls="--", lw=0.8, color=style.SPINE_COLOR)
        ax.axvline(log2fc, ls="--", lw=0.8, color=style.SPINE_COLOR)
        ax.axvline(-log2fc, ls="--", lw=0.8, color=style.SPINE_COLOR)

        # Label top genes
        if label_col in fm.columns and top_n > 0:
            sig = fm.loc[fm["_deg"] != "NS"].nsmallest(top_n, "padj")
            for _, row in sig.iterrows():
                ax.text(
                    row["log2FoldChange"],
                    row["-log10padj"],
                    row[label_col],
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10(padj)")
        ax.legend(title="", frameon=False)
        style.style_axes(ax)

        self.fig.tight_layout()

        if save_path is not None:
            self.save(save_path, dpi=dpi)

        return ax
