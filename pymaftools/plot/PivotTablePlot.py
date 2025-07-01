"""
PivotTablePlot Module

Dedicated plotting functionality for PivotTable objects.
Provides a clean interface for various visualization methods while keeping
plotting logic separate from the core PivotTable class.
"""

from __future__ import annotations

# Standard library imports
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator

class PivotTablePlot:
    """
    Plotting functionality for PivotTable objects.
    
    This class provides a clean interface for various visualization methods
    while keeping plotting logic separate from the core PivotTable class.
    
    Parameters
    ----------
    pivot_table : PivotTable
        The PivotTable instance to create plots for.
        
    Examples
    --------
    >>> # Using property accessor (recommended)
    >>> pivot_table.plot.plot_pca_samples(group_col="subtype")
    >>> pivot_table.plot.plot_boxplot_with_annot(test_col="TMB")
    
    >>> # Direct instantiation (not recommended)
    >>> plotter = PivotTablePlot(pivot_table)
    >>> plotter.plot_pca_samples(group_col="subtype")
    """
    
    def __init__(self, pivot_table):
        """
        Initialize PivotTablePlot with a PivotTable instance.
        
        Parameters
        ----------
        pivot_table : PivotTable
            The PivotTable instance to create plots for.
        """
        self.pivot_table = pivot_table
    
    def plot_boxplot_with_annot(
        self,
        data: Optional[pd.DataFrame] = None,
        group_col: str = "subtype",
        test_col: str = "mutations_count",
        palette: Optional[Union[str, Dict]] = None,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        test: str = 'Mann-Whitney',
        alpha: float = 0.8,
        order: Optional[List[str]] = None,
        fontsize: int = 12,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> Axes:
        """
        Create boxplot with statistical annotations.

        Creates a boxplot comparing groups with statistical significance testing
        and annotations. Supports pairwise comparisons between all groups using
        various statistical tests.

        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame containing the data to plot. If None, uses the PivotTable's
            sample_metadata.
        group_col : str, default "subtype"
            Column name in data to use for grouping samples.
        test_col : str, default "mutations_count"
            Column name in data containing values to plot on y-axis.
        palette : str or dict, optional
            Color palette for the boxplot. Can be a seaborn palette name or
            a dictionary mapping group values to colors.
        title : str, optional
            Plot title. If None, auto-generates title from column names.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure and axes.
        test : str, default 'Mann-Whitney'
            Statistical test for pairwise comparisons. Options include:
            'Mann-Whitney', 't-test_ind', 'Welch', etc.
        alpha : float, default 0.8
            Transparency level for boxplot fill colors (0-1).
        order : list of str, optional
            Order of groups on x-axis. If None, uses natural order.
        fontsize : int, default 12
            Font size for axis labels and tick labels.
        save_path : str, optional
            Path to save the figure. Format determined by file extension.
        dpi : int, default 300
            Resolution for saved figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.

        Notes
        -----
        This method uses statannotations library for statistical annotations.
        P-values are displayed as stars: * p<0.05, ** p<0.01, *** p<0.001.

        Examples
        --------
        >>> # Basic boxplot comparing TMB by subtype
        >>> pivot_table.plot.plot_boxplot_with_annot(
        ...     test_col="TMB",
        ...     group_col="subtype"
        ... )

        >>> # Custom styling with specific order and colors
        >>> pivot_table.plot.plot_boxplot_with_annot(
        ...     test_col="MSI",
        ...     group_col="subtype",
        ...     palette={"LUAD": "orange", "ASC": "green", "LUSC": "blue"},
        ...     order=["LUAD", "ASC", "LUSC"],
        ...     title="MSI Score by Cancer Subtype"
        ... )
        """
        # Use sample_metadata if no data provided
        if data is None:
            data = self.pivot_table.sample_metadata
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if title is None:
            title = f"Boxplot of {test_col} by {group_col}"

        gb = data.groupby(group_col)
        group_pairs = list(combinations(gb.groups.keys(), 2))

        ax.set_title(title, fontsize=fontsize)

        # boxplot
        boxplot = sns.boxplot(data=data, x=group_col, y=test_col,
                              ax=ax, hue=group_col, palette=palette, order=order)

        # alpha
        for patch in boxplot.patches:
            patch.set_facecolor((patch.get_facecolor()[0],  # R
                                patch.get_facecolor()[1],  # G
                                patch.get_facecolor()[2],  # B
                                alpha))

        # stat
        annotator = Annotator(ax=ax, pairs=group_pairs,
                              data=data, x=group_col, y=test_col, order=order)
        annotator.configure(test=test, text_format='star',
                            loc='inside', verbose=False)
        annotator.apply_and_annotate()

        # xticklabel with sample size
        sample_counts = gb.size().to_dict()
        xticks_labels = [
            f"{group} (n={sample_counts[group]})" for group in order]
        ax.set_xticklabels(xticks_labels, fontsize=fontsize)

        ax.set_xlabel(group_col, fontsize=fontsize)
        ax.set_ylabel(test_col, fontsize=fontsize)

        # save if path is given
        if save_path is not None:
            format_ext = save_path.split('.')[-1].lower()
            pil_kwargs = {
                "compression": "tiff_lzw"} if format_ext == "tiff" else {}
            fig.savefig(save_path, 
                        dpi=dpi, 
                        bbox_inches='tight',
                        pil_kwargs=pil_kwargs)
            print(f"[INFO] Figure saved to: {save_path}")

        return ax
    
    def plot_pca_samples(
        self,
        group_col: str = "subtype",
        figsize: Tuple[int, int] = (8, 6),
        to_binary: bool = False,
        palette_dict: Optional[Dict[str, str]] = None,
        alpha: float = 0.8,
        title: str = "PCA of samples",
        cmap: str = "summer",
        is_numeric: bool = False,
        save_path: Optional[str] = None,
        fontsize: int = 12,
        titlesize: int = 14
    ) -> Tuple[pd.DataFrame, np.ndarray, PCA]:
        """
        Plot PCA scatter plot of samples colored by group_col.

        Performs Principal Component Analysis on the PivotTable data and creates
        a scatter plot showing the first two principal components. Samples are
        colored by the specified grouping variable.

        Parameters
        ----------
        group_col : str, default "subtype"
            Column name in sample_metadata to use for coloring points.
        figsize : tuple of int, default (8, 6)
            Figure size as (width, height) in inches.
        to_binary : bool, default False
            Whether to convert data to binary (0/1) before PCA.
            Useful for mutation data where only presence/absence matters.
        palette_dict : dict, optional
            Dictionary mapping group values to colors.
            Example: {"LUAD": "orange", "ASC": "green", "LUSC": "blue"}
        alpha : float, default 0.8
            Transparency level for scatter points (0-1).
        title : str, default "PCA of samples"
            Plot title.
        cmap : str, default "summer"
            Colormap name for numeric group_col values.
        is_numeric : bool, default False
            Whether group_col contains numeric values.
            If True, uses colormap; if False, uses discrete colors.
        save_path : str, optional
            Path to save the figure. Format determined by file extension.
        fontsize : int, default 12
            Font size for axis labels.
        titlesize : int, default 14
            Font size for plot title.

        Returns
        -------
        tuple
            - pca_result_df : pd.DataFrame with PC1 and PC2 for each sample
            - explained_variance : np.ndarray of variance ratios for PC1 and PC2
            - pca : sklearn.decomposition.PCA fitted object

        Notes
        -----
        PCA is performed on the transposed data matrix where samples are rows
        and features (genes/mutations) are columns. Missing values should be
        handled before calling this method.

        Examples
        --------
        >>> # Basic PCA plot colored by subtype
        >>> pca_df, variance, pca_obj = pivot_table.plot.plot_pca_samples(
        ...     group_col="subtype"
        ... )

        >>> # Binary mutation data with custom colors
        >>> pivot_table.plot.plot_pca_samples(
        ...     group_col="subtype",
        ...     to_binary=True,
        ...     palette_dict={"LUAD": "orange", "ASC": "green", "LUSC": "blue"},
        ...     title="PCA of Binary Mutation Data"
        ... )

        >>> # Numeric grouping (e.g., age) with colormap
        >>> pivot_table.plot.plot_pca_samples(
        ...     group_col="age",
        ...     is_numeric=True,
        ...     cmap="viridis"
        ... )
        """
        # Calculate PCA result
        pca_result_df, explained_variance, pca = self.pivot_table.PCA(to_binary=to_binary)

        # Ensure group_col exists in sample_metadata
        if group_col not in self.pivot_table.sample_metadata.columns:
            raise ValueError(
                f"Column '{group_col}' not found in sample_metadata.")
        pca_result_df[group_col] = self.pivot_table.sample_metadata[group_col]

        # Plot PCA scatter plot
        plt.figure(figsize=figsize)

        if is_numeric:
            # Numeric → cmap
            scatter = plt.scatter(pca_result_df["PC1"],
                                  pca_result_df["PC2"],
                                  c=pca_result_df[group_col],
                                  cmap=cmap,
                                  alpha=alpha)
            plt.colorbar(scatter, label=group_col)  # Add colorbar manually
        else:
            # Categorical → palette
            scatter = sns.scatterplot(data=pca_result_df,
                                      x="PC1",
                                      y="PC2",
                                      hue=group_col,
                                      palette=palette_dict or "Set1",
                                      alpha=alpha)

        plt.title(title, fontsize=titlesize)
        plt.xlabel(
            f"Principal Component 1 ({explained_variance[0] * 100:.2f}%)", fontsize=fontsize)
        plt.ylabel(
            f"Principal Component 2 ({explained_variance[1] * 100:.2f}%)", fontsize=fontsize)

        if not is_numeric:
            plt.legend(title=group_col, title_fontsize=fontsize)
        if save_path:
            format_ext = save_path.split('.')[-1].lower()
            pil_kwargs = {
                "compression": "tiff_lzw"} if format_ext == "tiff" else {}

            plt.savefig(save_path,
                        dpi=300,
                        format=format_ext,
                        bbox_inches='tight',
                        pil_kwargs=pil_kwargs)
            print(f"[INFO] Figure saved to: {save_path}")
        plt.show()
        return pca_result_df, explained_variance, pca
