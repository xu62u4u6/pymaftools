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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator

# Local imports
from .BasePlot import BasePlot

class PivotTablePlot(BasePlot):
    """
    Plotting functionality for PivotTable objects.
    
    This class provides a clean interface for various visualization methods
    while keeping plotting logic separate from the core PivotTable class.
    Inherits from BasePlot to provide legend management and figure saving capabilities.
    
    Parameters
    ----------
    pivot_table : PivotTable
        The PivotTable instance to create plots for.
        
    Examples
    --------
    >>> # Using property accessor (recommended)
    >>> pivot_table.plot.plot_pca_samples(color_col="subtype")
    >>> pivot_table.plot.plot_boxplot_with_annot(test_col="TMB")
    
    >>> # Direct instantiation (not recommended)
    >>> plotter = PivotTablePlot(pivot_table)
    >>> plotter.plot_pca_samples(color_col="subtype")
    """
    
    def __init__(self, pivot_table):
        """
        Initialize PivotTablePlot with a PivotTable instance.
        
        Parameters
        ----------
        pivot_table : PivotTable
            The PivotTable instance to create plots for.
        """
        super().__init__()  # Initialize BasePlot
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

        # Save figure using inherited method
        if save_path is not None:
            self.save_figure(fig, save_path, dpi)

        return ax
    
    def plot_pca_samples(
        self,
        color_col: str = "subtype",
        shape_col: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        to_binary: bool = False,
        palette: Optional[Union[str, Dict[str, str]]] = None,
        alpha: float = 0.8,
        title: str = "PCA of samples",
        is_numeric: bool = False,
        save_path: Optional[str] = None,
        fontsize: int = 12,
        titlesize: int = 14,
        width_ratios: Tuple[int, int] = (4, 1),
        legend_item_spacing: float = 0.04,
        legend_group_spacing: float = 0.06
    ) -> Tuple[pd.DataFrame, np.ndarray, PCA]:
        """
        Plot PCA scatter plot of samples with color and shape encoding.

        Performs Principal Component Analysis on the PivotTable data and creates
        a scatter plot showing the first two principal components. Samples are
        colored by one variable and optionally shaped by another variable.
        Uses GridSpec layout with legend displayed in a separate axis.

        Parameters
        ----------
        color_col : str, default "subtype"
            Column name in sample_metadata to use for coloring points.
        shape_col : str, optional
            Column name in sample_metadata to use for point shapes.
            If None, all points use the same shape.
        figsize : tuple of int, default (12, 6)
            Figure size as (width, height) in inches.
        to_binary : bool, default False
            Whether to convert data to binary (0/1) before PCA.
            Useful for mutation data where only presence/absence matters.
        palette : str or dict, optional
            Color palette for the plot. Can be:
            - seaborn palette name (e.g., "Set1", "viridis") for categorical data
            - dictionary mapping values to colors for categorical data
            - colormap name (e.g., "viridis", "plasma") for numeric data
        alpha : float, default 0.8
            Transparency level for scatter points (0-1).
        title : str, default "PCA of samples"
            Plot title.
        is_numeric : bool, default False
            Whether color_col contains numeric values.
            If True, uses colormap; if False, uses discrete colors.
        save_path : str, optional
            Path to save the figure. Format determined by file extension.
        fontsize : int, default 12
            Font size for axis labels.
        titlesize : int, default 14
            Font size for plot title.
        width_ratios : tuple of int, default (4, 1)
            Width ratio between PCA plot and legend axis.
        legend_item_spacing : float, default 0.04
            Vertical spacing between items within the same legend group.
        legend_group_spacing : float, default 0.06
            Vertical spacing between different legend groups (color vs shape).

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
        ...     color_col="subtype"
        ... )

        >>> # PCA with both color and shape encoding
        >>> pivot_table.plot.plot_pca_samples(
        ...     color_col="subtype",
        ...     shape_col="sex",
        ...     palette={"LUAD": "orange", "ASC": "green", "LUSC": "blue"},
        ...     title="PCA with Subtype and Sex Encoding"
        ... )

        >>> # Numeric coloring with colormap
        >>> pivot_table.plot.plot_pca_samples(
        ...     color_col="age",
        ...     is_numeric=True,
        ...     palette="viridis"
        ... )
        """
        # Calculate PCA result
        pca_result_df, explained_variance, pca = self.pivot_table.PCA(to_binary=to_binary)

        # Ensure color_col exists in sample_metadata
        if color_col not in self.pivot_table.sample_metadata.columns:
            raise ValueError(
                f"Column '{color_col}' not found in sample_metadata.")
        pca_result_df[color_col] = self.pivot_table.sample_metadata[color_col]
        
        # Add shape_col if specified
        if shape_col is not None:
            if shape_col not in self.pivot_table.sample_metadata.columns:
                raise ValueError(
                    f"Column '{shape_col}' not found in sample_metadata.")
            pca_result_df[shape_col] = self.pivot_table.sample_metadata[shape_col]

        # Create GridSpec layout for PCA plot and legend
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios, figure=fig)
        ax_pca = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[1])

        # Set up palette/colormap
        if palette is None:
            palette = "viridis" if is_numeric else "Set1"

        if is_numeric:
            # Numeric color encoding with colormap
            if isinstance(palette, dict):
                raise ValueError("For numeric data, palette should be a colormap name (str), not a dictionary")
            
            scatter = ax_pca.scatter(
                pca_result_df["PC1"],
                pca_result_df["PC2"], 
                c=pca_result_df[color_col],
                cmap=str(palette),
                alpha=alpha,
                s=60  # marker size
            )
            # Add colorbar to legend axis
            cbar = plt.colorbar(scatter, cax=ax_legend)
            cbar.set_label(color_col, fontsize=fontsize)
            
        else:
            # Categorical color encoding
            unique_colors = pca_result_df[color_col].unique()
            
            # Handle shape encoding
            if shape_col is not None:
                unique_shapes = pca_result_df[shape_col].unique()
                shape_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(unique_shapes)]
                shape_dict = dict(zip(unique_shapes, shape_markers))
                
                # Get colors for color_col
                if isinstance(palette, dict):
                    # Use provided color mapping
                    color_palette = palette
                else:
                    # Generate colors using seaborn palette
                    colors = sns.color_palette(palette, len(unique_colors))
                    color_palette = dict(zip(unique_colors, colors))
                
                # Plot with both color and shape
                for color_val in unique_colors:
                    for shape_val in unique_shapes:
                        mask = (pca_result_df[color_col] == color_val) & (pca_result_df[shape_col] == shape_val)
                        subset = pca_result_df[mask]
                        if len(subset) > 0:
                            ax_pca.scatter(
                                subset["PC1"], subset["PC2"],
                                c=[color_palette.get(color_val, 'gray')], 
                                marker=shape_dict[shape_val],  # type: ignore
                                alpha=alpha, s=60,
                                # Don't add labels here - we'll create separate legends
                            )
                
                # Create separate legends in the legend axis
                ax_legend.axis('off')
                
                # Color legend (top half)
                legend_y_start = 0.9
                ax_legend.text(0.05, 0.95, color_col, fontsize=fontsize, fontweight='bold', 
                              transform=ax_legend.transAxes)
                
                y_pos = legend_y_start
                for color_val in unique_colors:
                    # Color rectangle
                    rect_color = Rectangle((0.05, y_pos-0.015), 0.06, 0.025, 
                                         facecolor=color_palette.get(color_val, 'gray'),
                                         transform=ax_legend.transAxes)
                    ax_legend.add_patch(rect_color)
                    # Color label
                    ax_legend.text(0.15, y_pos, str(color_val), fontsize=fontsize-1, 
                                  va='center', transform=ax_legend.transAxes)
                    y_pos -= legend_item_spacing
                
                # Shape legend (bottom half)
                shape_y_start = y_pos - legend_group_spacing
                ax_legend.text(0.05, shape_y_start + 0.03, shape_col, fontsize=fontsize, fontweight='bold',
                              transform=ax_legend.transAxes)
                
                y_pos = shape_y_start
                for shape_val in unique_shapes:
                    # Shape marker
                    ax_legend.scatter([0.08], [y_pos], marker=shape_dict[shape_val],  # type: ignore
                                    c='black', s=40, transform=ax_legend.transAxes)
                    # Shape label
                    ax_legend.text(0.15, y_pos, str(shape_val), fontsize=fontsize-1, 
                                  va='center', transform=ax_legend.transAxes)
                    y_pos -= legend_item_spacing
                
            else:
                # Color only
                if isinstance(palette, dict):
                    colors = [palette.get(val, 'gray') for val in unique_colors]
                    color_palette = palette
                else:
                    colors = sns.color_palette(palette, len(unique_colors))
                    color_palette = dict(zip(unique_colors, colors))
                
                for color_val in unique_colors:
                    mask = pca_result_df[color_col] == color_val
                    subset = pca_result_df[mask]
                    if len(subset) > 0:
                        ax_pca.scatter(
                            subset["PC1"], subset["PC2"],
                            c=[color_palette[color_val]], alpha=alpha, s=60,
                            label=f"{color_val}"
                        )
                
                # Create single legend in the legend axis
                ax_legend.axis('off')
                handles, labels = ax_pca.get_legend_handles_labels()
                legend = ax_legend.legend(handles, labels, loc='upper left', 
                           title=color_col, title_fontsize=fontsize, fontsize=fontsize-1,
                           labelspacing=0.3, handletextpad=0.5, handlelength=1.0)

        # Set PCA plot properties
        ax_pca.set_title(title, fontsize=titlesize)
        ax_pca.set_xlabel(
            f"Principal Component 1 ({explained_variance[0] * 100:.2f}%)", fontsize=fontsize)
        ax_pca.set_ylabel(
            f"Principal Component 2 ({explained_variance[1] * 100:.2f}%)", fontsize=fontsize)

        # Save figure using inherited method
        if save_path is not None:
            self.save_figure(fig, save_path)
        
        plt.tight_layout()
        plt.show()
        return pca_result_df, explained_variance, pca
