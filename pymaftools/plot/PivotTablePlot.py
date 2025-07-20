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
            self.fig, ax = plt.subplots(figsize=(10, 6))
        else:
            self.fig = ax.figure

        if title is None:
            title = f"Boxplot of {test_col} by {group_col}"

        gb = data.groupby(group_col)
        if order is None:
            group_pairs = list(combinations(gb.groups.keys(), 2))
        else:
            group_pairs = list(combinations(order, 2))

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

        # Save figure
        if save_path is not None:
            self.save(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        return ax
    
    def plot_pca_samples(
        self,
        color_col: str = "subtype",
        shape_col: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        to_binary: bool = False,
        palette: Optional[Union[str, Dict[str, str]]] = None,
        alpha: float = 0.8,
        title: Optional[str] = None,
        titlesize: int = 14,
        is_numeric: bool = False,
        save_path: Optional[str] = None,
        fontsize: int = 12,
        width_ratios: Tuple[int, int] = (4, 1),
        legend_item_spacing: float = 0.04,
        legend_group_spacing: float = 0.06,
        dpi: int = 300,
        s: int = 60
    ) -> Tuple[pd.DataFrame, np.ndarray, PCA]:
        """
        Plot PCA scatter plot using ColorManager and LegendManager for unified styling.

        This method leverages the full power of ColorManager for color generation
        and LegendManager for consistent legend appearance across all plot types.
        Provides better integration with the overall plotting architecture.

        Parameters
        ----------
        Same as plot_pca_samples method.

        Returns
        -------
        tuple
            Same as plot_pca_samples method.

        Notes
        -----
        This method uses ColorManager for automatic color generation and LegendManager 
        for legend rendering, providing the most consistent styling with other plot 
        types in the package.

        Examples
        --------
        >>> # Basic PCA plot with unified legend style
        >>> pca_df, variance, pca_obj = pivot_table.plot.plot_pca_samples_with_legend_manager(
        ...     color_col="subtype"
        ... )
        
        >>> # PCA with both color and shape encoding using ColorManager
        >>> pca_df, variance, pca_obj = pivot_table.plot.plot_pca_samples_with_legend_manager(
        ...     color_col="case_ID",
        ...     shape_col="sample_type",
        ...     palette="tab20"  # ColorManager will handle the conversion
        ... )
        """
        from .LegendManager import LegendManager
        from .ColorManager import ColorManager
        
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
        self.fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios, figure=self.fig)
        ax_pca = self.fig.add_subplot(gs[0])
        ax_legend = self.fig.add_subplot(gs[1])

        # Initialize managers
        legend_manager = LegendManager(ax_legend)
        color_manager = ColorManager()

        # Set up palette/colormap
        if palette is None:
            palette = "viridis" if is_numeric else "Set1"

        # Check if data is actually numeric
        is_actually_numeric = is_numeric and pd.api.types.is_numeric_dtype(pca_result_df[color_col])
        
        # Show warning if user specified is_numeric=True but data is not numeric
        if is_numeric and not is_actually_numeric:
            print(f"Warning: Column '{color_col}' contains non-numeric data but is_numeric=True. "
                  f"Treating as categorical data instead.")
        
        if is_actually_numeric:
            # Numeric color encoding with colormap
            if isinstance(palette, dict):
                raise ValueError("For numeric data, palette should be a colormap name (str), not a dictionary")
            
            scatter = ax_pca.scatter(
                pca_result_df["PC1"],
                pca_result_df["PC2"], 
                c=pca_result_df[color_col],
                cmap=palette,
                alpha=alpha,
                s=s
            )
            
            # Use LegendManager but modify it to create custom colorbar
            legend_manager.add_numeric_colorbar(
                legend_name=color_col,
                scatter_obj=scatter,
                target_ax=ax_pca,
                label=color_col,
                orientation='vertical', 
                fraction=0.08, 
                pad=0.02
            )
            # Hide the legend axis since we're using colorbar on the plot
            ax_legend.set_visible(False)
            
        else:
            # Categorical color encoding using ColorManager
            unique_colors = pca_result_df[color_col].unique()
            
            # Generate colors using ColorManager
            if isinstance(palette, dict):
                # Use provided color mapping directly
                color_palette = palette
            elif isinstance(palette, str):
                # Use ColorManager to generate colors from colormap name
                color_palette = color_manager.generate_cmap_from_list(
                    categories=unique_colors.tolist(), 
                    cmap_name=palette, 
                    as_hex=True
                )
            else:
                # Fallback to seaborn palette
                colors = sns.color_palette("Set1", len(unique_colors))
                color_palette = {str(cat): color for cat, color in zip(unique_colors, colors)}
            
            # Handle shape encoding
            if shape_col is not None:
                unique_shapes = pca_result_df[shape_col].unique()
                shape_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(unique_shapes)]
                shape_dict = dict(zip(unique_shapes, shape_markers))
                
                # Plot with both color and shape
                for color_val in unique_colors:
                    for shape_val in unique_shapes:
                        mask = (pca_result_df[color_col] == color_val) & (pca_result_df[shape_col] == shape_val)
                        subset = pca_result_df[mask]
                        if len(subset) > 0:
                            ax_pca.scatter(
                                subset["PC1"], subset["PC2"],
                                c=[color_palette.get(str(color_val), 'gray')], 
                                marker=shape_dict[shape_val],  # type: ignore
                                alpha=alpha, s=60
                            )
                
                # Add legends using LegendManager
                # Ensure colors are in hex format for LegendManager
                color_legend_dict = {}
                for val, color in color_palette.items():
                    if isinstance(color, (tuple, list, np.ndarray)):
                        # Convert RGB/RGBA to hex
                        import matplotlib.colors as mcolors
                        color_legend_dict[str(val)] = mcolors.to_hex(color)
                    else:
                        color_legend_dict[str(val)] = str(color)
                
                legend_manager.add_legend(color_col, color_legend_dict)
                legend_manager.add_shape_legend(shape_col, {str(k): str(v) for k, v in shape_dict.items()})
                legend_manager.plot_pca_legends(
                    color_legend=color_col, 
                    shape_legend=shape_col, 
                    fontsize=fontsize,
                    legend_item_spacing=legend_item_spacing,
                    legend_group_spacing=legend_group_spacing
                )
                
            else:
                # Color only
                for color_val in unique_colors:
                    mask = pca_result_df[color_col] == color_val
                    subset = pca_result_df[mask]
                    if len(subset) > 0:
                        ax_pca.scatter(
                            subset["PC1"], subset["PC2"],
                            c=[color_palette.get(str(color_val), 'gray')], 
                            alpha=alpha, s=s
                        )
                
                # Add color legend using LegendManager
                color_legend_dict = {}
                for val, color in color_palette.items():
                    if isinstance(color, (tuple, list, np.ndarray)):
                        import matplotlib.colors as mcolors
                        color_legend_dict[str(val)] = mcolors.to_hex(color)
                    else:
                        color_legend_dict[str(val)] = str(color)
                
                legend_manager.add_legend(color_col, color_legend_dict)
                legend_manager.plot_pca_legends(color_legend=color_col, fontsize=fontsize)

        # Set PCA plot properties
        if title:
            ax_pca.set_title(title, fontsize=titlesize)
        ax_pca.set_xlabel(
            f"Principal Component 1 ({explained_variance[0] * 100:.2f}%)", fontsize=fontsize)
        ax_pca.set_ylabel(
            f"Principal Component 2 ({explained_variance[1] * 100:.2f}%)", fontsize=fontsize)

        # Save figure using BasePlot's save method
        if save_path is not None:
            self.save(save_path, dpi=dpi)
                    
        plt.tight_layout()
        plt.show()
        return pca_result_df, explained_variance, pca
