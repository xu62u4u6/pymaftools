import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib import cm, ticker
from matplotlib.colors import ListedColormap, Normalize
from .BasePlot import BasePlot

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
        
        Parameters:
        -----------
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

        self.set_config(**kwargs)

    def set_config(self, 
                   line_color: str = "white", 
                   cmap: str = "nonsynonymous",
                   figsize=(20, 15), 
                   width_ratios=[25, 1, 1, 2], 
                   height_ratios=[1, 20], 
                   wspace=0.015, 
                   hspace=0.02, 
                   categorical_columns=[], 
                   numeric_columns=[],
                   ytick_fontsize=10):
        """
        Configure OncoPlot appearance and layout settings.
        
        Parameters:
        -----------
        line_color : str, default "white"
            Color of lines between heatmap cells
        cmap : str or dict, default "nonsynonymous"
            Color map for mutation types
        figsize : tuple, default (20, 15)
            Figure size (width, height)
        width_ratios : list, default [25, 1, 1, 2]
            Width ratios for subplot columns
        height_ratios : list, default [1, 20]
            Height ratios for subplot rows
        wspace : float, default 0.015
            Width spacing between subplots
        hspace : float, default 0.02
            Height spacing between subplots
        categorical_columns : list, default []
            List of categorical metadata columns to display
        numeric_columns : list, default []
            List of numeric metadata columns to display
        ytick_fontsize : int, default 10
            Font size for y-axis tick labels
            
        Returns:
        --------
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
        self.update_layout()
        return self

    def update_layout(self):
        """
        Update the subplot layout based on configured metadata columns.
        
        Creates a GridSpec layout with appropriate dimensions for the main heatmap,
        TMB bar plot, frequency plot, legend area, and metadata annotations.
        """
        num_categorical = len(self.categorical_columns)
        num_numeric = len(self.numeric_columns)
        height_ratios = [1, 20] + [1] * num_categorical + [1] * num_numeric

        # Make sure only one figure is created
        plt.close("all")
        self.fig = plt.figure(figsize=self.figsize)
        self.gs = GridSpec(
            2 + num_categorical + num_numeric, 
            4, 
            width_ratios=self.width_ratios, 
            height_ratios=height_ratios, 
            wspace=self.wspace, 
            hspace=self.hspace
        )

        self.ax_bar = self.fig.add_subplot(self.gs[0, 0])
        self.ax_heatmap = self.fig.add_subplot(self.gs[1, 0])
        self.ax_legend = self.fig.add_subplot(self.gs[1, 3])
        self.ax_freq = self.fig.add_subplot(self.gs[1, 1])
        self.axs_categorical_columns = {col: self.fig.add_subplot(self.gs[2+i, 0]) for i, col in enumerate(self.categorical_columns)}
        self.axs_numeric_columns = {col: self.fig.add_subplot(self.gs[2+len(self.categorical_columns)+i, 0]) for i, col in enumerate(self.numeric_columns)}

    def plot_numeric_metadata(self, annotate=False, annotation_font_size=10, fmt=".2f", cmap="Blues", cmap_dict=None, alpha=1):
        """
        Plot numeric metadata as heatmaps below the main mutation heatmap.
        
        Parameters:
        -----------
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
            
        Returns:
        --------
        self : OncoPlot
            Returns self for method chaining
        """
        for col, ax in self.axs_numeric_columns.items():
            cmap = cmap_dict.get(col, "Blues") if cmap_dict else "Blues"
            data = self.sample_metadata[[col]].T 
            # Set vmin and vmax if coolwarm cmap 
            if cmap == "coolwarm":
                vextreme = max(abs(data.min().min()), abs(data.max().max()))
                vmin, vmax = -vextreme, vextreme 
            else:
                vmin, vmax = None, None  

            sns.heatmap(
                data,
                cmap=cmap,
                cbar=False,
                linewidths=1,
                linecolor=self.line_color,
                ax=ax,  
                xticklabels=False,
                yticklabels=list(data.index),
                annot=annotate,  # Enable/disable annotation
                fmt = fmt if annotate else "",  # Format to 2 decimal places if enabled
                annot_kws ={"size": annotation_font_size} if annotate else None,  # Font size for annotations
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_yticks([i + 0.5 for i in range(len(data.index))])  # Shift the ticks by +0.5
            ax.set_yticklabels(data.index, rotation=0, fontsize=self.ytick_fontsize)  # Set labels horizontally
        return self

    def heatmap_rectangle(self, show_frame=False, n=3, cmap=None, table=None, width=1, height=1, line_color="white"):
        """
        Plot mutation heatmap using colored rectangles for each mutation type.
        
        Parameters:
        -----------
        show_frame : bool, default False
            Whether to show frames around groups of columns
        n : int, default 3
            Number of columns per frame group
        cmap : dict, optional
            Color mapping for mutation types
        table : DataFrame, optional
            Mutation table to plot (defaults to self.pivot_table)
        width : float, default 1
            Width of rectangles (0-1)
        height : float, default 1
            Height of rectangles (0-1)
        line_color : str, default "white"
            Color of lines between rectangles
            
        Returns:
        --------
        self : OncoPlot
            Returns self for method chaining
        """
        if table is None:
            table = self.pivot_table
        if cmap is None:
            cmap = self.cmap
        
        def color_encode(val):
            return cmap.get(val, "#ffffff")
        color_matrix = table.map(color_encode)
        
        self.plot_color_heatmap(self.ax_heatmap, 
                        color_matrix,
                        linecolor=line_color,
                        linewidth=1,
                        xticklabels=False,
                        width=width,
                        height=height,
                        ytick_fontsize=self.ytick_fontsize,)
        
        # Add frame every n columns
        if show_frame:
            for i in range(0, color_matrix.shape[1], n): 
                rect = Rectangle(
                    (i, -0.5), 
                    n,
                    color_matrix.shape[0] + 1,
                    linewidth=1,
                    edgecolor='lightgray',
                    facecolor='none'
                )
                self.ax_heatmap.add_patch(rect)

        self.add_legend("Variant Types", self.cmap)
        return self

    @staticmethod
    def categorical_heatmap(table, category_cmap, ax=None, fig_size=(10, 6), unknown_color="white", linecolor="white", **kwargs):
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
        sns.heatmap(table_mapped, cmap=cmap, cbar=False, ax=ax, linecolor=linecolor, linewidths=0.5, **kwargs)

        # perpare legend info
        legend_info = list(category_cmap.items())
        if has_unknown:
            legend_info.append(("Unknown", unknown_color))

        return fig, ax, legend_info
    
    def mutation_heatmap(self, cmap_dict=None, linecolor="white", linewidth=1, show_frame=False, n=3, yticklabels=True):
        """
        Plot the main mutation heatmap using categorical color coding.
        
        Parameters:
        -----------
        cmap_dict : dict, optional
            Color mapping for mutation types
        linecolor : str, default "white"
            Color of lines between cells
        linewidth : int, default 1
            Width of lines between cells
        show_frame : bool, default False
            Whether to show frames around groups of columns
        n : int, default 3
            Number of columns per frame group
        yticklabels : bool, default True
            Whether to show y-axis labels (gene names)
            
        Returns:
        --------
        self : OncoPlot
            Returns self for method chaining
        """
        if cmap_dict is None:
            cmap_dict = self.cmap

        fig, ax, legend_info = self.categorical_heatmap(table=self.pivot_table, 
                                                        category_cmap=cmap_dict, 
                                                        linecolor=linecolor, 
                                                        linewidth=linewidth,
                                                        ax=self.ax_heatmap,
                                                        vmin=0, # Ensure mapping uses full range
                                                        vmax=len(cmap_dict))
        
        ax.set_xticks([])
        ax.set_xlabel("")
        if yticklabels:
            ax.set_yticks([i + 0.5 for i in range(len(self.pivot_table.index))])
            ax.set_yticklabels(self.pivot_table.index, rotation=0, fontsize=self.ytick_fontsize)

        # Show frame every `n` columns
        if show_frame:
            for i in range(0, len(self.pivot_table.columns), n): 
                rect = Rectangle(
                    (i, -0.5),  # X, y
                    n,  # width
                    len(self.pivot_table) + 1,  # height
                    linewidth=1,
                    edgecolor='lightgray',
                    facecolor='none'
                )
                self.ax_heatmap.add_patch(rect)
        
        mutation_legend = {key: cmap_dict[key] for key in cmap_dict.keys() if key != "Unknown"}
        self.add_legend("Mutation", mutation_legend)
        
        return self
         
    def plot_bar(self, fontsize=6, bar_value=False, bar_col="TMB", ylabel_size=8):
        """
        Plot bar chart showing values (typically TMB) for each sample.
        
        Parameters:
        -----------
        fontsize : int, default 6
            Font size for bar value annotations
        bar_value : bool, default False
            Whether to show values on top of bars
        bar_col : str, default "TMB"
            Column name in sample_metadata to use for bar values
        ylabel_size : int, default 8
            Font size for y-axis label
            
        Returns:
        --------
        self : OncoPlot
            Returns self for method chaining
        """
        if bar_col == "TMB" and bar_col not in self.sample_metadata.columns:
            raise ValueError(f"Column '{bar_col}' not found in sample metadata. Please do table.calculate_tmb() first.")
        if bar_col not in self.sample_metadata.columns:
            raise ValueError(f"Column '{bar_col}' not found in sample metadata.")
        bar_values = self.sample_metadata[bar_col].values
        x = np.arange(len(bar_values))
        width = 0.95

        self.ax_bar.bar(x, bar_values, width=width, color='gray', edgecolor='white')
        self.ax_bar.set_xlim(-0.5, len(bar_values) - 0.5)
        if bar_value:
            for i, tmb_value in enumerate(bar_values):
                self.ax_bar.text(i, tmb_value + 2, f"{bar_values:.1f}", ha='center', fontsize=fontsize)

        self.ax_bar.spines['left'].set_visible(True) # True !!!

        self.ax_bar.spines['top'].set_visible(False)
        self.ax_bar.spines['right'].set_visible(False)
        self.ax_bar.spines['bottom'].set_visible(False)
        self.ax_bar.set_xticks([])
        self.ax_bar.set_ylabel(bar_col, fontsize=ylabel_size)
        return self

    def plot_freq(self, freq_columns=["freq"]):
        freq_data = self.feature_metadata[freq_columns]
        sns.heatmap(
                freq_data,
                cbar=False,
                linewidths=1,
                linecolor=self.line_color,
                ax=self.ax_freq,
                xticklabels=freq_data.columns,
                yticklabels=False,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 9},
                cmap="Blues"
            )
        self.ax_freq.set_ylabel("")
        self.ax_freq.set_yticks([])  # hide y-axis
        return self

    def plot_categorical_metadata(self, annotate=False, cmap_dict=None, alpha=1.0, default_cmap="pastel", annotation_font_size=10, annotate_text_color="black"):
        """
        Plot categorical metadata as color-coded heatmaps below the main mutation heatmap.
        
        Parameters:
        -----------
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
            
        Returns:
        --------
        self : OncoPlot
            Returns self for method chaining
        """
        for col, ax in self.axs_categorical_columns.items():
            data = self.sample_metadata[[col]].T  # Ensure you pass a DataFrame
            
            # Use ColorManager to generate color mapping
            column_cmap = cmap_dict.get(col, {}) if cmap_dict else {}
            if not column_cmap:
                column_cmap = self.color_manager.generate_categorical_cmap(
                    data.iloc[0], 
                    default_palette=default_cmap
                )
            
            # Use categorical_heatmap method
            fig, ax, legend_info = self.categorical_heatmap(
                table=data,
                category_cmap=column_cmap,
                ax=ax,
                linecolor=self.line_color,
                linewidth=1,
                xticklabels=False,
                yticklabels=list(data.index),
                alpha=alpha
            )

            # Add text annotations (if needed)
            if annotate:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(
                            j + 0.5, i + 0.5,  # Center the text
                            f"{data.iloc[i, j]}",  # Display the actual value
                            ha='center', va='center',
                            fontsize=annotation_font_size,
                            color=annotate_text_color
                        )

            # Set axis labels and ticks
            ax.set_xticks([])
            ax.set_yticks([i + 0.5 for i in range(len(data.index))])
            ax.set_yticklabels(data.index, rotation=0, fontsize=self.ytick_fontsize)
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            
            self.add_legend(col, column_cmap)
            
        return self

    @staticmethod
    def plot_color_heatmap(ax, 
                        color_matrix: pd.DataFrame, 
                        linecolor='white', 
                        linewidth=1, 
                        xticklabels=False, 
                        yticklabels=True,
                        alpha=1.0,
                        width=1.0, 
                        height=1.0,
                        ytick_fontsize=10):
        """
        Plot a heatmap using colored rectangles based on a color matrix.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        color_matrix : pd.DataFrame
            DataFrame containing hex color codes for each cell
        linecolor : str, default 'white'
            Color of lines between cells
        linewidth : int, default 1
            Width of lines between cells
        xticklabels : bool, default False
            Whether to show x-axis labels
        yticklabels : bool, default True
            Whether to show y-axis labels
        alpha : float, default 1.0
            Transparency level (0-1)
        width : float, default 1.0
            Width of rectangles (0-1)
        height : float, default 1.0
            Height of rectangles (0-1)
        ytick_fontsize : int, default 10
            Font size for y-axis tick labels
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The modified axes
        """
        
        ones_matrix = color_matrix.copy()
        ones_matrix[:] = 0 
        ones_matrix = ones_matrix.astype(float)

        # Plot background heatmap (using ones_matrix to hold size)
        sns.heatmap(
            ones_matrix,
            cbar=False,
            linewidths=linewidth,
            linecolor=linecolor,
            ax=ax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cmap="Blues",
            alpha=0
        )

        for i in range(color_matrix.shape[0]):
            for j in range(color_matrix.shape[1]):
                face_color = color_matrix.iloc[i, j]
                if face_color == "#ffffff":
                    continue
                ax.add_patch(Rectangle(
                    (j + (1 - width) / 2, i + (1 - height) / 2),  # Adjust x and y to center the rectangle
                    width,  
                    height,
                    fill=True,
                    facecolor=face_color,
                    edgecolor=linecolor,
                    lw=linewidth,
                    alpha=alpha
                ))

        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_yticks([i + 0.5 for i in range(len(color_matrix.index))])  # Shift the ticks by +0.5
        ax.set_yticklabels(color_matrix.index, rotation=0, fontsize=ytick_fontsize)  # Set labels horizontally
        return ax

    def add_xticklabel(self):
        """
        Add x-axis tick labels to the bottom-most subplot.
        
        Finds the subplot in the bottom row and first column, then adds
        sample names as x-axis tick labels with 90-degree rotation.
        
        Returns:
        --------
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
                if subplotspec.rowspan.start == max_row and subplotspec.colspan.start == 0:
                    target_ax = ax
                    break
            except AttributeError:
                # Handle cases where get_subplotspec is not available
                continue

        # Add xtick labels and xticks
        if target_ax:
            target_ax.set_xticks([i + 0.5 for i in range(len(self.sample_metadata))])
            target_ax.set_xticklabels(self.sample_metadata.index, rotation=90)
        return self

    def numeric_heatmap(self, cmap="Blues", vmin=None, vmax=None, symmetric=False, yticklabels=True): 
        ax = self.ax_heatmap
        table = self.pivot_table
        
        # decide color range
        if vmin is None and vmax is None:
            if symmetric:
                vextreme = max(abs(table.min().min()), abs(table.max().max()))
                vmin = -vextreme
                vmax = vextreme
                center = 0
            else:
                vmin = table.min().min()
                vmax = table.max().max()
                center = (vmin + vmax) / 2
        elif vmin is None or vmax is None:
            raise ValueError("Both vmin and vmax must be specified.")

        else:
            center = 0
        # Draw heatmap
        hm = sns.heatmap(table, ax=ax, cmap=cmap, cbar=False,
                        vmin=vmin, vmax=vmax, center=center, yticklabels=yticklabels)

        ax.set_xticks([])
        ax.set_xlabel("")
        if yticklabels:
            ax.set_yticks([i + 0.5 for i in range(len(table.index))])
            ax.set_yticklabels(table.index, rotation=0, fontsize=self.ytick_fontsize)

        # Create colorbar
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar = self.fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.ax_freq,
            ticks=np.linspace(vmin, vmax, 5),
            shrink=0.5
            )
        cbar.ax.set_aspect(18)
        
        # Hide colorbar borders
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)
        
        cbar.ax.tick_params(labelsize=10, length=6, width=1)
        if yticklabels:
            cbar.ax.yaxis.set_tick_params(color="gray", labelcolor="black")
            # Format tick labels
            if max(abs(vmin), abs(vmax)) < 0.01 or max(abs(vmin), abs(vmax)) > 1000:
                cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            else:
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        self.ax_legend.axis('off')
        self.ax_bar.axis('off')
        return self

    @staticmethod
    def default_oncoplot(pivot_table, figsize=(30, 15), width_ratios=[20, 1, 2]):
        """
        Create a default oncoplot with standard configuration.
        
        This is a convenience method that creates an OncoPlot with commonly used
        settings and plots the main components (mutation heatmap, frequency plot,
        TMB bar plot, and legends).
        
        Parameters:
        -----------
        pivot_table : PivotTable
            The PivotTable instance containing mutation data
        figsize : tuple, default (30, 15)
            Figure size (width, height)
        width_ratios : list, default [20, 1, 2]
            Width ratios for subplot columns
            
        Returns:
        --------
        oncoplot : OncoPlot
            Configured and plotted OncoPlot instance
        """
        oncoplot = OncoPlot(pivot_table=pivot_table, figsize=figsize, width_ratios=width_ratios)
        oncoplot.mutation_heatmap()
        oncoplot.plot_freq()
        oncoplot.plot_bar()
        oncoplot.plot_all_legends()  # Plot all legends
        oncoplot.add_xticklabel()
        return oncoplot

