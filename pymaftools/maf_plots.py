import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from .maf_utils import PivotTable


class OncoPlot:
    all_mutation_cmap = {
        'False': '#FFFFFF', 
        'Missense_Mutation': 'gray', 
        'Frame_Shift_Ins':'#FF4500',     # Dark red  
        'Frame_Shift_Del': '#4682B4',    # Dark blue
        'In_Frame_Ins': '#FF707A',       # Light red
        'In_Frame_Del':'#ADD8E6',        # Light blue
        'Nonsense_Mutation': '#90EE90',  # Low-saturation green
        'Splice_Site': '#CB704D',        # Low-saturation brown
        'Multi_Hit': '#000000',          # Black (multiple mutations)
        "Silent": "#eeeeee",             # Light gray
        "3'UTR": "#bbbbcc",              # Light purple
        "5'UTR": "#bbbbcc",              # Light purple
        "IGR": "#bbbbcc",                # Light purple
        "Intron": "#bbbbcc",             # Light purple
        "RNA": "#bbbbcc",                # Light purple
    }

    nonsynymous_cmap = {
        'False': '#FFFFFF', 
        'Missense_Mutation': 'gray', 
        'Frame_Shift_Ins':'#FF4500',     # Dark red
        'Frame_Shift_Del': '#4682B4',    # Dark blue
        'In_Frame_Ins': '#FF707A',       # Light red
        'In_Frame_Del':'#ADD8E6',        # Light blue
        'Nonsense_Mutation': '#90EE90',  # Low-saturation green
        'Splice_Site': '#CB704D',        # Low-saturation brown
        'Multi_Hit': '#000000',          # Black (multiple mutations)
    }

    cnv_cmp = {"AMP": "salmon", "DEL": "steelblue", "AMP&DEL": "gray"}
    
    def __init__(self, pivot_table: PivotTable, **kwargs):
        # load pivottable
        self.pivot_table = pivot_table
        self.gene_metadata = pivot_table.gene_metadata
        self.sample_metadata = pivot_table.sample_metadata

        self.set_config(**kwargs)
        self.update_layout()

    def set_config(self, 
                   line_color: str = "white", 
                   cmap: dict = nonsynymous_cmap, 
                   figsize=(20, 15), 
                   width_ratios=[20, 1, 1.5], 
                   height_ratios=[1, 20], 
                   wspace=0.015, 
                   hspace=0.02, 
                   categorical_columns=[], 
                   numeric_columns=[]):
        
        self.line_color = line_color
        self.cmap = cmap
        self.figsize = figsize
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.wspace = wspace
        self.hspace = hspace
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

    def update_layout(self):
        num_categorical = len(self.categorical_columns)
        num_numeric = len(self.numeric_columns)
        height_ratios = [1, 20] + [1] * num_categorical + [1] * num_numeric

        self.fig = plt.figure(figsize=self.figsize)
        self.gs = plt.GridSpec(
            2 + num_categorical + num_numeric, 
            3, 
            width_ratios=self.width_ratios, 
            height_ratios=height_ratios, 
            wspace=self.wspace, 
            hspace=self.hspace
        )

        self.ax_bar = self.fig.add_subplot(self.gs[0, 0])
        self.ax_heatmap = self.fig.add_subplot(self.gs[1, 0])
        self.ax_heatmap_legend = self.fig.add_subplot(self.gs[1, 2])
        self.ax_freq = self.fig.add_subplot(self.gs[1, 1])
        self.axs_categorical_columns = {col: self.fig.add_subplot(self.gs[2+i, 0]) for i, col in enumerate(self.categorical_columns)}
        self.axs_numeric_columns = {col: self.fig.add_subplot(self.gs[2+len(self.categorical_columns)+i, 0]) for i, col in enumerate(self.numeric_columns)}

    def plot_numeric_metadata(self, annotate=False, annotation_font_size=10, fmt=".2f", cmap="Blues"):
        for col, ax in self.axs_numeric_columns.items():
            data = self.sample_metadata[[col]].T  # Ensure you pass a DataFrame
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
            )
            
            ax.set_yticks([i + 0.5 for i in range(len(data.index))])  # Shift the ticks by +0.5
            ax.set_yticklabels(data.index, rotation=0)  # Set labels horizontally

    def heatmap(self, show_frame=False, n=3):
        def color_encode(val):
            return self.cmap.get(val, '#ffffff')
        color_matrix = self.pivot_table.map(color_encode)
        
        self.plot_color_heatmap(self.ax_heatmap, 
                        color_matrix,
                        linecolor=self.line_color,
                        linewidth=1,
                        xticklabels=False
                        )
        
        # add frame every n columns
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

        # add legend
        legend_elements = [Rectangle((0, 0), 1, 1, color=self.cmap[key], label=key) for key in self.cmap.keys()]
        self.ax_heatmap_legend.legend(handles=legend_elements, title="Variant Types", loc='center', fontsize='small', frameon=False)
        self.ax_heatmap_legend.axis('off')
        
    def plot_bar(self, tmb=None, fontsize=6, bar_value=False):    
        tmb = tmb or self.sample_metadata.TMB
        x = np.arange(len(tmb))
        width = 0.95

        self.ax_bar.bar(x, tmb, width=width, color='gray', edgecolor='white')
        self.ax_bar.set_xlim(-0.5, len(tmb) - 0.5)
        if bar_value:
            for i, tmb_value in enumerate(tmb):
                self.ax_bar.text(i, tmb_value + 2, f"{tmb:.1f}", ha='center', fontsize=fontsize)

        self.ax_bar.spines['left'].set_visible(True)

        self.ax_bar.spines['top'].set_visible(False)
        self.ax_bar.spines['right'].set_visible(False)
        self.ax_bar.spines['bottom'].set_visible(False)
        self.ax_bar.set_xticks([])
        self.ax_bar.set_ylabel('TMB')

    def plot_freq(self, freq_columns=["freq"]):
        freq_data = self.gene_metadata[freq_columns]
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

    @staticmethod
    def categorical_cmap(data, cmap="pastel"):
        unique_categories = pd.unique(data.values.ravel())
        palette = sns.color_palette(cmap, len(unique_categories))
        color_dict = {category: palette[i] for i, category in enumerate(unique_categories)}

        # map the categories to their respective colors
        color_matrix = data.map(lambda x: color_dict.get(x, '#ffffff'))
        return color_matrix
    
    def plot_categorical_metadata(self, annotate=False, annotation_font_size=10, annotate_text_color="black"):
        for col, ax in self.axs_categorical_columns.items():
            data = self.sample_metadata[[col]].T  # Ensure you pass a DataFrame
            color_matrix = self.categorical_cmap(data)
            self.plot_color_heatmap(ax, 
                color_matrix=color_matrix,
                linecolor=self.line_color,
                linewidth=1,
                xticklabels=False,
                yticklabels=list(data.index),
                )

            if annotate:
                for i in range(color_matrix.shape[0]):
                    for j in range(color_matrix.shape[1]):
                        ax.text(
                            j + 0.5, i + 0.5,  # Center the text
                            f"{data.iloc[i, j]}",  # Display the color value or any desired text
                            ha='center', va='center',
                            fontsize=annotation_font_size,
                            color=annotate_text_color
                    )


            ax.set_yticks([i + 0.5 for i in range(len(color_matrix.index))])  # Shift the ticks by +0.5
            ax.set_yticklabels(color_matrix.index, rotation=0)  # Set labels horizontally

    @staticmethod
    def plot_color_heatmap(ax, 
                           color_matrix: pd.DataFrame, 
                           linecolor='white', 
                           linewidth=1, 
                           xticklabels=True, 
                           yticklabels=True,
                           ):
        
        ones_matrix = color_matrix.copy()
        ones_matrix[:] = 0 
        ones_matrix = ones_matrix.astype(float)

        # Plot background heatmap (using ones matrix to hold size)
        sns.heatmap(
            ones_matrix,
            cbar=False,
            linewidths=linewidth,
            linecolor=linecolor,
            ax=ax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cmap="Blues",
            
        )

        # Overlay color matrix
        for i in range(color_matrix.shape[0]):
            for j in range(color_matrix.shape[1]):
                ax.add_patch(Rectangle(
                    (j, i), 1, 1,
                    fill=True,
                    facecolor=color_matrix.iloc[i, j],
                    edgecolor=linecolor,
                    lw=linewidth
                ))

        ax.set_yticks([i + 0.5 for i in range(len(color_matrix.index))])  # Shift the ticks by +0.5
        ax.set_yticklabels(color_matrix.index, rotation=0)  # Set labels horizontally
        return ax

    def add_xticklabel(self):
        # get the maximum row number
        max_row = max([spec.rowspan.stop for spec in self.gs]) - 1

        # find target axis
        target_ax = None
        for ax in self.fig.axes:
            subplotspec = ax.get_subplotspec()
            if subplotspec.rowspan.start == max_row and subplotspec.colspan.start == 0:
                target_ax = ax
                break

        # add xtick labels and xticks
        if target_ax:
            target_ax.set_xticks([i + 0.5 for i in range(len(self.sample_metadata))])
            target_ax.set_xticklabels(self.sample_metadata.index, rotation=90)

