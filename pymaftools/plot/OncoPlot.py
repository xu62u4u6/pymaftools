import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib import cm, ticker
from ..core.PivotTable import PivotTable

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
        self.feature_metadata = pivot_table.feature_metadata
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

    def plot_numeric_metadata(self, annotate=False, annotation_font_size=10, fmt=".2f", cmap="Blues", cmap_dict=None, alpha=1):
        for col, ax in self.axs_numeric_columns.items():
            cmap = cmap_dict.get(col, "Blues") if cmap_dict else "Blues"
            data = self.sample_metadata[[col]].T 
            # set vmin and vmax if coolwarm cmap 
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
            ax.set_yticklabels(data.index, rotation=0)  # Set labels horizontally

    def heatmap(self, show_frame=False, n=3, cmap=None, table=None, width=1, height=1, line_color="white"):
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
                        height=height
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

        self.ax_bar.spines['left'].set_visible(True) # True !!!

        self.ax_bar.spines['top'].set_visible(False)
        self.ax_bar.spines['right'].set_visible(False)
        self.ax_bar.spines['bottom'].set_visible(False)
        self.ax_bar.set_xticks([])
        self.ax_bar.set_ylabel('TMB')

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

    @staticmethod
    def categorical_cmap(data, cmap_dict=None, default_cmap="pastel"):
        """
        Function to map categories in `data` to specific colors based on `cmap_dict`.
        If no mapping is found for a category, it applies the default colormap.

        Parameters:
        - data: DataFrame or Series containing categorical data.
        - cmap_dict: Dictionary of custom color mappings, such as:
        {"subtype": {"LUAD": "orange", "LUSC": "blue", "ASC": "green"},
        "smoke": {"is_smoke": "gray", "no_smoke": "white"}}
        - default_cmap: Default colormap if no custom mapping is found.
        
        Returns:
        - color_matrix: A DataFrame or Series of color values based on the mappings.
        """
        # Set default colormap
        if cmap_dict is None:
            cmap_dict = {}

        # Get unique categories in the data
        unique_categories = pd.unique(data.values.ravel())

        # Create a color palette for categories not found in cmap_dict
        palette = sns.color_palette(default_cmap, len(unique_categories))
        default_color_dict = {category: palette[i] for i, category in enumerate(unique_categories)}

        color_dict = default_color_dict.copy()  # Start with the default color map

        # Process each key in cmap_dict and update the color_dict with custom mappings
        for key, category_dict in cmap_dict.items():
            for category, color in category_dict.items():
                color_dict[category] = color  # Override the default color with the custom one

        # Map the categories to their respective colors, falling back to default if necessary
        color_matrix = data.map(lambda x: color_dict.get(x, '#ffffff'))  # Default to white if no mapping

        return color_matrix
    
    def plot_categorical_metadata(self, annotate=False, cmap_dict=None, alpha=1.0, default_cmap="pastel", annotation_font_size=10, annotate_text_color="black"):
        """
        cmap_dict = {
        "subtype": {
            "LUAD": "orange",
            "LUSC": "blue",
            "ASC": "green"
        },
        "smoke": {
            "is_smoke": "gray",
            "no_smoke": "white"
        },
        """
        for col, ax in self.axs_categorical_columns.items():
            data = self.sample_metadata[[col]].T  # Ensure you pass a DataFrame
            color_matrix = self.categorical_cmap(data, cmap_dict=cmap_dict, default_cmap=default_cmap)
            self.plot_color_heatmap(ax, 
                color_matrix=color_matrix,
                linecolor=self.line_color,
                linewidth=1,
                xticklabels=False,
                yticklabels=list(data.index),
                alpha=alpha
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

            ax.set_xticks([])
            ax.set_yticks([i + 0.5 for i in range(len(color_matrix.index))])  # Shift the ticks by +0.5
            ax.set_yticklabels(color_matrix.index, rotation=0)  # Set labels horizontally
            ax.set_xlabel("")  # Hide x-axis label if it exists
            ax.tick_params(axis='x', which='both', bottom=False, top=False)  # Hide x ticks completely

    @staticmethod
    def plot_color_heatmap(ax, 
                        color_matrix: pd.DataFrame, 
                        linecolor='white', 
                        linewidth=1, 
                        xticklabels=False, 
                        yticklabels=True,
                        alpha=1.0,
                        width=1.0, 
                        height=1.0):
        
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
                    (j + (1 - width) / 2, i + (1 - height) / 2),  # 調整 x 和 y，讓方格置中
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

    def numeric_heatmap(self, cmap="RdBu"): 
        ax = self.ax_heatmap
        table = self.pivot_table
        
        # calculate vmin and vmax
        vextreme = max(abs(table.min().min()), abs(table.max().max()))

        # Create the heatmap without a colorbar first
        hm = sns.heatmap(table, ax=ax, cmap=cmap, cbar=False,
                        vmin=-vextreme, vmax=vextreme, center=0)
        
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_yticks([i + 0.5 for i in range(len(table.index))])
        ax.set_yticklabels(table.index, rotation=0)
        
        # Create a colorbar
        norm = plt.Normalize(vmin=-vextreme, vmax=vextreme)
        
        cbar = self.fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.ax_freq,
            ticks=np.linspace(-vextreme, vextreme, 5),
            shrink=0.7
        )
        cbar.ax.set_aspect(18)
        cbar.outline.set_visible(False)
        
        cbar.ax.yaxis.set_tick_params(color="gray")
        cbar.ax.yaxis.set_tick_params(labelcolor="black")
        cbar.ax.tick_params(labelsize=10, length=6, width=1)
        
        # Formatter
        if vextreme < 0.01 or vextreme > 1000:
            cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        else:
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        self.ax_heatmap_legend.axis('off')
        self.ax_bar.axis('off')

    @staticmethod
    def default_oncoplot(pivot_table, figsize=(30, 15), width_ratios=[20, 1, 2]):
        oncoplot = OncoPlot(pivot_table=pivot_table, figsize=figsize, width_ratios=width_ratios)
        oncoplot.heatmap()
        oncoplot.plot_freq()
        oncoplot.plot_bar()
        oncoplot.add_xticklabel()
        return oncoplot
    
    def save_figure(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight', transparent: bool = False, **kwargs):
        if self.fig is None:
            print("Figure is not exist.")
            return

        try:
            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
                **kwargs
            )
            print(f"Figure saved to: {filename}")
        except Exception as e:
            print(f"Error while saving figure: {e}")
