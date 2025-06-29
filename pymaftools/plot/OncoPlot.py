import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib import cm, ticker
from matplotlib.colors import ListedColormap, Normalize
from ..core.PivotTable import PivotTable
from .ColorManager import ColorManager

class OncoPlot:
    def __init__(self, pivot_table: PivotTable, **kwargs):
        # load PivotTable
        self.pivot_table = pivot_table
        self.feature_metadata = pivot_table.feature_metadata
        self.sample_metadata = pivot_table.sample_metadata
        
        # initialize ColorManager
        self.color_manager = ColorManager()
        
        # initialize legend dictionary
        self.legend_dict = {}

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
        num_categorical = len(self.categorical_columns)
        num_numeric = len(self.numeric_columns)
        height_ratios = [1, 20] + [1] * num_categorical + [1] * num_numeric

        # make sure only one figure is created
        plt.close("all")
        self.fig = plt.figure(figsize=self.figsize)
        self.gs = plt.GridSpec(
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

    def add_legend(self, legend_name: str, color_dict: dict):
        """
        添加圖例信息到 legend_dict
        
        Args:
            legend_name: 圖例名稱，如 'mutation', 'sex', 'subtype' 等
            color_dict: 顏色映射字典，如 {'M': 'blue', 'F': 'red'}
        """
        self.legend_dict[legend_name] = color_dict
        return self
    
    def plot_all_legends(self, fontsize=8, title_fontsize=10, legend_spacing=0.08, item_spacing=0.02):
        """
        在 ax_legend 上繪製所有圖例
        
        Args:
            fontsize: 圖例字體大小
            title_fontsize: 圖例標題字體大小
            legend_spacing: 不同圖例之間的間距
            item_spacing: 同一圖例內項目之間的間距
        """
        if not self.legend_dict:
            return self
            
        self.ax_legend.clear()
        self.ax_legend.axis('off')
        self.ax_legend.set_xlim(0, 1)
        self.ax_legend.set_ylim(0, 1)
        
        # 從上往下繪製圖例，起始位置更接近頂部
        y_position = 0.95
        
        for legend_name, color_dict in self.legend_dict.items():
            # 繪製圖例標題
            self.ax_legend.text(0.05, y_position, legend_name, 
                               fontsize=title_fontsize, fontweight='bold', 
                               va='top', ha='left')
            y_position -= 0.05  # 標題後的間距縮小
            
            # 繪製圖例項目
            for label, color in color_dict.items():
                # 繪製顏色方塊（縮小尺寸，無邊框）
                rect = Rectangle((0.05, y_position - 0.015), 0.04, 0.03, 
                               facecolor=color, edgecolor='none', linewidth=0)
                self.ax_legend.add_patch(rect)
                
                # 繪製標籤文字
                self.ax_legend.text(0.12, y_position, label, 
                                   fontsize=fontsize, va='center', ha='left')
                
                y_position -= 0.035  # 項目間距縮小
            
            # 添加圖例之間的間距（縮小）
            y_position -= 0.03
            
        return self

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
            ax.set_yticklabels(data.index, rotation=0, fontsize=self.ytick_fontsize)  # Set labels horizontally
        return self

    def heatmap_rectangle(self, show_frame=False, n=3, cmap=None, table=None, width=1, height=1, line_color="white"):
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
    
    def heatmap(self, cmap_dict=None, linecolor="white", linewidth=1, show_frame=False, n=3, yticklabels=True):
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
            
            # 使用 ColorManager 生成顏色映射
            column_cmap = cmap_dict.get(col, {}) if cmap_dict else {}
            if not column_cmap:
                column_cmap = self.color_manager.generate_categorical_cmap(
                    data.iloc[0], 
                    default_palette=default_cmap
                )
            
            # 使用 categorical_heatmap 方法
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

            # 添加文字註解（如果需要）
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

            # 設定軸標籤和刻度
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
        ax.set_yticklabels(color_matrix.index, rotation=0, fontsize=ytick_fontsize)  # Set labels horizontally
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
        
        try:
            cbar.outline.set_visible(False)
        except AttributeError:
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
        oncoplot = OncoPlot(pivot_table=pivot_table, figsize=figsize, width_ratios=width_ratios)
        oncoplot.heatmap()
        oncoplot.plot_freq()
        oncoplot.plot_bar()
        oncoplot.plot_all_legends()  # 繪製所有圖例
        oncoplot.add_xticklabel()
        return oncoplot
    
    def save(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight', transparent: bool = False, **kwargs):
        if self.fig is None:
            print("Figure is not exist.")
            return

        try:
            format = filename.split('.')[-1].lower()
            pil_kwargs = {"compression": "tiff_lzw"} if format == "tiff" else {}

            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
                format=format,
                pil_kwargs=pil_kwargs,
                **kwargs
            )
            print(f"Figure saved to: {filename}")
        except Exception as e:
            print(f"Error while saving figure: {e}")
