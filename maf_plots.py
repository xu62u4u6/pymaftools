import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Union

target_col = [
        "Hugo_Symbol",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2"
    ]


def create_oncoplot(pivot_table, 
                    color_map=None, 
                    mutation_counts : Union[bool, pd.Series]=True, 
                    figsize=(18, 16),
                    wspace=0.5, 
                    hspace=0.01, 
                    freq_columns=["freq"], 
                    ax_main_range=(0, 24), 
                    ax_freq_range=(24, 28), 
                    ax_legend_range=(29, 31),
                    square=False,
                    show_frame=False,
                    bar_annot_fontsize=7):
    
    # freq_columns = freq_columns or [f"{sample_type}_freq" for sample_type in ["A", "T", "S"]] + ['all_freq']
    heatmap_data = pivot_table#sorted_df.drop(columns=freq_columns)
    freq_data = pivot_table.gene_metadata[freq_columns].values
    
    # 預設的顏色映射
    color_map = color_map or {
        'False': '#FFFFFF',          # 白色 (無突變)
        'Missense_Mutation': 'gray',  # 淺灰色
        'Frame_Shift_Ins':'#FF4500',       # 較深色紅  
        'Frame_Shift_Del': '#4682B4',       # 較深色藍
        'In_Frame_Ins': '#FF707A',    # 淺色紅
        'In_Frame_Del':'#ADD8E6',    # 淺色藍
        'Nonsense_Mutation': '#90EE90',  # 低飽和度綠色
        'Splice_Site': '#CB704D',        # 低飽和度咖啡色
        'Multi_Hit': '#000000',           # 黑色 (多重突變)
        "Silent": "#eeeeee",
        "3'UTR": "#bbbbcc",
        "5'UTR": "#bbbbcc",
        "IGR": "#bbbbcc",
        "Intron": "#bbbbcc",
        "RNA": "#bbbbcc",
    }

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 32, height_ratios=[1, 12], wspace=wspace, hspace=hspace)
    
    if mutation_counts is not None:
        if mutation_counts == True:
            mutation_counts = pivot_table.sample_metadata.mutations_count.values
        ax_bar = fig.add_subplot(gs[0, ax_main_range[0]:ax_main_range[1]])     # Bar chart
        plot_bar(ax_bar, mutation_counts, fontsize=bar_annot_fontsize)
    else:
        ax_bar = None  # 如果沒有 bar chart, 不繪製上方區域
    
    ax_main = fig.add_subplot(gs[1, ax_main_range[0]:ax_main_range[1]])    # Main heatmap
    ax_freq = fig.add_subplot(gs[1, ax_freq_range[0]:ax_freq_range[1]])   # Frequency heatmap
    ax_legend = fig.add_subplot(gs[1, ax_legend_range[0]:ax_legend_range[1]]) # Legend

    plot_heatmap(ax_main, heatmap_data, color_map, square=square, show_frame=show_frame)
    plot_freq(ax_freq, freq_data, freq_columns, square=square)
    plot_legend(ax_legend, color_map)

    ax_main.set_xlabel("Mutations")
    #if mutation_counts is None:
        #gs.tight_layout(fig, rect=[0, 0, 1, 0.9])  # 調整繪圖佈局，避免空白區域過多
    #else:
        #plt.tight_layout()

    
def plot_bar(ax_bar, mutation_counts, fontsize=6):    

    x = np.arange(len(mutation_counts))
    width = 0.95

    # Create bars
    tmbs = np.where(mutation_counts == 0, 0, mutation_counts/40)
    ax_bar.bar(x, tmbs, width=width, color='gray', edgecolor='white')
    
    # Set x-axis limits to exactly match the heatmap
    # The -0.5 ensures the bars align perfectly with heatmap cells
    ax_bar.set_xlim(-0.5, len(mutation_counts) - 0.5)
    
    # 在柱子上添加數值標籤
    for i, tmb in enumerate(tmbs):
        ax_bar.text(i, tmb + 2, f"{tmb:.1f}", ha='center', fontsize=fontsize)

    # 隐藏柱状图的边框和刻度
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(True)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.set_xticks([])
    ax_bar.set_xlabel('TMB')


def plot_heatmap(ax_main, heatmap_data, color_map, linecolor="white", square=True, show_frame=False):

    # 創建數值映射
    def color_encode(val):
        return color_map.get(val, '#FFFFFF')

    # 轉換數據
    data_matrix = heatmap_data.map(color_encode)

    # 創建熱圖
    sns.heatmap(
        heatmap_data.notna(),
        cmap=['white', 'grey'],  # 使用白色和灰色表示數據存在與否
        cbar=False,
        linewidths=1,
        linecolor=linecolor,
        ax=ax_main,
        square=square
    )
    ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0)

    # 添加顏色
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            ax_main.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                fill=True,
                facecolor=data_matrix.iloc[i, j],
                edgecolor=linecolor,
                lw=1
            ))

     # 添加每三個樣本的淺色框
    if show_frame:
        for i in range(0, heatmap_data.shape[1], 3):  # 每三個樣本
            rect = Rectangle((i, -0.5), 3, heatmap_data.shape[0] + 1,
                            linewidth=1, edgecolor='lightgray', facecolor='none')
            ax_main.add_patch(rect)

def plot_freq(ax_freq, freq_data, freq_columns, square=True, show_frame=True):
    # 繪製頻率熱圖
    sns.heatmap(freq_data,
                cmap='Blues',
                linewidths=0.5,
                ax=ax_freq,
                cbar=False,  # 不顯示頻率熱圖的colorbar
                vmin=0,
                vmax=freq_data.max(),
                alpha=0.8,
                square=square)  # 根據頻率數據的最大值設置vmax

    # 隱藏頻率熱圖的索引
    ax_freq.set_xticks([])  # 隱藏 x 軸的標籤
    ax_freq.set_yticks([])  # 隱藏 y 軸的標籤

    # 設置頻率熱圖的標籤和數值，並隱藏索引
    for i in range(freq_data.shape[0]):  # 每行
        for j in range(freq_data.shape[1]):  # 每列
            value = freq_data[i, j]
            color = 'black' if value < 0.6 * freq_data.max() else 'white'  # 高频率用白色，低频率用黑色
            ax_freq.text(j + 0.5, i + 0.5, f"{value:.2f}",
                         va='center', ha='center', color=color)
            
    ax_freq.set_title('Frequency', pad=20)  # 頻率熱圖的標題
    ax_freq.set_xticks(np.arange(len(freq_columns))+0.5)  # 設置 x 軸刻度數量
    ax_freq.set_xticklabels(freq_columns, rotation=90)  # 設置 x 軸標籤並旋轉90度

def plot_legend(ax_legend, color_map):
    # 修正圖例
    legend_elements = [Rectangle((0, 0), 1, 1, color=color_map[key], label=key) for key in color_map.keys()]
    ax_legend.legend(handles=legend_elements, title="Variant Types", loc='center', fontsize='small', frameon=False)
    ax_legend.axis('off')  # 隱藏圖例軸的坐標系