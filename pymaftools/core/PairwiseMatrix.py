import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu
from typing import Optional, Union, Tuple, List, Dict, Any
import matplotlib.gridspec as gridspec
import os


class PairwiseMatrix(pd.DataFrame):
    @property
    def _constructor(self):
        return PairwiseMatrix
    
class CooccurrenceMatrix(PairwiseMatrix):
    @property
    def _constructor(self):
        return CooccurrenceMatrix
    
class SimilarityMatrix(PairwiseMatrix):
    @property
    def _constructor(self):
        return SimilarityMatrix
    
    def get_group_similarity(self, groups, group_order=None):
        if group_order is None:
            group_order = groups.unique()

        result_df = pd.DataFrame(columns=group_order, index=group_order, dtype=float)
        n = len(group_order)
        for i in range(n):
            for j in range(n):
                cohort1, cohort2 = group_order[i], group_order[j]
                indices1, indices2 = np.where(groups == cohort1)[0], np.where(groups == cohort2)[0]
                pairwise_subset = self.iloc[indices1, indices2]
                result_df.loc[cohort1, cohort2] = pairwise_subset.mean().mean()
        return result_df

    def generate_permutation_list(self, groups, group_order, n_permutations=1000):
        """
        Generate a list of group similarity matrices under label permutations.
        """
        permutation_list = []
        for _ in range(n_permutations):
            shuffled_groups = groups.sample(frac=1, replace=False).reset_index(drop=True)
            permuted_similarity = self.get_group_similarity(shuffled_groups, group_order)
            permutation_list.append(permuted_similarity)
        return permutation_list

    @staticmethod
    def calculate_group_similarity_pvalues(true_group_similarity, permutation_list, group_order, tail="right"):
        """
        Calculate permutation p-values for each pairwise group similarity.
        """
        pvalues_df = pd.DataFrame(index=group_order, columns=group_order, dtype=float)
        
        for g1 in group_order:
            for g2 in group_order:
                true_val = true_group_similarity.loc[g1, g2]
                permuted_vals = np.array([perm.loc[g1, g2] for perm in permutation_list])
                
                if tail == "right":
                    pval = np.mean(permuted_vals >= true_val)
                elif tail == "left":
                    pval = np.mean(permuted_vals <= true_val)
                elif tail == "two":
                    diff = np.abs(true_val - permuted_vals.mean())
                    pval = np.mean(np.abs(permuted_vals - permuted_vals.mean()) >= diff)
                else:
                    raise ValueError("tail must be 'right', 'left', or 'two'")
                
                pvalues_df.loc[g1, g2] = pval
                
        return pvalues_df

    @staticmethod
    def plot_group_heatmap(
        result_df: pd.DataFrame, 
        title: str, 
        cmap: str = "Blues", 
        tick_size: int = 14, 
        fontsize: int = 14, 
        annot_size: int = 14, 
        mask_lower_triangle: bool = True, 
        ax: Optional[Any] = None, 
        save_path: Optional[str] = None, 
        dpi: int = 300
    ) -> None:
        """
        Plot a heatmap of group affinity matrix.

        Parameters
        ----------
        result_df : pd.DataFrame
            Group affinity matrix to plot.
        title : str
            Title for the heatmap.
        cmap : str, default='Blues'
            Colormap for the heatmap.
        tick_size : int, default=14
            Size of tick labels.
        fontsize : int, default=14
            Font size for title.
        annot_size : int, default=14
            Font size for annotations.
        mask_lower_triangle : bool, default=True
            Whether to mask the lower triangle.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.
        save_path : str, optional
            Path to save the figure.
        dpi : int, default=300
            DPI for saved figure.

        Examples
        --------
        >>> AffinityMatrix.plot_group_heatmap(group_matrix, "Group Similarities")
        """
        result_df = result_df.astype(float)
        base_mask = result_df.isna()

        # 建立下三角遮罩
        if mask_lower_triangle:
            lower_triangle_mask = np.tril(np.ones(result_df.shape), -1).astype(bool)
            combined_mask = base_mask | pd.DataFrame(lower_triangle_mask, 
                                                    index=result_df.index, 
                                                    columns=result_df.columns)
        else:
            combined_mask = base_mask

        # 建立圖表
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(result_df, cmap=cmap, mask=combined_mask, annot=True, fmt=".2f", 
                    annot_kws={"size": annot_size}, ax=ax)

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_size)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_size)
        ax.set_title(title, fontsize=fontsize)

        # 儲存圖片（如果有指定）
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

        if ax is None:
            plt.show()

    def plot_similarity_matrix(
        self,
        groups: pd.Series,
        figsize: Tuple[int, int] = (20, 20), 
        group_cmap: Dict[str, str] = {"LUAD": "orange", "ASC": "green", "LUSC": "blue"},
        title: str = "Cosine Similarity",
        cmap: str = "coolwarm",
        ax: Optional[Any] = None, 
        save_path: Optional[str] = None, 
        dpi: int = 300
    ) -> None:
        """
        Plot the similarity matrix with group annotations.

        Parameters
        ----------
        groups : pd.Series
            Group labels for each sample.
        figsize : tuple of int, default=(20, 20)
            Figure size as (width, height).
        group_cmap : dict, default={'LUAD': 'orange', 'ASC': 'green', 'LUSC': 'blue'}
            Color mapping for groups.
        title : str, default='Cosine Similarity'
            Title for the plot.
        cmap : str, default='coolwarm'
            Colormap for the similarity matrix.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.
        save_path : str, optional
            Path to save the figure.
        dpi : int, default=300
            DPI for saved figure.

        Examples
        --------
        >>> groups = pd.Series(['A', 'A', 'B', 'B'])
        >>> affinity_matrix.plot_similarity_matrix(groups, title="Sample Similarities")
        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 3, width_ratios=[40, 1, 4], height_ratios=[40, 1], wspace=0.02, hspace=0.02)
            ax_heatmap = fig.add_subplot(gs[0, 0])
            ax_colorbar = fig.add_subplot(gs[0, 1])
            ax_groupbar = fig.add_subplot(gs[1, 0])
        else:
            gs = ax.get_gridspec()
            ax_heatmap, ax_colorbar, ax_groupbar = ax

        sns.heatmap(self, 
                    ax=ax_heatmap, 
                    cbar=True, 
                    cmap=cmap, 
                    alpha=0.8, 
                    cbar_ax=ax_colorbar, 
                    yticklabels=False,
                    vmin=-1, 
                    vmax=1)

        ax_heatmap.set_xticks([])
        ax_heatmap.set_xlabel("")

        sns.heatmap(self.values[0].reshape((1, len(self.columns))), 
                    ax=ax_groupbar, 
                    cbar=False, 
                    alpha=0)

        color = groups.map(group_cmap).values
        for i, c in enumerate(color):
            ax_groupbar.add_patch(Rectangle(
                (i, 0), 1, 1,  # (x, y), width, height
                fill=True,
                facecolor=c,
                edgecolor="white",
                lw=0.1,
                alpha=0.4
            ))

        ax_groupbar.set_xticks([])
        ax_groupbar.set_yticks([])
        
        ax_heatmap.set_title(title, fontsize=20)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

        if ax is None:
            plt.show()

    def compare_group_pairs(
        self, 
        groups: pd.Series, 
        pair1: Tuple[str, str], 
        pair2: Tuple[str, str]
    ) -> Tuple[float, float]:
        """
        Perform statistical test comparing affinity between two group pairs.

        Parameters
        ----------
        groups : pd.Series
            Group labels for each sample.
        pair1 : tuple of str
            First group pair to compare (group1, group2).
        pair2 : tuple of str  
            Second group pair to compare (group1, group2).

        Returns
        -------
        stat : float
            Mann-Whitney U test statistic.
        p_value : float
            P-value of the test.

        Examples
        --------
        >>> stat, p = affinity_matrix.compare_group_pairs(
        ...     groups, ('A', 'B'), ('A', 'C')
        ... )
        """
        pair1_indices1, pair1_indices2 = np.where(groups == pair1[0])[0], np.where(groups == pair1[1])[0]
        pair2_indices1, pair2_indices2 = np.where(groups == pair2[0])[0], np.where(groups == pair2[1])[0]

        pair1_subset = self.iloc[pair1_indices1, pair1_indices2]
        pair2_subset = self.iloc[pair2_indices1, pair2_indices2]

        stat, p = mannwhitneyu(pair1_subset.to_numpy().flatten(),
                            pair2_subset.to_numpy().flatten())
        return stat, p
    
    def to_edges_dataframe(
        self, 
        label: str, 
        freq_threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Convert affinity matrix to edge list format for network analysis.

        Parameters
        ----------
        label : str
            Label to assign to all edges.
        freq_threshold : float, default=0.1
            Minimum frequency threshold for edge inclusion.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: source, target, frequency, label.
            Self-loops are removed.

        Examples
        --------
        >>> edges_df = affinity_matrix.to_edges_dataframe('similarity', 0.2)
        """
        edges_dataframe = self.melt(
            ignore_index=False,  # 保留索引
            var_name='target', 
            value_name='frequency'
        ).reset_index().rename(columns={'Hugo_Symbol': 'source'})

        # filter low frequency edges
        filtered_edges_dataframe = edges_dataframe[edges_dataframe.frequency >= freq_threshold]

        # remove self-loops
        filtered_edges_dataframe = filtered_edges_dataframe[~(filtered_edges_dataframe.source == filtered_edges_dataframe.target)]

        # add label attribute to edges
        filtered_edges_dataframe['label'] = label

        return filtered_edges_dataframe

    def to_networkx_graph(
        self, 
        label: str, 
        freq_threshold: float = 0.1
    ) -> nx.MultiGraph:
        """
        Convert affinity matrix to NetworkX graph for network analysis.

        Parameters
        ----------
        label : str
            Label to assign to all edges.
        freq_threshold : float, default=0.1
            Minimum frequency threshold for edge inclusion.

        Returns
        -------
        nx.MultiGraph
            NetworkX graph with frequency and label as edge attributes.

        Examples
        --------
        >>> graph = affinity_matrix.to_networkx_graph('similarity', 0.2)
        >>> print(f"Graph has {graph.number_of_nodes()} nodes")
        """
        edges_dataframe = self.to_edges_dataframe(label, freq_threshold)
        graph = nx.from_pandas_edgelist(
            edges_dataframe, 
            source='source', 
            target='target', 
            edge_attr=['frequency', 'label'],
            create_using=nx.MultiGraph()
        )
        return graph
    
    @staticmethod
    def plot_permutation_distribution(
        permutation_list: List[pd.DataFrame], 
        true_result_df: pd.DataFrame, 
        group1: str, 
        group2: str,
        figsize: Tuple[int, int] = (6, 4),
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Plot the distribution of permuted values vs. the true observed value.

        Parameters
        ----------
        permutation_list : list of pd.DataFrame
            List of permuted affinity matrices.
        true_result_df : pd.DataFrame
            True observed affinity matrix.
        group1 : str
            First group name.
        group2 : str
            Second group name.
        figsize : tuple of int, default=(6, 4)
            Figure size as (width, height).
        save_path : str, optional
            Path to save the figure.
        dpi : int, default=300
            DPI for saved figure.

        Examples
        --------
        >>> AffinityMatrix.plot_permutation_distribution(
        ...     perm_list, true_matrix, 'A', 'B'
        ... )
        """
        # 取出所有 permutation 的相似度  
        permuted_values = []
        for p in permutation_list:
            val = p.loc[group1, group2]
            # Simple conversion to float
            try:
                permuted_values.append(float(val))
            except (TypeError, ValueError):
                # Handle complex scalars by taking real part
                permuted_values.append(float(np.real(val)))
        
        true_val = true_result_df.loc[group1, group2] 
        try:
            true_value = float(true_val)
        except (TypeError, ValueError):
            true_value = float(np.real(true_val))

        plt.figure(figsize=figsize)
        # Convert to numpy array to ensure compatibility
        values_array = np.array(permuted_values, dtype=float)
        if sns is not None:
            sns.histplot(data=values_array, bins=30, kde=True, color="skyblue", edgecolor="white")
        else:
            plt.hist(values_array, bins=30, color="skyblue", edgecolor="white", alpha=0.7)
        plt.axvline(true_value, color="red", linestyle="--", 
                   label=f"True similarity = {true_value:.2f}")
        plt.title(f"Permutation Distribution: {group1} vs {group2}")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        
        plt.show()

    def plot_similarity(self,
                    groups,
                    figsize=(20, 20), 
                    group_cmap={"LUAD": "orange", "ASC": "green", "LUSC": "blue"},
                    title="cosine similarity",
                    cmap="coolwarm",
                    ax=None, 
                    save_path=None, 
                    dpi=300):
    
        if ax is None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 3, 
                                width_ratios=[40, 1, 4], 
                                height_ratios=[40, 1], 
                                wspace=0.02, hspace=0.02)
            ax_heatmap = fig.add_subplot(gs[0, 0])
            ax_colorbar = fig.add_subplot(gs[0, 1])
            ax_groupbar = fig.add_subplot(gs[1, 0])
        elif isinstance(ax, tuple) and len(ax) == 3:
            ax_heatmap, ax_colorbar, ax_groupbar = ax
        else:
            raise ValueError("If ax is provided, it must be a tuple of (ax_heatmap, ax_colorbar, ax_groupbar)")

        sns.heatmap(self, 
                    ax=ax_heatmap, 
                    cbar=True, 
                    cmap=cmap, 
                    alpha=0.8, 
                    cbar_ax=ax_colorbar, 
                    yticklabels=False,
                    vmin=-1, 
                    vmax=1)

        ax_heatmap.set_xticks([])
        ax_heatmap.set_xlabel("")

        sns.heatmap(self.values[0].reshape((1, len(self.columns))), 
                    ax=ax_groupbar, 
                    cbar=False, 
                    alpha=0)

        color = groups.map(group_cmap).values
        for i, c in enumerate(color):
            ax_groupbar.add_patch(Rectangle(
                (i, 0), 1, 1,  # (x, y), width, height
                fill=True,
                facecolor=c,
                edgecolor="white",
                lw=0.1,
                alpha=0.4
            ))

        ax_groupbar.set_xticks([])
        ax_groupbar.set_yticks([])
        # 儲存圖片（如果有指定）
        #plt.suptitle(title, fontsize=20, y=0.9)
        ax_heatmap.set_title(title, fontsize=20)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

        if ax is None:
            plt.show()

    
    @staticmethod
    def plot_heatmap(result_df, title, cmap="Blues", tick_size=14, fontsize=14, annot_size=14, 
                    mask_lower_triangle=True, ax=None, save_path=None, dpi=300):
        result_df = result_df.astype(float)
        base_mask = result_df.isna()

        # 建立下三角遮罩
        if mask_lower_triangle:
            lower_triangle_mask = np.tril(np.ones(result_df.shape), -1).astype(bool)
            combined_mask = base_mask | pd.DataFrame(lower_triangle_mask, 
                                                    index=result_df.index, 
                                                    columns=result_df.columns)
        else:
            combined_mask = base_mask

        # 建立圖表
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(result_df, cmap=cmap, mask=combined_mask, annot=True, fmt=".2f", 
                    annot_kws={"size": annot_size}, ax=ax)

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_size)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_size)
        ax.set_title(title, fontsize=fontsize)

        # 儲存圖片（如果有指定）
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

        if ax is None:
            plt.show()

    @staticmethod
    def analyze_similarity(table, 
                        groups,
                        title,
                        group_order,
                        method,
                        similarity_cmap="coolwarm",
                        group_cmap={"LUAD": "orange", "ASC": "green", "LUSC": "blue"},
                        group_avg_cmap="Blues",
                        group_pvalues_cmap="Reds_r",
                        save_dir="./figures/Similarity",
                        dpi=300):
        os.makedirs(save_dir, exist_ok=True)
        filename_base = title.replace(" ", "_")

        similarity_matrix = table.compute_similarity(method=method)
        true_group_similarity = similarity_matrix.get_mean_group_similarity(groups, 
                                                                        group_order=group_order)
        permutated_group_similarities = similarity_matrix.generate_permutation_list(groups, 
                                                                                        group_order=group_order, 
                                                                                        n_permutations=1000)
        group_pvalues_df = SimilarityMatrix.calculate_group_similarity_pvalues(true_group_similarity, 
                                                                            permutated_group_similarities, 
                                                                            group_order=group_order)
        

        fig = plt.figure(figsize=(20, 12))

        # 定義 gridspec：4, 4
        gs = gridspec.GridSpec(4, 4, 
                width_ratios=[15, 1, 1, 9],  
                height_ratios=[15, 2, 14, 1], 
                wspace=0.06, hspace=0.04
        )

        # 左邊：Similarity Heatmap
        ax_similarity = fig.add_subplot(gs[0:3, 0])      # 左邊兩列都佔
        ax_colorbar   = fig.add_subplot(gs[0:3, 1])      # 垂直 colorbar
        ax_groupbar   = fig.add_subplot(gs[3, 0])      # groupbar 疊在下方 (可改用 inset)

        # 右邊上：Group Mean Similarity
        ax_group_mean = fig.add_subplot(gs[0, 3])

        # 右邊下：Group Similarity P-values
        ax_group_pval = fig.add_subplot(gs[2:4, 3])


        # ---- 相似度矩陣
        similarity_matrix.plot_similarity(
            groups,
            group_cmap=group_cmap,
            cmap=similarity_cmap,
            title=f"{title} Matrix",
            ax=(ax_similarity, ax_colorbar, ax_groupbar),
            #save_path=os.path.join(save_dir, filename_base + "_matrix.png"),
        )

        # ---- Right top: 群體平均相似度
        
        SimilarityMatrix.plot_heatmap(
            true_group_similarity,
            title="Group Mean Similarity",
            cmap=group_avg_cmap,
            mask_lower_triangle=True,
            ax=ax_group_mean,
            #save_path=os.path.join(save_dir, f"{filename_base}_{'_'.join(title)}.png"),
        )

        # ---- Right bottom: permutation P-values
        
        SimilarityMatrix.plot_heatmap(
            group_pvalues_df,
            title="Group Similarity P-values",
            cmap=group_pvalues_cmap,
            mask_lower_triangle=True,
            ax=ax_group_pval,
            #save_path=os.path.join(save_dir, f"{filename_base}_{'_'.join(title)}.png"),
        )

        # ---- 儲存 / 顯示
        if save_dir:
            plt.savefig(os.path.join(save_dir, filename_base + ".png"), bbox_inches="tight", dpi=dpi)
        else:
            plt.show()

        return {
            "similarity_matrix": similarity_matrix,
            "group_similarity": true_group_similarity,
            "pval_matrix": group_pvalues_df
        }
    
    def paired_similarity_utest(self, groups, pair1, pair2):
        pair1_indices1, pair1_indices2 = np.where(groups == pair1[0])[0], np.where(groups == pair1[1])[0]
        pair2_indices1, pair2_indices2 = np.where(groups == pair2[0])[0], np.where(groups == pair2[1])[0]

        pair1_subset = self.iloc[pair1_indices1, pair1_indices2]
        pair2_subset = self.iloc[pair2_indices1, pair2_indices2]

        stat, p = mannwhitneyu(pair1_subset.to_numpy().flatten(),
                            pair2_subset.to_numpy().flatten())
        return stat, p