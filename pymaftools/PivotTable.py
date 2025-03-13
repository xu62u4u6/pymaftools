import pandas as pd
import numpy as np
import networkx as nx
import pickle

from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency, fisher_exact
from .CooccurMatrix import CooccurMatrix

class PivotTable(pd.DataFrame):
    # columns: gene or mutation, row: sample or case
    _metadata = ["gene_metadata", "sample_metadata"]
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.gene_metadata = pd.DataFrame(index=self.index)
        self.sample_metadata = pd.DataFrame(index=self.columns)

    @property
    def _constructor(self):
        def _new_constructor(*args, **kwargs):
            obj = PivotTable(*args, **kwargs)
            obj._validate_metadata()
            return obj
        return _new_constructor
    
    def _validate_metadata(self):
        if not self.gene_metadata.index.equals(self.index):
            raise ValueError("gene_metadata index does not match PivotTable index.")

        if not self.sample_metadata.index.equals(self.columns):
            raise ValueError("sample_metadata index does not match PivotTable columns.")
        
    def to_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {file_path}.")

    @classmethod
    def read_pickle(cls, file_path):
        with open(file_path, 'rb') as f:
            pivot_table_instance = pickle.load(f)
        pivot_table_instance._validate_metadata()
        return pivot_table_instance

    def to_hierarchical_clustering(self, 
                             method: str = 'ward',
                             metric: str = 'euclidean') -> dict:
        from scipy.cluster.hierarchy import linkage
        
        # 基因聚類
        gene_linkage = linkage(self.values, 
                            method=method,
                            metric=metric)
        
        # 樣本聚類
        sample_linkage = linkage(self.values.T, 
                                method=method,
                                metric=metric)
        
        return {
            'gene_linkage': gene_linkage,
            'sample_linkage': sample_linkage
        }

    def copy(self, deep=True):
        pivot_table = super().copy(deep=deep)
        pivot_table.gene_metadata = self.gene_metadata.copy(deep=deep)
        pivot_table.sample_metadata = self.sample_metadata.copy(deep=deep)
        return pivot_table

    def subset(self, 
        genes: list = [], 
        samples: list = [], 
        how: str = "inner") -> 'PivotTable':

        if how not in ["inner", "outer"]:
            raise ValueError("how must be 'inner' or 'outer'")

        pivot_table = self.copy()

        # Subset samples
        if len(samples) > 0:
            if isinstance(samples, pd.Series):  # If it's a boolean Series
                samples = samples.reindex(self.columns, fill_value=False)  # Align with columns
                if samples.dtype != bool:
                    raise ValueError("When samples is a Series, it must be of boolean type.")
                samples = samples[samples].index  # Convert to index
            
            if how == "inner":
                samples = [s for s in samples if s in self.columns]  # Remove missing samples
            pivot_table = pivot_table.reindex(columns=samples)
            pivot_table.sample_metadata = pivot_table.sample_metadata.reindex(samples)

        # Subset genes
        if len(genes) > 0:
            if isinstance(genes, pd.Series):  # If it's a boolean Series
                genes = genes.reindex(self.index, fill_value=False)  # Align with index
                if genes.dtype != bool:
                    raise ValueError("When genes is a Series, it must be of boolean type.")
                genes = genes[genes].index  # Convert to index
            
            if how == "inner":
                genes = [g for g in genes if g in self.index]  # Remove missing genes
            pivot_table = pivot_table.reindex(index=genes)
            pivot_table.gene_metadata = pivot_table.gene_metadata.reindex(genes)

        return pivot_table
    
    @staticmethod
    def calculate_frequency(df: pd.DataFrame) -> pd.Series:
        return (df != False).sum(axis=1) / df.shape[1]
    
    def add_freq(self, groups: dict={}) -> "PivotTable":
        """
        example:
        groups: {"S": pd.dataframe, 
                 "A": pd.dataframe....} 
        groupname: subset of pivot table
        """
        pivot_table = self.copy()
        freq_data = pd.DataFrame()
        for group in groups.keys():
            freq_data[f"{group}_freq"] = PivotTable.calculate_frequency(groups[group])
        freq_data["freq"] = PivotTable.calculate_frequency(pivot_table)
        pivot_table.gene_metadata[freq_data.columns] = freq_data
        return pivot_table
    
    def sort_genes_by_freq(self, by="freq", ascending=False):
        pivot_table = self.copy()
        sorted_index = pivot_table.gene_metadata.sort_values(by=by, ascending=ascending).index
        
        # sort pivot table
        pivot_table = pivot_table.loc[sorted_index]

        # also sort gene_metadata
        pivot_table.gene_metadata = pivot_table.gene_metadata.loc[sorted_index]
        return pivot_table

    def sort_samples_by_mutations(self, top: int = 10):
        def binary_sort_key(column: pd.Series) -> int: 
            # binary column to int  
            binary_str = "".join(column.astype(int).astype(str))
            return int(binary_str, 2)
        
        # tmp_pivot_table = pivot_table.drop(columns=freq_columns)
        pivot_table = self.copy()
        binary_pivot_table = pivot_table != False
        mutations_weight = binary_pivot_table.head(top).apply(binary_sort_key, axis=0)
        pivot_table.sample_metadata["mutations_weight"] = mutations_weight
        sorted_samples = (mutations_weight
                    .sort_values(ascending=False)  
                    .index)                        
        
        # sort by order
        pivot_table = pivot_table.loc[:, sorted_samples]
        pivot_table.sample_metadata = pivot_table.sample_metadata.loc[sorted_samples, :]
        return pivot_table
    
    def sort_samples_by_group(self, group_col="subtype", group_order=["LUAD", "ASC", "LUSC"], top=10):
        """
        Sort samples first by the given subtype order, then within each subtype, 
        apply sort_samples_by_mutations.

        Parameters:
        - group_col (str): The column in sample_metadata containing group information.
        - group_order (list): The order to sort the groups.
        - top (int): The number of top genes used for sorting within each subtype.

        Returns:
        - PivotTable: A new PivotTable sorted by subtype and mutations.
        """
        pivot_table = self.copy()
        
        # 確保 group_col 存在於 sample_metadata
        if group_col not in pivot_table.sample_metadata.columns:
            raise ValueError(f"Column '{group_col}' not found in sample_metadata.")
        
        sorted_samples = []
        
        # 依照 subtype_order 進行分組排序
        for subtype in group_order:
            subtype_samples = pivot_table.sample_metadata[pivot_table.sample_metadata[group_col] == subtype].index
            
            if len(subtype_samples) > 0:
                # 篩選出該 subtype 的樣本，並應用 sort_samples_by_mutations
                subtype_pivot = pivot_table.subset(samples=subtype_samples)
                sorted_subtype_pivot = subtype_pivot.sort_samples_by_mutations(top=top)
                
                sorted_samples.extend(sorted_subtype_pivot.columns)  # 儲存排序後的樣本順序
        
        # 重新排列 PivotTable
        pivot_table = pivot_table.loc[:, sorted_samples]
        pivot_table.sample_metadata = pivot_table.sample_metadata.loc[sorted_samples, :]
        
        return pivot_table

    def plot_pca_samples(self, group_col="subtype", figsize=(8, 6), binary=True, 
                     palette_dict=None, alpha=0.8, title="PCA of samples", cmap="summer", is_numeric=False):

        # 1. 選擇數據表
        pivot_table = self.to_binary_table() if binary else self  

        # 2. 執行 PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pivot_table.T)  # 樣本在 row，所以轉置
        explained_variance = pca.explained_variance_ratio_

        # 3. 建立 PCA 結果 DataFrame
        df = pd.DataFrame(pca_result, index=pivot_table.columns, columns=["PC1", "PC2"])
        df[group_col] = self.sample_metadata[group_col]

        xlabel = f"Principal Component 1 ({explained_variance[0] * 100:.2f}%)"
        ylabel = f"Principal Component 2 ({explained_variance[1] * 100:.2f}%)"

        # 5. 繪製 PCA 圖
        plt.figure(figsize=figsize)

        if is_numeric:
            # 連續數值 → 使用 cmap
            scatter = plt.scatter(df["PC1"], df["PC2"], c=df[group_col], cmap=cmap, alpha=alpha)
            plt.colorbar(scatter, label=group_col)  # 手動加上 colorbar
        else:
            # 類別變數 → 使用 palette
            sns.scatterplot(data=df, x="PC1", y="PC2", hue=group_col, palette=palette_dict or "Set1", alpha=alpha)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if not is_numeric:
            plt.legend(title=group_col)

        plt.show()
        return df, pca

    def head(self, n = 50):
        pivot_table = self.copy()
        pivot_table = pivot_table.iloc[:n]
        pivot_table.gene_metadata = pivot_table.gene_metadata.iloc[:n]
        return pivot_table
    
    def filter_by_freq(self, threshold=0.05):
        if "freq" not in self.gene_metadata.columns:
            raise ValueError("freq column not found in gene_metadata.")
        pivot_table = self.copy()
        return pivot_table.subset(genes=pivot_table.gene_metadata.freq >= threshold)
    
    def to_cooccur_matrix(self, freq=True) -> 'CooccurMatrix':
        matrix = (self != False).astype(int)
        cooccur_matrix = matrix.dot(matrix.T)
        if freq:
            cooccur_matrix = cooccur_matrix / matrix.shape[1]

        return CooccurMatrix(cooccur_matrix)
    
    def to_binary_table(self):
        binary_pivot_table = self.copy()
        binary_pivot_table[:] = (binary_pivot_table != False) # avoid class transform
        return binary_pivot_table

    def chisquare_test(self, group_col, group1, group2, alpha=0.05, minimum_mutations=2):
        binary_pivot_table = self.to_binary_table()
        sample_metadata = binary_pivot_table.sample_metadata
        subset1 = binary_pivot_table.subset(samples=sample_metadata[group_col] == group1)
        subset2 = binary_pivot_table.subset(samples=sample_metadata[group_col] == group2)

        df = pd.DataFrame(index=binary_pivot_table.index, 
                        columns=[f"{group1}_True", f"{group1}_False", f"{group2}_True", f"{group2}_False"])
        
        df[f"{group1}_True"] = subset1.sum(axis=1)
        df[f"{group1}_False"] = len(subset1.columns) - df[f"{group1}_True"] 

        df[f"{group2}_True"] = subset2.sum(axis=1)
        df[f"{group2}_False"] = len(subset2.columns) - df[f"{group2}_True"]

        df = df[(df[f"{group1}_True"] + df[f"{group2}_True"]) > minimum_mutations]

        def get_chisquare_p_value(row):
            contingency_table = row.values.reshape(2, 2)
            _, p, _, _ = chi2_contingency(contingency_table)
            return p

        df["p_value"] = df.apply(get_chisquare_p_value, axis=1)
        p_values = df["p_value"].values
        reject, adjusted_p_value, _, _ = multipletests(p_values, method="fdr_bh", alpha=alpha)

        df["adjusted_p_value"] = adjusted_p_value
        df["is_significant"] = reject
        return df
    
    def merge(tables: list["PivotTable"], fill_table_na_with=False, fill_metadata_na_with=np.nan, join="outer"):
        if join not in ["inner", "outer"]:
            raise ValueError("join must be either 'inner' or 'outer'.")

        if join not in ["inner", "outer"]:
            raise ValueError("join must be either 'inner' or 'outer'.")

        if join == "inner":
            common_index = set(tables[0].data.index)
            for t in tables[1:]:
                common_index &= set(t.data.index)

            if not common_index:
                raise ValueError("No common indices found for 'inner' merge.")

            common_index = sorted(common_index)
            merged_data = pd.concat(
                [table.loc[common_index] for table in tables], axis=1, join="inner"
            )

            merged_metadata = pd.concat(
                [table.sample_metadata.loc[common_index] for table in tables], axis=0, join="inner"
            )

        else:
            merged_data = pd.concat([t for t in tables], axis=1, join="outer")
            merged_metadata = pd.concat([t.sample_metadata for t in tables], axis=0, join="outer")
            
        merged_table = PivotTable(merged_data).fillna(fill_table_na_with)
        merged_table.sample_metadata = merged_metadata.fillna(fill_metadata_na_with)
        return merged_table
    def cosine_similarity(self, figsize=(30, 30), 
                          cmap = {"LUAD": "orange", 
                                  "ASC": "green", 
                                  "LUSC": "blue"},
                          title="cosine similarity"):
                        
        
        data_matrix = self.T.values
        cosine_sim = cosine_similarity(data_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, 
                                    index=self.columns, 
                                    columns=self.columns)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[40, 1], wspace=0.02, hspace=0.02)

        ax_heatmap = fig.add_subplot(gs[0, 0])
        ax_colorbar = fig.add_subplot(gs[0, 1])
        ax_groupbar = fig.add_subplot(gs[1, 0])

        sns.heatmap(cosine_sim_df, 
                    ax=ax_heatmap, 
                    cbar=True, 
                    cmap="coolwarm", 
                    alpha=0.8, 
                    cbar_ax=ax_colorbar, 
                    yticklabels=False,
                    vmin=-1, 
                    vmax=1)
        ax_heatmap.set_xticks([])

        sns.heatmap(cosine_sim_df.values[0].reshape((1, len(self.columns))), 
                    ax=ax_groupbar, 
                    cbar=False, 
                    alpha=0, 
                    )

        group = self.sample_metadata['subtype']
        color = group.map(cmap).values

        for i, color in enumerate(color):
            ax_groupbar.add_patch(Rectangle(
                (i, 0), 1, 1,  # (x, y), width, height
                fill=True,
                facecolor=color,
                edgecolor="white",
                lw=0.1,
                alpha=0.4
            ))

        ax_groupbar.set_xticks([])
        ax_groupbar.set_yticks([])
