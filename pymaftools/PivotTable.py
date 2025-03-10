import pandas as pd
import numpy as np
import networkx as nx

from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


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
           samples: list  = []) -> 'PivotTable':
        """
        Subset the PivotTable by specified genes and/or samples.

        Parameters:
            genes: List of genes or a boolean Series for genes. Default is an empty list (all genes).
            samples: List of samples or a boolean Series for samples. Default is an empty list (all samples).

        Returns:
            A new PivotTable containing the specified subset.
        """
        pivot_table = self.copy()

        # Subset samples
        if len(samples) > 0:
            if isinstance(samples, pd.Series):  # If it's a boolean Series
                samples = samples.loc[self.columns]  # Align with columns
                if samples.dtype != bool:
                    raise ValueError("When samples is a Series, it must be of boolean type.")
                samples = samples[samples].index  # Convert to index
            pivot_table = pivot_table.loc[:, samples]
            pivot_table.sample_metadata = pivot_table.sample_metadata.loc[samples, :]

        # Subset genes
        if len(genes) > 0:
            if isinstance(genes, pd.Series):  # If it's a boolean Series
                genes = genes.loc[self.index]  # Align with index
                if genes.dtype != bool:
                    raise ValueError("When genes is a Series, it must be of boolean type.")
                genes = genes[genes].index  # Convert to index
            pivot_table = pivot_table.loc[genes, :]
            pivot_table.gene_metadata = pivot_table.gene_metadata.loc[genes, :]
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
    