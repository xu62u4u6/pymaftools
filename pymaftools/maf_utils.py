import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency

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
    
    def head(self, n = 50):
        pivot_table = self.copy()
        pivot_table = pivot_table.iloc[:n]
        pivot_table.gene_metadata = pivot_table.gene_metadata.iloc[:n]
        return pivot_table
    
    def filter_by_freq(self, threshold=0.05):
        if "freq" not in self.gene_metadata.columns:
            raise ValueError("freq column not found in gene_metadata.")
        pivot_table = self.copy()
        return pivot_table.subset(gene=pivot_table.gene_metadata.freq >= threshold)
    
    def to_cooccur_matrix(self, freq=True) -> 'CooccurMatrix':
        matrix = (self != False).astype(int)
        cooccur_matrix = matrix.dot(matrix.T)
        if freq:
            cooccur_matrix = cooccur_matrix / matrix.shape[1]

        return CooccurMatrix(cooccur_matrix)
    
    def to_binary_table(self):
        pivot_table = self.copy()
        return pivot_table != False

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
class CooccurMatrix(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        return CooccurMatrix
    
    def to_edges_dataframe(cooccur_matrix, label, freq_threshold=0.1):
        edges_dataframe = cooccur_matrix.melt(
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

    def to_graph(self, label, freq_threshold=0.1):
        edges_dataframe = self.to_edges_dataframe(label, freq_threshold)
        graph = nx.from_pandas_edgelist(
            edges_dataframe, 
            source='source', 
            target='target', 
            edge_attr=['frequency', 'label'],
            create_using=nx.MultiGraph()
        )
        return graph

class MAF(pd.DataFrame):
    index_col = [
        "Hugo_Symbol",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2"
    ]

    # GDC MAF file fields:
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/Mutation_Annotation_Format_TCGAv2/
        
    vaild_variant_classfication = [
            "Frame_Shift_Del", 
            "Frame_Shift_Ins",
            "In_Frame_Del", 
            "In_Frame_Ins",
            "Missense_Mutation",
            "Nonsense_Mutation",
            "Silent",
            "Splice_Site",
            "Translation_Start_Site",
            "Nonstop_Mutation",
            "3'UTR",
            "3'Flank",
            "5'UTR",
            "5'Flank",
            "IGR",
            "Intron",
            "RNA",
            "Targeted_Region"
        ]
    
    nonsynonymous_types = [
        "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
        "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
        "Translation_Start_Site", "Nonstop_Mutation"
    ]
    
    @classmethod
    def read_maf(cls, maf_path, case_ID, preffix="", suffix=""):
        maf = cls(pd.read_csv(maf_path, skiprows=1, sep="\t"))
        maf["case_ID"] = f"{preffix}{case_ID}{suffix}"
        maf.index = maf.loc[:, cls.index_col].apply(lambda row: "|".join(row.astype(str)), axis=1) # concat column
        maf = maf.filter_maf(cls.vaild_variant_classfication)
        return cls(maf)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        # make sure returned object is MAF type
        return MAF
    
    def filter_maf(self, mutation_types):
        return self[self.Variant_Classification.isin(mutation_types)]

    @staticmethod
    def merge_mutations(column):
        if (column == False).all() :
            return False
        # Get unique non-False mutation types
        unique_mutations = column[column != False].unique()
        if len(unique_mutations) > 1:
            return "Multi_Hit"
        elif len(unique_mutations) == 1:
            return unique_mutations[0]
        
    def to_pivot_table(self) -> PivotTable: 
        pivot_table =  self.pivot_table(
                            values="Variant_Classification",
                            index="Hugo_Symbol",
                            columns="case_ID",
                            aggfunc=MAF.merge_mutations
                            ).fillna(False)
        pivot_table = PivotTable(pivot_table)
        pivot_table.sample_metadata["mutations_count"] = self.mutations_count
        pivot_table.sample_metadata["TMB"] = self.mutations_count / 40
        return pivot_table
    
    def to_mutation_table(self):
        mutation_table = self.pivot_table(index=self.index, 
                                columns="case_ID",
                                values="Variant_Classification",
                                aggfunc="first").fillna(False)
        mutation_table = PivotTable(mutation_table)
        mutation_table.sample_metadata["mutations_count"] = self.mutations_count
        mutation_table.sample_metadata["TMB"] = self.mutations_count / 40
        return mutation_table

    def change_index_level(self, index_col):
        maf = self.copy()
        new_index_col = maf.loc[:, index_col].apply(lambda row: "|".join(row.astype(str)), axis=1)
        maf.index = new_index_col
        return maf

    @property
    def mutations_count(self) -> pd.Series: 
        return self.groupby(self.case_ID).size()
    
    def sort_by_chrom(self) -> 'MAF':
        return self.sort_values(by=['Chromosome', 'Start_Position', 'End_Position'])
    
    @staticmethod
    def merge_mafs(mafs: list['MAF']) -> 'MAF':
        return MAF(pd.concat(mafs))
    
    @classmethod
    def read_csv(self, csv_path, sep="\t"):
        return MAF(pd.read_csv(csv_path, index_col=0, sep=sep))
    
    def to_csv(self, csv_path, **kwargs):
        # Set default arguments
        kwargs.setdefault("index", True)  # Ensure index is saved by default
        kwargs.setdefault("sep", "\t")   # Default to tab-separated values
        
        # Call the parent class's to_csv method
        super().to_csv(csv_path, **kwargs)