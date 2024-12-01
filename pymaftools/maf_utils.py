import pandas as pd
import numpy as np

class PivotTable(pd.DataFrame):
    # columns: gene or mutation, row: sample or case
    _metadata = ["gene_metadata", "sample_metadata"]
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.gene_metadata = pd.DataFrame(index=self.index)
        self.sample_metadata = pd.DataFrame(index=self.columns)
        
    @property
    def _constructor(self):
        return PivotTable
    
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
    
    def to_cooccur_matrix(self) -> 'CooccurMatrix':
        matrix = self.replace(False, np.nan).notna().astype(int)
        cooccur_matrix = matrix.dot(matrix.T)
        return CooccurMatrix(cooccur_matrix)

class CooccurMatrix(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        return CooccurMatrix

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
        maf.index = maf.loc[:, cls.target_col].apply(lambda row: "|".join(row.astype(str)), axis=1) # concat column
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