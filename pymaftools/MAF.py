import pandas as pd
import numpy as np
import networkx as nx

from .PivotTable import PivotTable

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
        # maf = maf.filter_maf(cls.vaild_variant_classfication)
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

    def to_base_change_pivot_table(self):
        maf = self.copy()
        base_change = maf.loc[maf.Variant_Type == "SNP", ["Reference_Allele", "Tumor_Seq_Allele2", "case_ID"]]
        base_change["Base_Change"] = base_change["Reference_Allele"] + "→" + base_change["Tumor_Seq_Allele2"]
        pivot_table = base_change.pivot_table(
            values="Reference_Allele", 
            index="case_ID",
            columns="Base_Change",
            aggfunc="count",
            fill_value=0
        )
        return pivot_table
    
    def write_maf(self, file_path):
        self.to_csv(file_path, sep="\t", index=False)

    def write_SigProfilerMatrixGenerator_format(self, output_path):
        """
        轉換 MAF 為 SigProfilerMatrixGenerator 需要的格式，並存成 TSV 檔案。

        :param output_path: 儲存轉換後的 MAF 檔案路徑
        """
        # 重新命名符合 SigProfilerMatrixGenerator 標準
        rename_dict = {
            "Sample": "case_ID",
            "chrom": "Chromosome",
            "pos_start": "Start_Position",
            "pos_end": "End_Position",
            "ref": "Reference_Allele",
            "alt": "Tumor_Seq_Allele2",
            "mut_type": "Variant_Type"
        }
        maf = self.copy().rename(columns=rename_dict)

        # 確保 Variant_Type 只有 "SNP", "INS", "DEL"
        maf = maf[self.maf["Variant_Type"].isin(["SNP", "INS", "DEL"])]

        self.maf.to_csv(output_path, sep="\t", index=False)