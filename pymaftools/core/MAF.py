import pandas as pd
import numpy as np
import networkx as nx
import os
import warnings
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
    def read_maf(cls, maf_path, sample_ID, preffix="", suffix=""):
        maf = cls(pd.read_csv(maf_path, skiprows=1, sep="\t"))
        maf["sample_ID"] = f"{preffix}{sample_ID}{suffix}"
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
        # If a gene has ≥2 mutations in a sample, mark as 'Multi_Hit' (even if types are the same).
        # Behavior aligned with maftools fix for issue #347: https://github.com/PoisonAlien/maftools/issues/347
        non_false_mutations = column[column != False]
        if len(non_false_mutations) > 1:
            return "Multi_Hit"
        elif len(non_false_mutations) == 1:
            return non_false_mutations[0]
        
    def to_pivot_table(self) -> PivotTable: 
        pivot_table =  self.pivot_table(
                            values="Variant_Classification",
                            index="Hugo_Symbol",
                            columns="sample_ID",
                            aggfunc=MAF.merge_mutations
                            ).fillna(False)
        pivot_table = PivotTable(pivot_table)
        pivot_table.sample_metadata["mutations_count"] = self.mutations_count
        #pivot_table.sample_metadata["TMB"] = self.mutations_count / 40
        return pivot_table
    
    def to_mutation_table(self):
        mutation_table = self.pivot_table(index=self.index, 
                                columns="sample_ID",
                                values="Variant_Classification",
                                aggfunc="first").fillna(False)
        mutation_table = PivotTable(mutation_table)
        mutation_table.sample_metadata["mutations_count"] = self.mutations_count
        return mutation_table

    def change_index_level(self, index_col=None):
        maf = self.copy()
        if index_col is None:
            index_col = self.index_col
        new_index_col = maf.loc[:, index_col].apply(lambda row: "|".join(row.astype(str)), axis=1)
        maf.index = new_index_col
        return maf

    @property
    def mutations_count(self) -> pd.Series: 
        return self.groupby(self.sample_ID).size()

    def sort_by_chrom(self) -> 'MAF':
        return self.sort_values(by=['Chromosome', 'Start_Position', 'End_Position'])
    
    @staticmethod
    def merge_mafs(mafs: list['MAF']) -> 'MAF':
        return MAF(pd.concat(mafs))
    
    @classmethod
    def read_csv(cls, csv_path, sep="\t", reindex=False):
        if reindex:
            maf = cls(pd.read_csv(csv_path, sep=sep))
            maf = maf.change_index_level()
        else:
            maf = cls(pd.read_csv(csv_path, sep=sep, index_col=0))
        return maf
    
    def to_csv(self, csv_path, **kwargs):
        # Set default arguments
        kwargs.setdefault("index", True)  # Ensure index is saved by default
        kwargs.setdefault("sep", "\t")   # Default to tab-separated values
        
        # Call the parent class's to_csv method
        super().to_csv(csv_path, **kwargs)

    def to_MAF(self, maf_path, **kwargs):
        # Set default arguments
        kwargs.setdefault("index", False)  # Ensure index is saved by default
        kwargs.setdefault("sep", "\t")   # Default to tab-separated values
        
        # Call the parent class's to_csv method
        super().to_csv(maf_path, **kwargs)

    def to_base_change_pivot_table(self):
        maf = self.copy()
        base_change = maf.loc[maf.Variant_Type == "SNP", ["Reference_Allele", "Tumor_Seq_Allele2", "sample_ID"]]
        base_change["Base_Change"] = base_change["Reference_Allele"] + "→" + base_change["Tumor_Seq_Allele2"]
        pivot_table = base_change.pivot_table(
            values="Reference_Allele", 
            index="sample_ID",
            columns="Base_Change",
            aggfunc="count",
            fill_value=0
        )
        pivot_table = PivotTable(pivot_table.T)
        pivot_table.sample_metadata["ti"] = pivot_table.loc[['A→G', 'C→T', 'G→A', 'T→C']].sum()
        pivot_table.sample_metadata["tv"] = pivot_table.loc[['A→C', 'A→T', 'C→A', 'C→G', 'G→C', 'G→T', 'T→A', 'T→G']].sum()
        pivot_table.sample_metadata["ti/tv"] = pivot_table.sample_metadata.ti / pivot_table.sample_metadata.tv
        return pivot_table
    
    def get_protein_info(self, gene):
        def extract_protein_start(pos):
            if pd.isna(pos):
                return None
            pos = str(pos).split('/')[0]
            if '-' in pos:
                return int(pos.split('-')[0])
            try:
                return int(pos)
            except:
                return None

        maf = self.filter_maf(self.nonsynonymous_types)
        sub_df = maf.loc[
            maf["Hugo_Symbol"] == gene, 
            ['Protein_position', 'Variant_Classification', 'Variant_Type']
        ].copy()

        # add amino acid position
        sub_df['AA_Position'] = sub_df['Protein_position'].apply(extract_protein_start)

        # get total AA length（858/1210 → 1210）
        try:
            AA_length = int(sub_df["Protein_position"].dropna().values[0].split('/')[-1])
        except:
            AA_length = None 

        # count mutations and to dict
        mutations_data = (
            sub_df
            .dropna(subset=['AA_Position', 'Variant_Classification'])
            .groupby(['AA_Position', 'Variant_Classification'])
            .size()
            .reset_index(name='count')
            .rename(columns={'AA_Position': 'position', 'Variant_Classification': 'type'})
            .to_dict(orient='records')
        )

        return AA_length, mutations_data

    @staticmethod
    def get_domain_info(gene_name, AA_length, protein_domains_path=None):
        if protein_domains_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # get domain info from https://github.com/PoisonAlien/maftools/blob/master/inst/extdata/protein_domains.RDs
            protein_domains_path = os.path.join(script_dir, "../data/protein_domains.csv")

        protein_domains =  pd.read_csv(protein_domains_path, index_col=0, low_memory=False)   
        subset = protein_domains.loc[
            (protein_domains.HGNC == gene_name) & 
            (protein_domains["aa.length"] == AA_length)
        ]
        if subset.empty:
            raise ValueError(f"No domain info found for {gene_name} with length {AA_length}")

        refseq_ids = subset["refseq.ID"].unique()
        if len(refseq_ids) != 1:
            warnings.warn(
                f"Multiple refseq.IDs found for {gene_name} with length {AA_length}: {refseq_ids}. "
                f"Selecting the first one: {refseq_ids[0]}"
            )
            subset = subset[subset["refseq.ID"] == refseq_ids[0]]
        return subset[['Start', 'End', 'Label']].to_dict(orient='records'), refseq_ids[0]
    

    def write_maf(self, file_path):
        self.to_csv(file_path, sep="\t", index=False)

    def write_SigProfilerMatrixGenerator_format(self, output_path):
        """
        轉換 MAF 為 SigProfilerMatrixGenerator 需要的格式，並存成 TSV 檔案。

        :param output_path: 儲存轉換後的 MAF 檔案路徑
        """
        # 重新命名符合 SigProfilerMatrixGenerator 標準
        rename_dict = {
            "Sample": "sample_ID",
            "chrom": "Chromosome",
            "pos_start": "Start_Position",
            "pos_end": "End_Position",
            "ref": "Reference_Allele",
            "alt": "Tumor_Seq_Allele2",
            "mut_type": "Variant_Type"
        }
        maf = self.copy().rename(columns=rename_dict)

        # 確保 Variant_Type 只有 "SNP", "INS", "DEL"
        maf = maf[maf["Variant_Type"].isin(["SNP", "INS", "DEL"])]
        maf.to_csv(output_path, sep="\t", index=False)

    def select_samples(self, sample_IDs: list):
        """
        選擇特定樣本的 MAF 資料。

        :param sample_IDs: 要選擇的樣本 ID 列表
        :return: 選擇後的 MAF 資料
        """
        return self[self.sample_ID.isin(sample_IDs)].copy()
        