"""TCGA Expression data builder."""

from __future__ import annotations

import pandas as pd

from ...core.ExpressionTable import ExpressionTable
from .base import TCGATableBuilder


class TCGAExpressionBuilder(TCGATableBuilder):
    """
    Build an ExpressionTable from TCGA STAR-Counts files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing expression files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    count_column : str, default "unstranded"
        Column to extract from STAR-Counts files.
        Options: unstranded, stranded_first, stranded_second,
        tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded.
    """

    file_pattern = "*.rna_seq.augmented_star_gene_counts.tsv"

    def __init__(self, data_dir, mapping, count_column: str = "unstranded"):
        super().__init__(data_dir, mapping)
        self.count_column = count_column

    def read_and_merge(self, files: list[dict]) -> ExpressionTable:
        gene_info = None
        series_dict = {}

        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t", comment="#")

            if gene_info is None:
                gene_info = df[["gene_id", "gene_name", "gene_type"]].copy()

            gene_ids = df["gene_id"].str.replace(r"\.\d+$", "", regex=True)
            series_dict[f["case_id"]] = pd.Series(
                df[self.count_column].values,
                index=gene_ids.values,
                name=f["case_id"],
            )

        matrix = pd.DataFrame(series_dict)

        # Feature metadata
        gene_info["gene_id"] = gene_info["gene_id"].str.replace(r"\.\d+$", "", regex=True)
        feature_meta = gene_info.set_index("gene_id")
        feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]

        table = ExpressionTable(matrix)
        table.feature_metadata = feature_meta.reindex(matrix.index)
        return table
