"""TCGA gene-level CNV data builder."""

from __future__ import annotations

import pandas as pd

from ...core.CopyNumberVariationTable import CopyNumberVariationTable
from .base import TCGATableBuilder


class TCGACNVGeneBuilder(TCGATableBuilder):
    """
    Build a CopyNumberVariationTable from TCGA gene-level copy number files.

    Supports both ASCAT3 filename variants:
    - ``*.gene_level_copy_number.v36.tsv``
    - ``*.gene_level.copy_number_variation.tsv``

    Parameters
    ----------
    data_dir : str or Path
        Directory containing gene-level CNV files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    value_column : str, default "copy_number"
        Column to use as values. Options: copy_number, min_copy_number, max_copy_number.
    strip_gene_version : bool, default True
        Strip ENSG version suffix (e.g. ``.15``).
    """

    file_pattern = "*.gene_level_copy_number.v36.tsv"

    def __init__(
        self,
        data_dir,
        mapping,
        value_column: str = "copy_number",
        strip_gene_version: bool = True,
    ):
        super().__init__(data_dir, mapping)
        self.value_column = value_column
        self.strip_gene_version = strip_gene_version

    def resolve_files(self) -> list[dict]:
        files = super().resolve_files()
        if not files:
            # Try alternative filename pattern
            self.file_pattern = "*.gene_level.copy_number_variation.tsv"
            files = super().resolve_files()
            if not files:
                self.file_pattern = "*.gene_level_copy_number.v36.tsv"
        return files

    def read_and_merge(self, files: list[dict]) -> CopyNumberVariationTable:
        gene_info = None
        series_dict = {}

        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t")

            if gene_info is None:
                info_cols = [c for c in ["gene_id", "gene_name", "chromosome", "start", "end"] if c in df.columns]
                gene_info = df[info_cols].copy()

            gene_ids = df["gene_id"]
            if self.strip_gene_version:
                gene_ids = gene_ids.str.replace(r"\.\d+$", "", regex=True)

            series_dict[f["case_id"]] = pd.Series(
                df[self.value_column].values,
                index=gene_ids.values,
                name=f["case_id"],
            )

        matrix = pd.DataFrame(series_dict)

        # Feature metadata
        if self.strip_gene_version:
            gene_info["gene_id"] = gene_info["gene_id"].str.replace(r"\.\d+$", "", regex=True)
        feature_meta = gene_info.set_index("gene_id")
        feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]

        table = CopyNumberVariationTable(matrix)
        table.feature_metadata = feature_meta.reindex(matrix.index)
        return table
