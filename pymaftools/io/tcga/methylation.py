"""TCGA Methylation data builder."""

from __future__ import annotations

import pandas as pd

from ...core.PivotTable import PivotTable
from .base import TCGATableBuilder


class TCGAMethylationBuilder(TCGATableBuilder):
    """
    Build a PivotTable from TCGA methylation beta value files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing methylation beta files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    """

    file_pattern = "*.methylation_array.sesame.level3betas.txt"

    def read_and_merge(self, files: list[dict]) -> PivotTable:
        series_dict = {}

        for f in files:
            df = pd.read_csv(
                f["filepath"],
                sep="\t",
                header=None,
                names=["probe_id", "beta"],
                index_col="probe_id",
            )
            series_dict[f["case_id"]] = df["beta"]

        matrix = pd.DataFrame(series_dict)
        return PivotTable(matrix)
