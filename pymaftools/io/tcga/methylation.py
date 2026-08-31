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
    sample_type : str or None, default "Primary Tumor"
        Sample type to retain.
    sample_key : {"case_id", "sample_id"}, default "case_id"
        Identifier used for matrix columns. Use ``sample_id`` for exact
        specimen-level cross-omics alignment.
    """

    file_pattern = "*.methylation_array.sesame.level3betas.txt"

    def __init__(
        self,
        data_dir,
        mapping,
        sample_type: str | None = "Primary Tumor",
        sample_key: str = "case_id",
    ):
        super().__init__(
            data_dir,
            mapping,
            sample_type=sample_type,
            sample_key=sample_key,
        )

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
            identifier = self.sample_identifier(f)
            series_dict[identifier] = df["beta"].rename(identifier)

        matrix = pd.DataFrame(series_dict)
        return PivotTable(matrix)
