"""TCGA Methylation data builder."""

from __future__ import annotations

import numpy as np
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
        # TCGA methylation files contain hundreds of thousands of probes.  The
        # previous dict-of-Series implementation retained one copy of every
        # probe string for every sample, which could exceed available memory on
        # a complete cohort.  Discover the union of probe IDs one file at a
        # time, then fill one dense numeric array; probes absent from a file
        # remain NaN, matching pandas' normal Series alignment semantics.
        if not files:
            raise ValueError("At least one methylation file is required.")

        def read_probe_ids(file_info: dict) -> pd.Index:
            frame = pd.read_csv(
                file_info["filepath"],
                sep="\t",
                header=None,
                usecols=[0],
                names=["probe_id"],
                dtype={"probe_id": "string"},
            )
            # Store a plain object index so pandas/PyTables round-trips keep
            # the same index dtype as the other package readers.
            probe_ids = pd.Index(frame["probe_id"].astype(str), name="probe_id")
            if probe_ids.has_duplicates:
                raise ValueError(
                    f"Methylation file {file_info['filepath']} contains duplicate "
                    "probe identifiers."
                )
            return probe_ids

        probe_index = read_probe_ids(files[0])
        for file_info in files[1:]:
            probe_index = probe_index.union(read_probe_ids(file_info), sort=False)
        matrix_values = np.full(
            (len(probe_index), len(files)), np.nan, dtype=np.float64
        )
        identifiers = []
        for column, file_info in enumerate(files):
            frame = pd.read_csv(
                file_info["filepath"],
                sep="\t",
                header=None,
                names=["probe_id", "beta"],
                dtype={"probe_id": "string", "beta": "float64"},
            )
            probe_ids = pd.Index(frame["probe_id"].astype(str), name="probe_id")
            if probe_ids.has_duplicates:
                raise ValueError(
                    f"Methylation file {file_info['filepath']} contains duplicate "
                    "probe identifiers."
                )
            positions = probe_index.get_indexer(probe_ids)
            if (positions < 0).any():
                raise ValueError("Failed to align methylation probe identifiers.")
            matrix_values[positions, column] = frame["beta"].to_numpy(
                dtype=np.float64
            )
            identifiers.append(self.sample_identifier(file_info))

        matrix = pd.DataFrame(
            matrix_values, index=probe_index, columns=identifiers
        )
        return PivotTable(matrix)
