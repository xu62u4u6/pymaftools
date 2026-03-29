"""
Base class for TCGA data builders.

Each TCGA data type subclasses TCGATableBuilder and implements
``read_and_merge()`` to handle its specific file format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .mapping import load_file_mapping, resolve_files


class TCGATableBuilder(ABC):
    """
    Base builder for TCGA data types.

    Subclasses must implement:
    - ``file_pattern``: glob pattern for matching files
    - ``read_and_merge(files)``: read files and return a table

    Parameters
    ----------
    data_dir : str or Path
        Directory containing downloaded files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    """

    file_pattern: str  # To be set by subclass

    def __init__(
        self,
        data_dir: str | Path,
        mapping: str | Path | pd.DataFrame,
    ):
        self.data_dir = Path(data_dir)

        if isinstance(mapping, (str, Path)):
            self.mapping_df = load_file_mapping(mapping)
        else:
            self.mapping_df = mapping

    def resolve_files(self) -> list[dict]:
        """Scan directory and resolve files to case metadata."""
        return resolve_files(self.data_dir, self.file_pattern, self.mapping_df)

    @abstractmethod
    def read_and_merge(self, files: list[dict]):
        """
        Read resolved files and merge into a table.

        Parameters
        ----------
        files : list of dict
            Output from :meth:`resolve_files`.

        Returns
        -------
        PivotTable or subclass
            Table with columns as sample identifiers.
        """
        ...

    def build_sample_metadata(self, table, files: list[dict]) -> pd.DataFrame:
        """
        Build sample_metadata from resolved file info.

        Creates a DataFrame indexed by case_id with columns:
        case_id, sample_type, file_id, data_type.

        Parameters
        ----------
        table : PivotTable
            The built table (used to get column names).
        files : list of dict
            Resolved file info.

        Returns
        -------
        pd.DataFrame
            Sample metadata indexed by case_id.
        """
        meta_records = {}
        for f in files:
            cid = f["case_id"]
            if cid not in meta_records:
                meta_records[cid] = {
                    "case_id": cid,
                    "sample_type": f["sample_type"],
                    "file_id": f["file_id"],
                    "data_type": f["data_type"],
                }

        meta = pd.DataFrame(meta_records.values())
        meta = meta.set_index("case_id")
        # Reindex to match table columns (some cases may have been deduped)
        meta = meta.reindex(table.columns)
        return meta

    def build(self):
        """
        Execute the full build pipeline.

        Returns
        -------
        PivotTable or subclass
            Complete table with sample_metadata attached.
        """
        files = self.resolve_files()
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )

        table = self.read_and_merge(files)
        table.sample_metadata = self.build_sample_metadata(table, files)

        print(
            f"[{self.__class__.__name__}] "
            f"{table.shape[0]} features × {table.shape[1]} samples"
        )
        return table
