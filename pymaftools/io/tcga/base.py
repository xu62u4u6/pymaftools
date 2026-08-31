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
    sample_type : str or None, default "Primary Tumor"
        Sample type to retain when building a matrix. Set to None only when
        the input contains at most one file per case.
    sample_key : {"case_id", "sample_id"}, default "case_id"
        Identifier used for matrix columns and sample metadata indices.
        ``case_id`` preserves the historical case-level API. ``sample_id``
        keeps the exact GDC specimen barcode selected by the mapping, which
        is required when several modalities must be joined at specimen level.
    """

    file_pattern: str  # To be set by subclass

    def __init__(
        self,
        data_dir: str | Path,
        mapping: str | Path | pd.DataFrame,
        sample_type: str | None = "Primary Tumor",
        sample_key: str = "case_id",
    ):
        self.data_dir = Path(data_dir)
        self.sample_type = sample_type
        if sample_key not in {"case_id", "sample_id"}:
            raise ValueError("sample_key must be 'case_id' or 'sample_id'.")
        self.sample_key = sample_key

        if isinstance(mapping, (str, Path)):
            self.mapping_df = load_file_mapping(mapping)
        else:
            self.mapping_df = mapping

    def resolve_files(self) -> list[dict]:
        """Scan directory and resolve files to case metadata."""
        return resolve_files(self.data_dir, self.file_pattern, self.mapping_df)

    def select_files(self, files: list[dict]) -> list[dict]:
        """Select one deterministic input file for each case.

        Case-level matrices cannot represent multiple sample types or aliquots
        under the same column identifier. Filtering happens before both matrix
        construction and metadata construction so that they cannot disagree.
        """
        selected_pool = files
        if self.sample_type is not None:
            selected_pool = [
                f for f in files if f.get("sample_type") == self.sample_type
            ]
            if not selected_pool:
                raise ValueError(
                    f"No files have sample_type={self.sample_type!r}; "
                    "choose an available sample type or pass sample_type=None"
                )

        by_case: dict[str, list[dict]] = {}
        for file_info in selected_pool:
            by_case.setdefault(file_info["case_id"], []).append(file_info)

        selected = []
        for case_id, candidates in sorted(by_case.items()):
            candidates = sorted(
                candidates,
                key=lambda f: (str(f["filepath"]), str(f["file_id"])),
            )
            sample_types = {f.get("sample_type") for f in candidates}
            if self.sample_type is None and len(sample_types) > 1:
                raise ValueError(
                    f"Case {case_id!r} has multiple sample types: "
                    f"{sorted(str(value) for value in sample_types)}; "
                    "set sample_type explicitly"
                )
            sample_ids = {f.get("sample_id") for f in candidates if f.get("sample_id")}
            if len(sample_ids) > 1:
                raise ValueError(
                    f"Case {case_id!r} has files from multiple specimens: "
                    f"{sorted(sample_ids)}; run specimen alignment first"
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Case {case_id!r} has {len(candidates)} files for the same "
                    "modality/specimen; resolve duplicate files explicitly"
                )
            selected.append(candidates[0])

        identifiers = [self.sample_identifier(file_info) for file_info in selected]
        if len(identifiers) != len(set(identifiers)):
            duplicates = sorted(
                identifier
                for identifier in set(identifiers)
                if identifiers.count(identifier) > 1
            )
            raise ValueError(
                f"Selected files have duplicate {self.sample_key} values: "
                f"{duplicates}; resolve the mapping before building."
            )

        return selected

    def sample_identifier(self, file_info: dict) -> str:
        """Return the configured identifier for one resolved GDC file."""
        value = file_info.get(self.sample_key)
        if value is None or pd.isna(value) or str(value).strip() == "":
            raise ValueError(
                f"Mapping is missing '{self.sample_key}' for file "
                f"{file_info.get('file_id', file_info.get('filepath'))}; "
                f"cannot build with sample_key='{self.sample_key}'."
            )
        return str(value)

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

        Creates a DataFrame indexed by ``sample_key`` with provenance columns
        including case/specimen IDs, file IDs, and checksums.

        Parameters
        ----------
        table : PivotTable
            The built table (used to get column names).
        files : list of dict
            Resolved file info.

        Returns
        -------
        pd.DataFrame
            Sample metadata indexed by the configured sample identifier.
        """
        meta_records = {}
        for f in files:
            identifier = self.sample_identifier(f)
            if identifier not in meta_records:
                provenance_columns = [
                    "case_id",
                    "project",
                    "sample_id",
                    "sample_uuid",
                    "sample_type",
                    "portion_id",
                    "analyte_id",
                    "aliquot_id",
                    "aliquot_uuid",
                    "file_id",
                    "data_type",
                    "gdc_data_type",
                    "workflow_type",
                    "md5",
                    "size",
                    "state",
                    "mapping_status",
                ]
                meta_records[identifier] = {
                    column: f.get(column) for column in provenance_columns
                }

        meta = pd.DataFrame(meta_records.values())
        meta.index = pd.Index(meta_records.keys(), name=self.sample_key)
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
        resolved_files = self.resolve_files()
        if not resolved_files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )
        files = self.select_files(resolved_files)

        table = self.read_and_merge(files)
        table.sample_metadata = self.build_sample_metadata(table, files)

        print(
            f"[{self.__class__.__name__}] "
            f"{table.shape[0]} features × {table.shape[1]} samples"
        )
        return table
