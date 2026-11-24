"""TCGA Mutation (MAF) data builder."""

from __future__ import annotations

import pandas as pd

from ...core.MAF import MAF
from ...core.maf_io import read_leading_comments
from .base import TCGATableBuilder


class TCGAMutationBuilder(TCGATableBuilder):
    """
    Build a MAF object from TCGA masked somatic mutation files.

    Note: MAF is a flat DataFrame (not a PivotTable matrix).
    Use ``maf.to_pivot_table()`` to get a gene × sample matrix.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .maf.gz files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    sample_type : str or None, default "Primary Tumor"
        Sample type to retain during file selection.
    sample_key : {"case_id", "sample_id"}, default "case_id"
        Identifier written to the MAF ``sample_ID`` column. Use ``sample_id``
        for exact specimen-level cross-omics alignment.
    """

    file_pattern = "*.maf.gz"

    def _target_sample_type(self, tumor_only: bool | None) -> str | None:
        if tumor_only is False:
            return None
        if tumor_only is True:
            return self.sample_type or "Primary Tumor"
        return self.sample_type

    def read_and_merge(self, files: list[dict], tumor_only: bool | None = None) -> MAF:
        target_sample_type = self._target_sample_type(tumor_only)
        frames = []
        for f in files:
            if (
                target_sample_type is not None
                and f.get("sample_type") != target_sample_type
            ):
                continue
            df = pd.read_csv(
                f["filepath"],
                sep="\t",
                skiprows=len(read_leading_comments(f["filepath"])),
                low_memory=False,
            )
            df["sample_ID"] = self.sample_identifier(f)
            df["sample_type"] = f.get("sample_type")
            frames.append(df)

        if not frames:
            sample_filter = (
                target_sample_type
                if target_sample_type is not None
                else "any sample type"
            )
            raise ValueError(
                f"No mutation files remained after selecting {sample_filter}."
            )

        merged = pd.concat(frames, ignore_index=True)
        maf = MAF(merged)
        if not maf.empty:
            maf.index = maf.loc[:, MAF.index_col].apply(
                lambda row: "|".join(row.astype(str)), axis=1
            )
        return maf

    def build_sample_metadata(self, table, files):
        # Return a per-sample summary matching self.sample_type (including zero-mutation files)
        meta_records = {}
        for f in files:
            if (
                self.sample_type is not None
                and f.get("sample_type") != self.sample_type
            ):
                continue
            identifier = self.sample_identifier(f)
            if identifier not in meta_records:
                meta_records[identifier] = {
                    "case_id": f["case_id"],
                    "sample_id": f.get("sample_id"),
                    "sample_type": f.get("sample_type"),
                    "file_id": f.get("file_id"),
                    "data_type": f.get("data_type"),
                }
        metadata = pd.DataFrame(meta_records.values())
        metadata.index = pd.Index(meta_records.keys(), name=self.sample_key)
        return metadata

    def build_sample_manifest(self, files: list[dict]):
        """Build a SampleManifest covering every selected sequenced MAF file."""
        from ...core.SampleManifest import SampleManifest

        records = []
        seen: set[str] = set()
        for f in files:
            if (
                self.sample_type is not None
                and f.get("sample_type") != self.sample_type
            ):
                continue
            identifier = self.sample_identifier(f)
            if identifier in seen:
                continue
            seen.add(identifier)
            records.append(
                {
                    "sample_id": identifier,
                    "patient_id": f["case_id"],
                    "eligible": True,
                    "assay": "WES",
                    "sample_type": f.get("sample_type"),
                    "file_id": f.get("file_id"),
                }
            )
        frame = pd.DataFrame(
            records,
            columns=[
                "sample_id",
                "patient_id",
                "eligible",
                "assay",
                "sample_type",
                "file_id",
            ],
        )
        frame["eligible"] = frame["eligible"].astype(bool)
        return SampleManifest(frame, sample_id_col="sample_id")

    def build(self) -> MAF:
        resolved_files = self.resolve_files()
        if not resolved_files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )
        files = self.select_files(resolved_files)

        maf = self.read_and_merge(files)
        maf._sample_metadata = self.build_sample_metadata(maf, files)
        maf.sample_manifest = self.build_sample_manifest(files)

        print(
            f"[{self.__class__.__name__}] "
            f"{len(maf)} mutations across {maf['sample_ID'].nunique()} mutated "
            f"samples ({len(maf.sample_manifest.eligible_samples)} sequenced)"
        )
        return maf
