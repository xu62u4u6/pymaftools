"""TCGA Mutation (MAF) data builder."""

from __future__ import annotations

import pandas as pd

from ...core.MAF import MAF
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

    def read_and_merge(self, files: list[dict], tumor_only: bool = True) -> MAF:
        frames = []
        for f in files:
            if tumor_only and f.get("sample_type") != "Primary Tumor":
                continue
            df = pd.read_csv(f["filepath"], sep="\t", comment="#", low_memory=False)
            df["sample_ID"] = self.sample_identifier(f)
            df["sample_type"] = f["sample_type"]
            frames.append(df)

        if not frames:
            sample_filter = "Primary Tumor" if tumor_only else "any sample type"
            raise ValueError(
                f"No mutation files remained after selecting {sample_filter}."
            )

        merged = pd.concat(frames, ignore_index=True)
        maf = MAF(merged)
        maf.index = maf.loc[:, MAF.index_col].apply(
            lambda row: "|".join(row.astype(str)), axis=1
        )
        return maf

    def build_sample_metadata(self, table, files):
        # MAF is flat, sample_metadata doesn't apply the same way
        # Return a per-sample summary (Primary Tumor only)
        meta_records = {}
        for f in files:
            if f.get("sample_type") != "Primary Tumor":
                continue
            identifier = self.sample_identifier(f)
            if identifier not in meta_records:
                meta_records[identifier] = {
                    "case_id": f["case_id"],
                    "sample_id": f.get("sample_id"),
                    "sample_type": f["sample_type"],
                    "file_id": f["file_id"],
                    "data_type": f["data_type"],
                }
        metadata = pd.DataFrame(meta_records.values())
        metadata.index = pd.Index(meta_records.keys(), name=self.sample_key)
        return metadata

    def build(self) -> MAF:
        resolved_files = self.resolve_files()
        if not resolved_files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )
        files = self.select_files(resolved_files)

        maf = self.read_and_merge(files)
        maf._sample_metadata = self.build_sample_metadata(maf, files)

        print(
            f"[{self.__class__.__name__}] "
            f"{len(maf)} mutations across {maf['sample_ID'].nunique()} samples"
        )
        return maf
