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
    """

    file_pattern = "*.maf.gz"

    def read_and_merge(self, files: list[dict]) -> MAF:
        frames = []
        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t", comment="#", low_memory=False)
            df["sample_ID"] = f["case_id"]
            df["sample_type"] = f["sample_type"]
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        maf = MAF(merged)
        maf.index = maf.loc[:, MAF.index_col].apply(
            lambda row: "|".join(row.astype(str)), axis=1
        )
        return maf

    def build_sample_metadata(self, table, files):
        # MAF is flat, sample_metadata doesn't apply the same way
        # Return a per-case summary
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
        return pd.DataFrame(meta_records.values()).set_index("case_id")

    def build(self) -> MAF:
        files = self.resolve_files()
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )

        maf = self.read_and_merge(files)
        maf._sample_metadata = self.build_sample_metadata(maf, files)

        print(
            f"[{self.__class__.__name__}] "
            f"{len(maf)} mutations across {maf['sample_ID'].nunique()} cases"
        )
        return maf
