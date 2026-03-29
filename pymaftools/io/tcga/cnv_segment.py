"""TCGA CNV segment data builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...core.CopyNumberVariationTable import CopyNumberVariationTable
from .base import TCGATableBuilder

_CYTOBAND_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cytoBand.txt"


class TCGACNVSegmentBuilder(TCGATableBuilder):
    """
    Build CNV data from TCGA masked segment files.

    Provides two outputs:
    - ``build()`` → raw segment DataFrame (long format)
    - ``build_cytoband_table()`` → cytoband × sample CopyNumberVariationTable

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .seg.v2.txt files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    """

    file_pattern = "*.nocnv_grch38.seg.v2.txt"

    def read_and_merge(self, files: list[dict]) -> pd.DataFrame:
        segments = []
        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t")
            df["case_id"] = f["case_id"]
            df["sample_type"] = f["sample_type"]
            df = df.drop(columns=["GDC_Aliquot"], errors="ignore")
            segments.append(df)

        return pd.concat(segments, ignore_index=True)

    def build(self) -> pd.DataFrame:
        """Build raw segment DataFrame (long format)."""
        files = self.resolve_files()
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )

        seg_df = self.read_and_merge(files)
        print(
            f"[{self.__class__.__name__}] "
            f"{len(seg_df)} segments across {seg_df['case_id'].nunique()} cases"
        )
        return seg_df

    def build_cytoband_table(
        self,
        seg_df: pd.DataFrame | None = None,
        cytoband_path: str | Path | None = None,
    ) -> CopyNumberVariationTable:
        """
        Convert segments to cytoband × sample matrix.

        Computes overlap-weighted average Segment_Mean per cytoband per case.

        Parameters
        ----------
        seg_df : pd.DataFrame, optional
            Pre-built segment DataFrame. If None, calls ``build()`` first.
        cytoband_path : str or Path, optional
            Path to cytoBand.txt. Defaults to bundled file.

        Returns
        -------
        CopyNumberVariationTable
            Cytoband × sample matrix.
        """
        if seg_df is None:
            seg_df = self.build()

        if cytoband_path is None:
            cytoband_path = _CYTOBAND_PATH

        bands = pd.read_csv(
            cytoband_path, sep="\t", header=None,
            names=["chrom", "start", "end", "band", "stain"],
        )
        # Standard chromosomes only
        standard = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        bands = bands[bands["chrom"].isin(standard)].copy()
        bands["label"] = bands["chrom"] + bands["band"]

        # Normalize chromosome naming
        seg = seg_df.copy()
        seg["Chromosome"] = seg["Chromosome"].astype(str)
        if not seg["Chromosome"].iloc[0].startswith("chr"):
            seg["Chromosome"] = "chr" + seg["Chromosome"]

        records = []
        for _, b in bands.iterrows():
            chrom, bstart, bend, label = b["chrom"], b["start"], b["end"], b["label"]
            mask = (
                (seg["Chromosome"] == chrom)
                & (seg["Start"] < bend)
                & (seg["End"] > bstart)
            )
            overlap = seg.loc[mask]
            if overlap.empty:
                continue

            ol_start = overlap["Start"].clip(lower=bstart)
            ol_end = overlap["End"].clip(upper=bend)
            ol_len = ol_end - ol_start

            weighted = (
                overlap.assign(_ol_len=ol_len.values)
                .groupby("case_id")
                .apply(
                    lambda g: (g["Segment_Mean"] * g["_ol_len"]).sum() / g["_ol_len"].sum(),
                    include_groups=False,
                )
                .rename(label)
            )
            records.append(weighted)

        matrix = pd.DataFrame(records)
        matrix.index.name = "cytoband"

        # Feature metadata
        feature_meta = bands.set_index("label")[["chrom", "start", "end", "stain"]].copy()
        feature_meta = feature_meta.rename(columns={"chrom": "chromosome"})
        feature_meta["arm"] = feature_meta.index.str.extract(r"chr\w+([pq])")[0].values
        feature_meta = feature_meta.reindex(matrix.index)

        # Sample metadata from seg_df
        sample_meta = (
            seg_df.drop_duplicates("case_id")
            .set_index("case_id")[["sample_type"]]
            .reindex(matrix.columns)
        )

        table = CopyNumberVariationTable(matrix)
        table.feature_metadata = feature_meta
        table.sample_metadata = sample_meta

        print(
            f"[{self.__class__.__name__}] cytoband table: "
            f"{table.shape[0]} cytobands × {table.shape[1]} samples"
        )
        return table
