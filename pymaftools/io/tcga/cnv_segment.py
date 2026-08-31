"""TCGA CNV segment data builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    sample_type : str or None, default "Primary Tumor"
        Sample type to retain.
    sample_key : {"case_id", "sample_id"}, default "case_id"
        Identifier used for cytoband matrix columns. Use ``sample_id`` for
        exact specimen-level cross-omics alignment.
    """

    file_pattern = "*.ascat3.allelic_specific.seg.txt"

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

    def read_and_merge(self, files: list[dict]) -> pd.DataFrame:
        segments = []
        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t")
            df["case_id"] = f["case_id"]
            df["sample_ID"] = self.sample_identifier(f)
            df["sample_type"] = f["sample_type"]
            df["source_sample_id"] = f.get("sample_id")
            df["source_aliquot_id"] = f.get("aliquot_id")
            # Convert absolute copy number to log2 ratio (Segment_Mean convention)
            if "Segment_Mean" not in df.columns and "Copy_Number" in df.columns:
                df["Segment_Mean"] = np.log2(np.maximum(df["Copy_Number"], 0.001) / 2)
            segments.append(df)

        return pd.concat(segments, ignore_index=True)

    def build(self) -> pd.DataFrame:
        """Build raw segment DataFrame (long format)."""
        resolved_files = self.resolve_files()
        if not resolved_files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_dir}"
            )
        files = self.select_files(resolved_files)

        seg_df = self.read_and_merge(files)
        print(
            f"[{self.__class__.__name__}] "
            f"{len(seg_df)} segments across {seg_df['sample_ID'].nunique()} samples"
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
            cytoband_path,
            sep="\t",
            header=None,
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

        sample_column = "sample_ID" if "sample_ID" in seg.columns else "case_id"
        # Build segment–cytoband overlap pairs chromosome by chromosome.  The
        # previous implementation scanned every segment for every band; this
        # vectorized interval expansion only materializes actual overlaps and
        # keeps the result mathematically identical.
        overlap_frames = []
        for chromosome, seg_chr in seg.groupby("Chromosome", sort=False):
            bands_chr = bands.loc[bands["chrom"].eq(chromosome)].reset_index(drop=True)
            if bands_chr.empty or seg_chr.empty:
                continue
            band_starts = bands_chr["start"].to_numpy(dtype=np.int64)
            band_ends = bands_chr["end"].to_numpy(dtype=np.int64)
            seg_starts = seg_chr["Start"].to_numpy(dtype=np.int64)
            seg_ends = seg_chr["End"].to_numpy(dtype=np.int64)
            left = np.searchsorted(band_ends, seg_starts, side="right")
            right = np.searchsorted(band_starts, seg_ends, side="left")
            counts = np.maximum(right - left, 0)
            total = int(counts.sum())
            if total == 0:
                continue
            segment_indices = np.repeat(np.arange(len(seg_chr)), counts)
            offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            band_indices = np.repeat(left, counts) + offsets
            overlap_start = np.maximum(
                seg_starts[segment_indices], band_starts[band_indices]
            )
            overlap_end = np.minimum(seg_ends[segment_indices], band_ends[band_indices])
            overlap_length = overlap_end - overlap_start
            valid = overlap_length > 0
            if not valid.any():
                continue
            segment_values = seg_chr.iloc[segment_indices].reset_index(drop=True)
            overlap_frames.append(
                pd.DataFrame(
                    {
                        sample_column: segment_values.loc[
                            valid, sample_column
                        ].to_numpy(),
                        "label": bands_chr.loc[band_indices[valid], "label"].to_numpy(),
                        "weighted": (
                            segment_values.loc[valid, "Segment_Mean"].to_numpy()
                            * overlap_length[valid]
                        ),
                        "overlap_length": overlap_length[valid],
                    }
                )
            )

        if overlap_frames:
            overlaps = pd.concat(overlap_frames, ignore_index=True)
            grouped = overlaps.groupby([sample_column, "label"], sort=False)[
                ["weighted", "overlap_length"]
            ].sum()
            matrix = (
                grouped["weighted"]
                .div(grouped["overlap_length"])
                .unstack(sample_column)
            )
        else:
            matrix = pd.DataFrame()
        matrix.index.name = "cytoband"

        # Feature metadata
        feature_meta = bands.set_index("label")[
            ["chrom", "start", "end", "stain"]
        ].copy()
        feature_meta = feature_meta.rename(columns={"chrom": "chromosome"})
        feature_meta["arm"] = feature_meta.index.str.extract(r"chr\w+([pq])")[0].values
        feature_meta = feature_meta.reindex(matrix.index)

        # Sample metadata from seg_df
        sample_meta = (
            seg_df.drop_duplicates(sample_column)
            .set_index(sample_column)[["sample_type"]]
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
