"""TCGA Expression data builder."""

from __future__ import annotations

import pandas as pd

from ...core.ExpressionTable import ExpressionTable
from .base import TCGATableBuilder

# GDC STAR-Counts files include four QC summary rows before the gene rows.
_QC_ROWS = ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]


def _zero_if_missing(value):
    """Return zero for missing STAR QC counters without evaluating pd.NA."""
    return 0 if pd.isna(value) else value


class TCGAExpressionBuilder(TCGATableBuilder):
    """
    Build an ExpressionTable from TCGA STAR-Counts files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing expression files.
    mapping : str, Path, or pd.DataFrame
        Path to file_to_case.tsv or pre-loaded mapping DataFrame.
    count_column : str, default "unstranded"
        Column to extract from STAR-Counts files.
        Options: unstranded, stranded_first, stranded_second,
        tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded.
    enrich_coordinates : bool, default True
        If True, enrich feature_metadata with genomic coordinates
        (chromosome_name, start_position, end_position) from Ensembl BioMart
        via :func:`pymaftools.utils.geneinfo.load_ensembl_map`.
        The first call downloads and caches the map; subsequent calls are fast.
    sample_type : str or None, default "Primary Tumor"
        Sample type to retain in the case-level matrix.
    """

    file_pattern = "*.rna_seq.augmented_star_gene_counts.tsv"

    def __init__(
        self,
        data_dir,
        mapping,
        count_column: str = "unstranded",
        enrich_coordinates: bool = True,
        sample_type: str | None = "Primary Tumor",
    ):
        super().__init__(data_dir, mapping, sample_type=sample_type)
        self.count_column = count_column
        self.enrich_coordinates = enrich_coordinates

    def read_and_merge(self, files: list[dict]) -> ExpressionTable:
        gene_info = None
        series_dict: dict[str, pd.Series] = {}
        qc_dict: dict[str, dict] = {}

        for f in files:
            df = pd.read_csv(f["filepath"], sep="\t", comment="#")

            # --- Separate QC summary rows from gene rows ---
            is_qc = df["gene_id"].isin(_QC_ROWS)
            qc_rows = df[is_qc].set_index("gene_id")[self.count_column]
            gene_rows = df[~is_qc]

            qc_record = {row: qc_rows.get(row, pd.NA) for row in _QC_ROWS}
            # Derive mapped_reads = sum of per-gene counts for this sample
            qc_record["mapped_reads"] = gene_rows[self.count_column].sum()
            total = (
                qc_record["mapped_reads"]
                + _zero_if_missing(qc_record["N_unmapped"])
                + _zero_if_missing(qc_record["N_multimapping"])
                + _zero_if_missing(qc_record["N_noFeature"])
                + _zero_if_missing(qc_record["N_ambiguous"])
            )
            qc_record["mapping_rate"] = (
                qc_record["mapped_reads"] / total if total > 0 else pd.NA
            )
            qc_dict[f["case_id"]] = qc_record

            if gene_info is None:
                gene_info = gene_rows[["gene_id", "gene_name", "gene_type"]].copy()

            gene_ids = gene_rows["gene_id"].str.replace(r"\.\d+$", "", regex=True)
            series_dict[f["case_id"]] = pd.Series(
                gene_rows[self.count_column].values,
                index=gene_ids.values,
                name=f["case_id"],
            )

        matrix = pd.DataFrame(series_dict)

        # --- Feature metadata ---
        gene_info["gene_id"] = gene_info["gene_id"].str.replace(
            r"\.\d+$", "", regex=True
        )
        feature_meta = gene_info.set_index("gene_id")
        feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]

        if self.enrich_coordinates:
            feature_meta = self._enrich_coordinates(feature_meta)

        table = ExpressionTable(matrix)
        table.feature_metadata = feature_meta.reindex(matrix.index)

        # --- QC / sample metadata ---
        table.sample_metadata = pd.DataFrame(qc_dict).T.reindex(matrix.columns)
        table.sample_metadata.index.name = "sample_ID"

        return table

    @staticmethod
    def _enrich_coordinates(feature_meta: pd.DataFrame) -> pd.DataFrame:
        """Merge genomic coordinates from Ensembl BioMart into feature_metadata.

        Adds columns: chromosome_name, start_position, end_position.
        Rows without a BioMart match keep NaN for those columns.
        """
        try:
            from ...utils.geneinfo import load_ensembl_map
        except ImportError:
            return feature_meta

        ensembl_map = load_ensembl_map()
        coord_cols = [
            "ensembl_gene_id",
            "chromosome_name",
            "start_position",
            "end_position",
        ]
        coords = (
            ensembl_map[coord_cols]
            .drop_duplicates(subset=["ensembl_gene_id"])
            .set_index("ensembl_gene_id")
        )
        return feature_meta.join(coords, how="left")
