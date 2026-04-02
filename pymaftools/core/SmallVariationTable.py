from __future__ import annotations

import pandas as pd

from .PivotTable import PivotTable


class SmallVariationTable(PivotTable):
    """
    Table for handling small variation (SNV/INDEL) data.

    Inherits from PivotTable and provides specific functionality for
    small variation analysis. The ``_constructor`` property ensures that
    pandas operations return SmallVariationTable objects.
    """

    # Gene-level annotation columns — taken from first mutation per gene
    _GENE_LEVEL_COLS = [
        "Entrez_Gene_Id", "Gene", "BIOTYPE", "TRANSCRIPT_STRAND",
        "HGNC_ID", "RefSeq", "MANE", "APPRIS",
    ]

    # Mutation-level columns that need aggregation to gene level
    _GENE_LEVEL_AGG = {
        "hotspot": lambda s: (s == "Y").any(),
        "COSMIC": lambda s: s.notna().any(),
        "IMPACT": lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA,
        "GENE_PHENO": lambda s: (s == 1).any(),
    }

    def to_gene_level(self) -> "SmallVariationTable":
        """
        Collapse mutation-level SmallVariationTable to gene-level.

        The data matrix is collapsed by Hugo_Symbol using
        :meth:`MAF.merge_mutations` logic (Multi_Hit for multiple hits).
        feature_metadata gene-level annotation columns are taken from the
        first mutation per gene; mutation-level columns are aggregated.

        Returns
        -------
        SmallVariationTable
            Gene × sample matrix with gene-level feature_metadata.
        """
        from .MAF import MAF

        if "Hugo_Symbol" not in self.feature_metadata.columns:
            raise ValueError(
                "feature_metadata must contain 'Hugo_Symbol'. "
                "Run to_mutation_table() before to_gene_level()."
            )

        gene_col = self.feature_metadata["Hugo_Symbol"]

        # Collapse data matrix: group rows by gene, apply merge_mutations per sample
        gene_matrix = self.groupby(gene_col).agg(
            lambda col: MAF.merge_mutations(col)
        )
        result = SmallVariationTable(gene_matrix)
        result.sample_metadata = self.sample_metadata.copy()

        # Build gene-level feature_metadata
        fm = self.feature_metadata.copy()
        fm.index = gene_col

        # Gene-level cols: take first per gene
        present_gene_cols = [c for c in self._GENE_LEVEL_COLS if c in fm.columns]
        gene_fm = fm[present_gene_cols].groupby(level=0).first()

        # Aggregated cols
        for col, agg_fn in self._GENE_LEVEL_AGG.items():
            if col in fm.columns:
                gene_fm[col] = fm[col].groupby(level=0).agg(agg_fn)

        result.feature_metadata = gene_fm.reindex(result.index)
        return result
