from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .PivotTable import PivotTable

if TYPE_CHECKING:
    from ..plot.ExpressionTablePlot import ExpressionTablePlot


class ExpressionTable(PivotTable):
    """
    Table for handling RNA expression data.

    Inherits from PivotTable and provides specific functionality for
    gene expression analysis, including cluster-level aggregation.
    """

    @property
    def plot(self) -> "ExpressionTablePlot":
        """Access expression-specific plotting (volcano, PCA, boxplot, etc.)."""
        from ..plot.ExpressionTablePlot import ExpressionTablePlot

        return ExpressionTablePlot(self)

    def deseq2(
        self,
        group_col: str,
        target: str,
        control: str,
        **kwargs,
    ) -> None:
        """
        Run DESeq2 differential expression analysis (inplace).

        Results (log2FoldChange, pvalue, padj) are written into
        ``feature_metadata``.

        Parameters
        ----------
        group_col : str
            Column in ``sample_metadata`` that defines the two groups.
        target : str
            The experimental / target group value (numerator).
        control : str
            The control / reference group value (denominator).
        **kwargs
            Extra keyword arguments forwarded to
            ``DeseqDataSet.deseq2()`` (e.g. ``n_cpus=1``).
        """
        if group_col not in self.sample_metadata.columns:
            raise ValueError(f"Column '{group_col}' not found in sample_metadata.")
        if target == control:
            raise ValueError("target and control must be different group values.")

        group_values = self.sample_metadata[group_col]
        missing_groups = [
            group for group in (target, control) if group not in set(group_values.dropna())
        ]
        if missing_groups:
            raise ValueError(f"Group value(s) not found in '{group_col}': {missing_groups}")

        counts_frame = pd.DataFrame(self)
        numeric_counts = counts_frame.apply(pd.to_numeric, errors="coerce")
        if (
            numeric_counts.isna().any().any()
            or (numeric_counts < 0).any().any()
            or not np.equal(numeric_counts, np.floor(numeric_counts)).all().all()
        ):
            raise ValueError("DESeq2 requires non-negative integer counts without missing values.")

        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
        except ImportError as exc:
            raise ImportError(
                "pydeseq2 is required for differential expression analysis. "
                "Install it with: pip install 'pymaftools[expression]'"
            ) from exc

        # Build clean metadata with a safe column name (no spaces)
        metadata = self.sample_metadata[[group_col]].copy()
        label_map = {target: "target", control: "control"}
        metadata["_deseq_group"] = metadata[group_col].map(label_map)
        metadata = metadata.dropna(subset=["_deseq_group"])

        counts = numeric_counts.T.loc[metadata.index].astype(int)

        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design="~_deseq_group",
        )
        dds.deseq2(**kwargs)

        stat_res = DeseqStats(
            dds,
            contrast=["_deseq_group", "target", "control"],
        )
        stat_res.summary()

        results = stat_res.results_df

        # Write results into feature_metadata
        for col in ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]:
            if col in results.columns:
                self.feature_metadata[col] = results[col]

    def find_deg(
        self,
        padj: float = 0.05,
        log2fc: float = 1.0,
    ) -> pd.DataFrame:
        """
        Filter for differentially expressed genes.

        Parameters
        ----------
        padj : float, default 0.05
            Adjusted p-value threshold.
        log2fc : float, default 1.0
            Absolute log2 fold-change threshold.

        Returns
        -------
        pd.DataFrame
            Filtered subset of ``feature_metadata`` sorted by padj.
        """
        fm = self.feature_metadata
        required = {"padj", "log2FoldChange"}
        missing = sorted(required - set(fm.columns))
        if missing:
            raise ValueError("Run .deseq2() first.")
        mask = (fm["padj"] < padj) & (fm["log2FoldChange"].abs() > log2fc)
        return fm.loc[mask].sort_values("padj")

    def filter_low_expression(self, min_total: int = 10) -> "ExpressionTable":
        """
        Remove genes with low total counts across all samples.

        Parameters
        ----------
        min_total : int, default 10
            Minimum sum of counts across all samples to keep a gene.

        Returns
        -------
        ExpressionTable
            Filtered table with only sufficiently expressed genes.

        Raises
        ------
        TypeError
            If the data is not integer counts.
        """
        if not np.issubdtype(self.values.dtype, np.integer):
            raise TypeError(
                f"Expected integer counts, got {self.values.dtype}. "
                "DESeq2 requires raw counts."
            )
        mask = self.sum(axis=1) >= min_total
        return self.subset(features=mask)

    def to_cluster_table(self, cluster_col: str = "cluster") -> ExpressionTable:
        """
        Aggregate expression values by cluster assignment.

        Groups features (genes) by the specified cluster column in
        ``feature_metadata`` and computes the mean expression per cluster.

        Parameters
        ----------
        cluster_col : str, default "cluster"
            Column name in ``feature_metadata`` containing cluster labels.

        Returns
        -------
        ExpressionTable
            Cluster-level expression table with aggregated metadata.

        Raises
        ------
        ValueError
            If *cluster_col* is not found in ``feature_metadata``.
        """
        if cluster_col not in self.feature_metadata.columns:
            raise ValueError(f"Column '{cluster_col}' not found in feature_metadata.")
        labels = self.feature_metadata[cluster_col]
        included = labels.notna()
        data = pd.DataFrame(self).loc[included]
        cluster_table = ExpressionTable(data.groupby(labels.loc[included]).mean())
        cluster_table.sample_metadata = self.sample_metadata.copy()

        gb = self.feature_metadata.loc[included].groupby(cluster_col)
        cluster_table.feature_metadata["features"] = gb.apply(
            lambda df: list(df.index), include_groups=False
        )
        cluster_table.feature_metadata["features_count"] = (
            cluster_table.feature_metadata["features"].apply(len)
        )
        return cluster_table.rename_index_and_columns()
