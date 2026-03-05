from __future__ import annotations


import pandas as pd

from .PivotTable import PivotTable


class ExpressionTable(PivotTable):
    """
    Table for handling RNA expression data.

    Inherits from PivotTable and provides specific functionality for
    gene expression analysis, including cluster-level aggregation.
    """

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
        # save clustering results
        table = self.copy()
        table[cluster_col] = table.feature_metadata[cluster_col]

        # to cluster table
        cluster_table = ExpressionTable(pd.DataFrame(table).groupby(cluster_col).mean())
        cluster_table.sample_metadata = table.sample_metadata

        gb = table.feature_metadata.groupby(cluster_col)
        cluster_table.feature_metadata["features"] = gb.apply(lambda df: list(df.index))
        cluster_table.feature_metadata["features_count"] = cluster_table.feature_metadata["features"].apply(len)
        return cluster_table.rename_index_and_columns()
