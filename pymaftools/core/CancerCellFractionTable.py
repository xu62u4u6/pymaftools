from __future__ import annotations

import pandas as pd

from .PivotTable import PivotTable


class CancerCellFractionTable:
    """
    Handler for cancer cell fraction (CCF) data from clonal analysis tools.

    Provides methods for reading PyClone output and producing sorted
    PivotTable objects with cluster annotations.
    """

    @staticmethod
    def pyclone_to_sorted_table(filepath: str) -> PivotTable:
        """
        Read PyClone output and create a sorted PivotTable.

        Reads a tab-separated PyClone results file, pivots the data into a
        mutation-by-sample matrix of cellular prevalence values, and sorts
        mutations by cluster mean CCF (descending).

        Parameters
        ----------
        filepath : str
            Path to a PyClone results file (tab-separated) containing at
            minimum the columns ``mutation_id``, ``sample_id``,
            ``cellular_prevalence``, and ``cluster_id``.

        Returns
        -------
        PivotTable
            Sorted table with mutations as rows and samples as columns.
            Feature metadata includes ``mean_ccf``, ``cluster``, and
            ``cluster_text`` (e.g. "major", "minor1", "minor2", ...).
        """
        df = pd.read_csv(filepath, sep="\t")
        ccf_pivot = df.pivot(index="mutation_id",
                            columns="sample_id",
                            values="cellular_prevalence")
        table = PivotTable(ccf_pivot)
        table.feature_metadata["mean_ccf"] = df.groupby("mutation_id")["cellular_prevalence"].mean()
        table.feature_metadata["cluster"] = df.groupby("mutation_id")["cluster_id"].first()

        valid_clusters = table.feature_metadata["cluster"].unique()
        cluster_order = (table.feature_metadata.groupby("cluster")["mean_ccf"]
                        .mean()
                        .reindex(valid_clusters)
                        .sort_values(ascending=False).index
                        )
        sorted_mutation_id = (table.feature_metadata.reset_index()
                                .set_index("cluster")
                                .loc[cluster_order]
                                .sort_values(by="mean_ccf", ascending=False)
                                .reset_index()).mutation_id
        table_sorted = table.subset(features=sorted_mutation_id)
        cluster_text = ["major"] + [f"minor{i+1}" for i in range(len(cluster_order)-1)]
        table_sorted.feature_metadata["cluster_text"] = table_sorted.feature_metadata["cluster"].map(dict(zip(cluster_order, cluster_text)))
        return table_sorted
