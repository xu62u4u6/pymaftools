from .PivotTable import PivotTable
import pandas as pd

class ExpressionTable(PivotTable):
    """
    ExpressionTable class for handling RNA expression data.
    """

    def to_cluster_table(self, cluster_col="cluster") -> pd.DataFrame:
        """
        cluster must in feature_metadata.
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