from .PivotTable import PivotTable
import pandas as pd

class ExpressionTable(PivotTable):
    """
    ExpressionTable class for handling RNA expression data.
    """
    @property
    def _constructor(self):
        def _new_constructor(*args, **kwargs):
            obj = ExpressionTable(*args, **kwargs)
            # attempt to preserve metadata if available
            if hasattr(self, 'sample_metadata') and not self.sample_metadata.empty:
                try:
                    obj.sample_metadata = self.sample_metadata.copy()
                except:
                    pass
            if hasattr(self, 'feature_metadata') and not self.feature_metadata.empty:
                try:
                    obj.feature_metadata = self.feature_metadata.copy()
                except:
                    pass
            return obj
        return _new_constructor

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