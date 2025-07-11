from .PivotTable import PivotTable
import pandas as pd

class CancerCellFractionTable:
    """
    A class to handle cancer cell fraction data, including sorting and clustering.
    """
    @property
    def _constructor(self):
        def _new_constructor(*args, **kwargs):
            obj = CancerCellFractionTable(*args, **kwargs)
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
    
    def pyclone_to_sorted_table(filepath):
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
    