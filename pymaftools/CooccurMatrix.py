import networkx as nx
import pandas as pd

class CooccurMatrix(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        return CooccurMatrix
    
    def to_edges_dataframe(cooccur_matrix, label, freq_threshold=0.1):
        edges_dataframe = cooccur_matrix.melt(
            ignore_index=False,  # 保留索引
            var_name='target', 
            value_name='frequency'
        ).reset_index().rename(columns={'Hugo_Symbol': 'source'})

        # filter low frequency edges
        filtered_edges_dataframe = edges_dataframe[edges_dataframe.frequency >= freq_threshold]

        # remove self-loops
        filtered_edges_dataframe = filtered_edges_dataframe[~(filtered_edges_dataframe.source == filtered_edges_dataframe.target)]

        # add label attribute to edges
        filtered_edges_dataframe['label'] = label

        return filtered_edges_dataframe

    def to_graph(self, label, freq_threshold=0.1):
        edges_dataframe = self.to_edges_dataframe(label, freq_threshold)
        graph = nx.from_pandas_edgelist(
            edges_dataframe, 
            source='source', 
            target='target', 
            edge_attr=['frequency', 'label'],
            create_using=nx.MultiGraph()
        )
        return graph