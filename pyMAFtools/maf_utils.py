    def head(self, n = 50):
        pivot_table = pivot_table.iloc[:n]
        pivot_table.gene_metadata = pivot_table.gene_metadata.iloc[:n]
    
    def to_cooccur_matrix(self) -> 'CooccurMatrix':
        matrix = self.replace(False, np.nan).notna().astype(int)
        cooccur_matrix = matrix.dot(matrix.T)
        return CooccurMatrix(cooccur_matrix)

class CooccurMatrix(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self):
        return CooccurMatrix
