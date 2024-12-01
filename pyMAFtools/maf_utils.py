    def head(self, n = 50):
        pivot_table = pivot_table.iloc[:n]
        pivot_table.gene_metadata = pivot_table.gene_metadata.iloc[:n]
