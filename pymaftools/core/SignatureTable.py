from .PivotTable import PivotTable
import pandas as pd

class SignatureTable(PivotTable):
    """
    SignatureTable class for handling COSMIC Single Base Substitution (SBS) signature data.
    """

    @classmethod
    def read_signature(cls, file_path):
        df = pd.read_csv(file_path, sep="\t", index_col=0).T
        return cls(df)