from __future__ import annotations

import pandas as pd

from .PivotTable import PivotTable


class SignatureTable(PivotTable):
    """
    Table for handling COSMIC Single Base Substitution (SBS) signature data.

    Inherits from PivotTable and provides a convenience class method
    for reading signature weight files.
    """

    @classmethod
    def read_signature(cls, file_path: str) -> SignatureTable:
        """
        Read a signature weight file and return a SignatureTable.

        Parameters
        ----------
        file_path : str
            Path to a tab-separated signature file where rows are
            signatures and columns are mutation contexts.

        Returns
        -------
        SignatureTable
            Transposed table with signatures as columns.
        """
        df = pd.read_csv(file_path, sep="\t", index_col=0).T
        return cls(df)
