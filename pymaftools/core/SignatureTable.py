from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .PivotTable import PivotTable

if TYPE_CHECKING:
    from ..plot.SignatureTablePlot import SignatureTablePlot


class SignatureTable(PivotTable):
    """
    Table for handling COSMIC Single Base Substitution (SBS) signature data.

    Inherits from PivotTable and provides a convenience class method
    for reading signature weight files.
    """

    @property
    def plot(self) -> "SignatureTablePlot":
        """Access signature-specific plotting plus the shared plotters."""
        from ..plot.SignatureTablePlot import SignatureTablePlot

        return SignatureTablePlot(self)

    @classmethod
    def read_signature(
        cls,
        file_path: str,
        *,
        orientation: str = "samples_by_signatures",
    ) -> SignatureTable:
        """
        Read a signature weight file and return a SignatureTable.

        Parameters
        ----------
        file_path : str
            Path to a tab-separated signature exposure file.
        orientation : {"samples_by_signatures", "signatures_by_samples"}, \
default "samples_by_signatures"
            Orientation of the input file. Defaults to ``"samples_by_signatures"``
            (rows = samples, columns = signatures), matching SigProfilerExtractor /
            SigProfilerAssignment ``*_Activities.txt`` files and legacy behaviour.
            Pass ``orientation="signatures_by_samples"`` when the file already has
            signatures as rows and samples as columns. The returned table always
            follows the package's feature-by-sample contract (signatures × samples).

        Returns
        -------
        SignatureTable
            Exposure table with signatures as rows and samples as columns.
        """
        valid_orientations = {"signatures_by_samples", "samples_by_signatures"}
        if orientation not in valid_orientations:
            raise ValueError(
                f"orientation must be one of {sorted(valid_orientations)}; "
                f"got {orientation!r}."
            )
        df = pd.read_csv(file_path, sep="\t", index_col=0)
        if orientation == "samples_by_signatures":
            df = df.T
        return cls(df)
