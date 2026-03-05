from __future__ import annotations

from .PivotTable import PivotTable


class SmallVariationTable(PivotTable):
    """
    Table for handling small variation (SNV/INDEL) data.

    Inherits from PivotTable and provides specific functionality for
    small variation analysis. The ``_constructor`` property ensures that
    pandas operations return SmallVariationTable objects.
    """
