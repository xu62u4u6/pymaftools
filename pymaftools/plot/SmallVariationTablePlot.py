"""
SmallVariationTablePlot Module

Plotting functionality specific to SmallVariationTable objects.
Reachable via ``small_variation_table.plot``.
"""

from __future__ import annotations

from .PivotStatsPlot import PivotStatsPlot


class SmallVariationTablePlot(PivotStatsPlot):
    """
    Plotting interface for SmallVariationTable objects.

    Inherits the shared plotters from :class:`PivotStatsPlot` (``oncoplot``,
    ``pca``, ``boxplot``). This is a reserved
    extension point for mutation-specific plots; add them here as methods.
    """
