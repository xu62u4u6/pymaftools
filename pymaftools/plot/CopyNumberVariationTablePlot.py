"""
CopyNumberVariationTablePlot Module

Plotting functionality specific to CopyNumberVariationTable objects.
Reachable via ``cnv_table.plot``.
"""

from __future__ import annotations

from .PivotStatsPlot import PivotStatsPlot


class CopyNumberVariationTablePlot(PivotStatsPlot):
    """
    Plotting interface for CopyNumberVariationTable objects.

    Inherits the shared plotters from :class:`PivotStatsPlot` (``oncoplot``,
    ``pca``, ``boxplot``) and adds CNV-specific
    visualisations.
    """

    def band_ratio(self, **kwargs):
        """Plot gain/loss frequency across cytobands for a CNV cluster.

        Delegates to the table implementation; see
        :meth:`CopyNumberVariationTable._cnv_band_ratio` for parameters.
        """
        return self.pivot_table._cnv_band_ratio(**kwargs)
