"""
SignatureTablePlot Module

Plotting functionality specific to SignatureTable objects.
Reachable via ``signature_table.plot``.
"""

from __future__ import annotations

from .PivotStatsPlot import PivotStatsPlot


class SignatureTablePlot(PivotStatsPlot):
    """
    Plotting interface for SignatureTable objects.

    Inherits the shared plotters from :class:`PivotStatsPlot` (``oncoplot``,
    ``plot_pca_samples``, ``plot_boxplot_with_annot``). This is a reserved
    extension point for mutational-signature plots; add them here as methods.
    """
