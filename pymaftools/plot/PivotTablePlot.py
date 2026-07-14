"""Backward-compatible shim.

``PivotTablePlot`` was renamed to :class:`~pymaftools.plot.PivotStatsPlot.PivotStatsPlot`
in v0.5.0 (the old name wrongly implied it was the base for all PivotTable
plotting). The old import path is preserved::

    from pymaftools.plot.PivotTablePlot import PivotTablePlot  # still works
"""

from __future__ import annotations

from .PivotStatsPlot import PivotStatsPlot
from .PivotStatsPlot import PivotStatsPlot as PivotTablePlot

__all__ = ["PivotTablePlot", "PivotStatsPlot"]
