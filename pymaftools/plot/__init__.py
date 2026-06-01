"""
Plot Module

Visualization functionality for pymaftools.
"""

from .BasePlot import BasePlot
from .ColorManager import ColorManager
from .FontManager import FontManager
from .LegendManager import LegendManager
from .LollipopPlot import LollipopPlot
from .OncoPlot import OncoPlot
from .Track import (
    Track,
    MainMatrixTrack,
    BarTrack,
    FreqTrack,
    CategoricalTrack,
    NumericTrack,
)
from .PivotStatsPlot import PivotStatsPlot

# Backward compatibility: the old name remains importable via its original module
# path, ``from pymaftools.plot.PivotTablePlot import PivotTablePlot`` (the shim in
# PivotTablePlot.py). We deliberately do NOT alias ``PivotTablePlot`` at the
# package top level, because the same-named shim submodule would shadow it.
from .MethodsPlot import MethodsPlot
from .ModelPlot import ModelPlot

__all__ = [
    "BasePlot",
    "ColorManager",
    "FontManager",
    "LegendManager",
    "LollipopPlot",
    "OncoPlot",
    "Track",
    "MainMatrixTrack",
    "BarTrack",
    "FreqTrack",
    "CategoricalTrack",
    "NumericTrack",
    "PivotStatsPlot",
    "MethodsPlot",
    "ModelPlot",
]
