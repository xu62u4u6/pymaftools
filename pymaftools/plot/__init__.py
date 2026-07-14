"""
Plot Module

Visualization functionality for pymaftools.
"""

from .BasePlot import BasePlot
from .ColorManager import ColorManager
from .FontManager import FontManager
from .LegendManager import LegendManager
from .LollipopPlot import LollipopPlot
from .MafPlot import MafPlot
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
from .wes import (
    compare_cohorts,
    infer_vaf,
    mutation_burden_by_class,
    plot_cohort_comparison_forest,
    plot_forest,
    plot_rainfall,
    plot_somatic_interactions,
    plot_titv,
    plot_vaf,
    somatic_interactions,
    summarize_titv,
    top_mutated_genes,
)

__all__ = [
    "BasePlot",
    "ColorManager",
    "FontManager",
    "LegendManager",
    "LollipopPlot",
    "MafPlot",
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
    "compare_cohorts",
    "infer_vaf",
    "mutation_burden_by_class",
    "plot_cohort_comparison_forest",
    "plot_forest",
    "plot_rainfall",
    "plot_somatic_interactions",
    "plot_titv",
    "plot_vaf",
    "somatic_interactions",
    "summarize_titv",
    "top_mutated_genes",
]
