"""Top-level public API for pymaftools."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import Any

# Core table classes stay eager because they are the common package entry point
# and do not import plotting backends.
from .core.CancerCellFractionTable import CancerCellFractionTable
from .core.Cohort import Cohort
from .core.CopyNumberVariationTable import CopyNumberVariationTable
from .core.ExpressionTable import ExpressionTable
from .core.MAF import MAF
from .core.PivotTable import PivotTable, capture_size
from .core.SignatureTable import SignatureTable
from .core.SmallVariationTable import SmallVariationTable
from .datasets import example_maf_path, load_example_maf

__version__ = version("pymaftools")


_LAZY_EXPORTS = {
    # core clustering
    "table_to_distance": ".core.Clustering",
    "k_fold_clustering_evaluation": ".core.Clustering",
    "align_clusters": ".core.Clustering",
    "align_cluster_label_dict": ".core.Clustering",
    "convert_ndarray_to_list": ".core.Clustering",
    "calculate_ari_matrix": ".core.Clustering",
    "plot_ari_matrix": ".core.Clustering",
    "run_random_forest_cv": ".core.Clustering",
    "run_random_forest_multiple_seeds": ".core.Clustering",
    "plot_cluster_feature_importance_boxplot": ".core.Clustering",
    "plot_cluster_feature_importance": ".core.Clustering",
    "run_feature_clustering": ".core.Clustering",
    "plot_clustering_metrics_and_find_best_k": ".core.Clustering",
    "SimilarityMatrix": ".core.PairwiseMatrix",
    # plot
    "OncoPlot": ".plot.OncoPlot",
    "LollipopPlot": ".plot.LollipopPlot",
    "MafPlot": ".plot.MafPlot",
    "MethodsPlot": ".plot.MethodsPlot",
    "ColorManager": ".plot.ColorManager",
    "FontManager": ".plot.FontManager",
    "ModelPlot": ".plot.ModelPlot",
    "compare_cohorts": ".plot.wes",
    "infer_vaf": ".plot.wes",
    "mutation_burden_by_class": ".plot.wes",
    "plot_cohort_comparison_forest": ".plot.wes",
    "plot_forest": ".plot.wes",
    "plot_maf_summary": ".plot.wes",
    "plot_rainfall": ".plot.wes",
    "plot_somatic_interactions": ".plot.wes",
    "plot_titv": ".plot.wes",
    "plot_vaf": ".plot.wes",
    "somatic_interactions": ".plot.wes",
    "summarize_titv": ".plot.wes",
    "top_mutated_genes": ".plot.wes",
    # model
    "OmicsStackingModel": ".model.StackingModel",
    "ASCStackingModel": ".model.StackingModel",
    "evaluate_model": ".model.modelUtils",
    "get_importance": ".model.modelUtils",
    "cross_validate_importance": ".model.modelUtils",
    "plot_metric_comparison_with_annotation": ".model.modelUtils",
    "to_importance_table": ".model.modelUtils",
    "plot_top_feature_importance_heatmap": ".model.modelUtils",
    "run_rfecv_feature_selection": ".model.modelUtils",
    "run_model_evaluation": ".model.modelUtils",
    # io
    "GDCClient": ".io.tcga",
    "parse_tcga_barcode": ".io.tcga",
    # utils
    "read_GMT": ".utils.geneset",
    "fetch_msigdb_geneset": ".utils.geneset",
    "get_exon_size": ".utils.geneinfo",
    "load_gene_sizes": ".utils.geneinfo",
    "PCA_CCA": ".utils.reduction",
}


__all__ = [
    "__version__",
    "PivotTable",
    "capture_size",
    "MAF",
    "CopyNumberVariationTable",
    "SmallVariationTable",
    "SimilarityMatrix",
    "ExpressionTable",
    "SignatureTable",
    "CancerCellFractionTable",
    "Cohort",
    "load_example_maf",
    "example_maf_path",
    "read_h5",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily load heavy top-level exports on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


def read_h5(h5_path, table_cls=None):
    """Read a single-table HDF5 file via ``PivotTable.read_h5``."""
    return (table_cls or PivotTable).read_h5(h5_path)
