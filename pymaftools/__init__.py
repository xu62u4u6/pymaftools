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
from .core.ObservationMask import ObservationMask
from .core.PivotTable import PivotTable, capture_size
from .core.SampleManifest import SampleManifest
from .core.SignatureTable import SignatureTable
from .core.SmallVariationTable import SmallVariationTable
from .core.TMBAudit import TMBAudit
from .provenance import (
    Provenance,
    get_provenance,
    session_info,
    to_json,
    to_jsonable,
)
from .datasets import (
    example_maf_path,
    example_table_path,
    load_example_maf,
    load_example_table,
)

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
    "VCF": ".core.VCF",
    # plot
    "OncoPlot": ".plot.OncoPlot",
    "LollipopPlot": ".plot.LollipopPlot",
    "MafPlot": ".plot.MafPlot",
    "MethodsPlot": ".plot.MethodsPlot",
    "ColorManager": ".plot.ColorManager",
    "FontManager": ".plot.FontManager",
    "ModelPlot": ".plot.ModelPlot",
    "figure_summary": ".plot.summary",
    "compare_cohorts": ".plot.wes",
    "infer_vaf": ".plot.wes",
    "mutation_burden_by_class": ".plot.wes",
    "plot_cohort_comparison_forest": ".plot.wes",
    "plot_forest": ".plot.wes",
    "plot_rainfall": ".plot.wes",
    "plot_somatic_interactions": ".plot.wes",
    "plot_titv": ".plot.wes",
    "plot_vaf": ".plot.wes",
    "somatic_interactions": ".plot.wes",
    "summarize_titv": ".plot.wes",
    "top_mutated_genes": ".plot.wes",
    # model
    "OmicsStackingModel": ".model.StackingModel",
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
    "TCGAExpressionBuilder": ".io.tcga",
    "TCGAMutationBuilder": ".io.tcga",
    "TCGACNVSegmentBuilder": ".io.tcga",
    "TCGACNVGeneBuilder": ".io.tcga",
    "TCGAMethylationBuilder": ".io.tcga",
    "TCGAClinicalBuilder": ".io.tcga",
    # utils
    "get_exon_size": ".utils.geneinfo",
    "load_gene_sizes": ".utils.geneinfo",
}

# Names deprecated at the top level in 0.6.0: still importable (with a
# DeprecationWarning) until pymaftools 1.0, but no longer part of ``__all__``.
# Project-specific helpers belong in their submodule, not the package root.
_DEPRECATED_EXPORTS = {
    # name: (module, attribute, replacement shown to the user)
    "PCA_CCA": (".utils.reduction", "PCA_CCA", "pymaftools.utils.reduction.PCA_CCA"),
}

# Names removed in 0.6.0 (project-specific or superseded); accessing them
# raises with the replacement instead of a bare AttributeError.
_REMOVED_EXPORTS = {
    "ASCStackingModel": (
        "use OmicsStackingModel and read one class column: "
        "model.predict_proba(X)[:, model.class_order.index('LUSC')]"
    ),
    "read_GMT": "use gseapy.read_gmt (pip install gseapy)",
    "read_gmt": "use gseapy.read_gmt (pip install gseapy)",
    "fetch_msigdb_geneset": "use gseapy.Msigdb().get_gmt(category=...)",
}


__all__ = [
    "__version__",
    "PivotTable",
    "capture_size",
    "MAF",
    "SampleManifest",
    "ObservationMask",
    "TMBAudit",
    "CopyNumberVariationTable",
    "SmallVariationTable",
    "SimilarityMatrix",
    "ExpressionTable",
    "SignatureTable",
    "CancerCellFractionTable",
    "VCF",
    "Cohort",
    "load_example_maf",
    "example_maf_path",
    "load_example_table",
    "example_table_path",
    "read_h5",
    "Provenance",
    "get_provenance",
    "session_info",
    "to_json",
    "to_jsonable",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily load heavy top-level exports on first access."""
    if name in _DEPRECATED_EXPORTS:
        from ._deprecation import warn_deprecated

        module_name, attribute, replacement = _DEPRECATED_EXPORTS[name]
        warn_deprecated(f"pymaftools.{name}", replacement, stacklevel=3)
        # Not cached in globals(): every access keeps warning.
        return getattr(import_module(module_name, __name__), attribute)
    if name in _REMOVED_EXPORTS:
        raise AttributeError(
            f"pymaftools.{name} was removed in 0.6.0; {_REMOVED_EXPORTS[name]}"
        )
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
