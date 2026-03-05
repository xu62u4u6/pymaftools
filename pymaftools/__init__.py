# pyMAF/__init__.py
from importlib.metadata import version

__version__ = version("pymaftools")

# core
from .core.PivotTable import PivotTable
from .core.MAF import MAF
from .core.CopyNumberVariationTable import CopyNumberVariationTable
from .core.SmallVariationTable import SmallVariationTable
from .core.PairwiseMatrix import SimilarityMatrix
from .core.ExpressionTable import ExpressionTable
from .core.SignatureTable import SignatureTable
from .core.CancerCellFractionTable import CancerCellFractionTable
from .core.Cohort import Cohort
from .core.Clustering import (
    table_to_distance,
    k_fold_clustering_evaluation,
    align_clusters,
    align_cluster_label_dict,
    convert_ndarray_to_list,
    calculate_ari_matrix,
    plot_ari_matrix,
    run_random_forest_cv,
    run_random_forest_multiple_seeds,
    plot_cluster_feature_importance_boxplot,
    plot_cluster_feature_importance,
    run_feature_clustering,
    plot_clustering_metrics_and_find_best_k,
    gpt_known_genes_summary,
)

# plot
from .plot.OncoPlot import OncoPlot
from .plot.LollipopPlot import LollipopPlot
from .plot.MethodsPlot import MethodsPlot
from .plot.ColorManager import ColorManager
from .plot.FontManager import FontManager
from .plot.ModelPlot import ModelPlot

# model
from .model.StackingModel import OmicsStackingModel, ASCStackingModel
from .model.modelUtils import (
    evaluate_model,
    get_importance,
    cross_validate_importance,
    plot_metric_comparison_with_annotation,
    to_importance_table,
    plot_top_feature_importance_heatmap,
    run_rfecv_feature_selection,
    run_model_evaluation,
)

# utils
from .utils.geneset import read_GMT, fetch_msigdb_geneset
from .utils.reduction import PCA_CCA
