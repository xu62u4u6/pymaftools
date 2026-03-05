import numpy as np
import pandas as pd

from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.Clustering import (
    table_to_distance,
    align_clusters,
    align_cluster_label_dict,
    convert_ndarray_to_list,
    calculate_ari_matrix,
    run_random_forest_cv,
    run_random_forest_multiple_seeds,
    plot_clustering_metrics_and_find_best_k,
    gpt_known_genes_summary,
)


def test_table_to_distance_returns_square_matrix(sample_pivot_table):
    distance = table_to_distance(sample_pivot_table)

    assert distance.shape == (sample_pivot_table.shape[1], sample_pivot_table.shape[1])
    np.testing.assert_allclose(np.diag(distance), np.zeros(sample_pivot_table.shape[1]), atol=1e-8)


def test_align_related_cluster_helpers():
    ref = np.array([0, 0, 1, 1])
    target = np.array([1, 1, 0, 0])

    aligned = align_clusters(ref, target, n_clusters=2)
    assert np.array_equal(aligned, ref)

    cluster_label_dict = {2: {1: np.array([0, 0, 1, 1]), 2: np.array([1, 1, 0, 0])}}
    aligned_dict = align_cluster_label_dict(cluster_label_dict)

    assert 2 in aligned_dict
    ari_matrix = calculate_ari_matrix(aligned_dict, 2)
    assert ari_matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(ari_matrix), np.ones(2), atol=1e-8)


def test_convert_ndarray_to_list_recursively():
    obj = {"a": np.array([1, 2]), "b": [{"x": np.array([3, 4])}]}

    converted = convert_ndarray_to_list(obj)

    assert converted == {"a": [1, 2], "b": [{"x": [3, 4]}]}


def test_random_forest_cv_and_multiple_seeds_outputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6))
    y = np.array([0] * 15 + [1] * 15)
    feature_names = [f"f{i}" for i in range(6)]

    model, cv_scores, importance_df = run_random_forest_cv(
        X,
        y,
        feature_names=feature_names,
        n_splits=3,
        n_estimators=10,
        random_state=42,
    )

    assert hasattr(model, "feature_importances_")
    assert len(cv_scores) == 3
    assert importance_df.shape[0] == len(feature_names)
    assert "mean_importance" in importance_df.columns

    models, seed_importance_df = run_random_forest_multiple_seeds(
        X,
        y,
        feature_names=feature_names,
        seeds=[0, 1, 2],
        n_estimators=10,
    )
    assert len(models) == 3
    assert "mean_importance" in seed_importance_df.columns


def test_plot_metrics_and_gpt_summary(tmp_path):
    metric_df = pd.DataFrame(
        {
            "fold1_silhouette": [0.20, 0.25, 0.30],
            "fold2_silhouette": [0.21, 0.24, 0.29],
            "fold3_silhouette": [0.19, 0.23, 0.31],
            "fold4_silhouette": [0.22, 0.26, 0.28],
            "fold5_silhouette": [0.20, 0.25, 0.30],
            "mean_ari_5_fold": [0.10, 0.15, 0.12],
        },
        index=[2, 3, 4],
    )
    out = tmp_path / "cluster_metrics.png"

    best_k = plot_clustering_metrics_and_find_best_k(metric_df, filename=str(out))

    assert best_k in {2, 3, 4}
    assert out.exists()

    class MockClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature):
                    class Message:
                        content = "Gene: TP53, Reason: common tumor suppressor"

                    class Choice:
                        message = Message()

                    class Response:
                        choices = [Choice()]

                    return Response()

    result, prompt = gpt_known_genes_summary(MockClient(), ["TP53", "KRAS"], "17p")
    assert "TP53" in result
    assert "17p" in prompt
