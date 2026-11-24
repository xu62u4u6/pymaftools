"""Regression tests for cross-fold clustering stability."""

import numpy as np
import pandas as pd
import pytest

from pymaftools.core.Clustering import (
    align_cluster_label_dict,
    calculate_ari_matrix,
    k_fold_clustering_evaluation,
    table_to_distance,
)
from pymaftools.core.PivotTable import PivotTable


def test_fold_alignment_uses_shared_sample_identifiers():
    labels = {
        2: {
            1: pd.Series([0, 0, 1], index=["a", "b", "c"]),
            2: pd.Series([0, 1, 1], index=["b", "c", "d"]),
        }
    }

    aligned = align_cluster_label_dict(labels)[2]
    ari = calculate_ari_matrix({2: aligned}, 2)

    assert list(aligned.index) == ["a", "b", "c", "d"]
    assert pd.isna(aligned.loc["a", 2])
    assert pd.isna(aligned.loc["d", 1])
    assert ari.loc[1, 2] == 1.0


def test_distance_matrix_is_sample_by_sample_for_non_square_table():
    table = PivotTable(
        pd.DataFrame(
            np.arange(30).reshape(6, 5),
            columns=[f"sample-{i}" for i in range(5)],
        )
    )

    distance = table_to_distance(table)

    assert distance.shape == (5, 5)


def test_k_fold_labels_retain_training_sample_ids():
    rng = np.random.default_rng(7)
    sample_ids = [f"sample-{i}" for i in range(10)]
    table = PivotTable(pd.DataFrame(rng.random((6, 10)), columns=sample_ids))
    table.sample_metadata["subtype"] = ["A", "B"] * 5

    results, labels = k_fold_clustering_evaluation(
        table,
        min_clusters=2,
        max_clusters=2,
        random_state=7,
    )

    assert len(results) == 5
    assert set(labels[2]) == {1, 2, 3, 4, 5}
    for fold_labels in labels[2].values():
        assert isinstance(fold_labels, pd.Series)
        assert fold_labels.index.isin(table.columns).all()
        assert len(fold_labels) == 8
        assert np.issubdtype(fold_labels.dtype, np.integer)


def test_k_fold_stratifies_only_when_group_column_is_requested():
    rng = np.random.default_rng(8)
    sample_ids = [f"sample-{i}" for i in range(12)]
    table = PivotTable(pd.DataFrame(rng.random((6, 12)), columns=sample_ids))
    table.sample_metadata["subtype"] = ["A"] * 6 + ["B"] * 6

    _, labels = k_fold_clustering_evaluation(
        table,
        min_clusters=2,
        max_clusters=2,
        group_col="subtype",
        n_splits=3,
        random_state=2,
    )

    for fold_labels in labels[2].values():
        held_out = table.columns.difference(fold_labels.index)
        assert table.sample_metadata.loc[
            held_out, "subtype"
        ].value_counts().to_dict() == {
            "A": 2,
            "B": 2,
        }


def test_k_fold_rejects_groups_smaller_than_number_of_splits():
    table = PivotTable(pd.DataFrame(np.eye(6), columns=[f"s{i}" for i in range(6)]))
    table.sample_metadata["subtype"] = ["A", "A", "B", "B", "B", "B"]

    with np.testing.assert_raises_regex(ValueError, "needs at least 3 samples"):
        k_fold_clustering_evaluation(table, group_col="subtype", n_splits=3)


def test_align_cluster_labels_keeps_cluster_absent_from_common_samples_distinct():
    from pymaftools.core.Clustering import align_cluster_label_dict

    reference = pd.Series([0, 0, 2, 2], index=list("abcd"))
    # Target cluster 2 only appears on 'e', which the reference fold lacks.
    target = pd.Series([1, 1, 0, 0, 2], index=list("abcde"))

    aligned = align_cluster_label_dict({3: {1: reference, 2: target}})[3][2]

    assert aligned.loc[["a", "b"]].tolist() == [0, 0]
    assert aligned.loc[["c", "d"]].tolist() == [2, 2]
    assert aligned.loc["e"] == 1
    assert set(aligned) == {0, 1, 2}


def test_align_cluster_labels_rejects_labels_outside_zero_to_k_minus_one():
    import pytest

    from pymaftools.core.Clustering import align_cluster_label_dict

    reference = pd.Series([1, 1, 2, 2], index=list("abcd"))  # 1-indexed
    target = pd.Series([1, 1, 2, 2], index=list("abcd"))

    with pytest.raises(ValueError, match="0..1"):
        align_cluster_label_dict({2: {1: reference, 2: target}})


def test_table_to_distance_handles_zero_vector_for_precomputed_silhouette():
    from sklearn.metrics import silhouette_score

    table = PivotTable(
        pd.DataFrame(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            index=["F1", "F2"],
            columns=["S1", "S2", "S_zero", "S4"],
        )
    )

    distance = table_to_distance(table)
    assert np.all(distance >= 0.0)
    np.testing.assert_array_equal(np.diag(distance), np.zeros(4))
    score = silhouette_score(distance, [0, 0, 1, 1], metric="precomputed")
    assert -1.0 <= score <= 1.0


def test_run_feature_clustering_caps_max_clusters_and_handles_zero_vectors(tmp_path):
    import pytest

    from pymaftools.core.Clustering import run_feature_clustering

    feat_df = PivotTable(
        pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.1, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            index=["F1", "F2", "F3", "F_zero"],
            columns=["S1", "S2", "S3", "S4"],
        )
    )
    out_csv = tmp_path / "feature_clustering.csv"
    res = run_feature_clustering(feat_df, result_path=str(out_csv), max_clusters=10)
    assert list(res["n_clusters"]) == [2, 3]
    assert out_csv.exists()

    with pytest.raises(ValueError, match="at least 3 features"):
        run_feature_clustering(
            feat_df.head(2), result_path=str(out_csv), max_clusters=5
        )


def test_plot_cluster_feature_importance_accepts_features_count_and_gene_count():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pymaftools.core.Clustering import (
        plot_cluster_feature_importance,
        plot_cluster_feature_importance_boxplot,
    )

    base = pd.DataFrame(
        {
            "mean_importance": [0.5, 0.3, 0.2],
            "unique_chr_arm": ["1p", "2q", "17p"],
            "importance_1": [0.4, 0.3, 0.2],
            "importance_2": [0.6, 0.3, 0.2],
        },
        index=[0, 1, 2],
    )
    for col in ("features_count", "gene_count"):
        df = base.copy()
        df[col] = [10, 5, 3]
        plot_cluster_feature_importance(df, ["importance_1", "importance_2"])
        plt.close("all")
        plot_cluster_feature_importance_boxplot(df, ["importance_1", "importance_2"])
        plt.close("all")


def test_plot_clustering_metrics_dynamic_folds_and_explicit_format(tmp_path):
    from pymaftools.core.Clustering import plot_clustering_metrics_and_find_best_k

    metric_df = pd.DataFrame(
        {
            "fold1_silhouette": [0.20, 0.40, 0.30],
            "fold2_silhouette": [0.22, 0.42, 0.28],
            "fold3_silhouette": [0.18, 0.38, 0.32],
            "mean_ari_3_fold": [0.10, 0.35, 0.20],
        },
        index=[2, 3, 4],
    )
    original_cols = list(metric_df.columns)
    out_dir = tmp_path / "dir.v1"
    out_dir.mkdir()
    out_file = out_dir / "metrics_no_ext"

    best_k = plot_clustering_metrics_and_find_best_k(
        metric_df, filename=str(out_file), format="png"
    )
    assert best_k == 3
    assert list(metric_df.columns) == original_cols
    assert out_file.exists()
    assert out_file.read_bytes().startswith(b"\x89PNG")

    mismatched_df = metric_df.rename(columns={"mean_ari_3_fold": "mean_ari_5_fold"})
    with pytest.raises(ValueError, match="mean_ari_3_fold"):
        plot_clustering_metrics_and_find_best_k(mismatched_df, filename=str(out_file))
