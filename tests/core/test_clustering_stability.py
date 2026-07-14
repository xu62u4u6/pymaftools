"""Regression tests for cross-fold clustering stability."""

import numpy as np
import pandas as pd

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
        assert table.sample_metadata.loc[held_out, "subtype"].value_counts().to_dict() == {
            "A": 2,
            "B": 2,
        }


def test_k_fold_rejects_groups_smaller_than_number_of_splits():
    table = PivotTable(pd.DataFrame(np.eye(6), columns=[f"s{i}" for i in range(6)]))
    table.sample_metadata["subtype"] = ["A", "A", "B", "B", "B", "B"]

    with np.testing.assert_raises_regex(ValueError, "needs at least 3 samples"):
        k_fold_clustering_evaluation(table, group_col="subtype", n_splits=3)
