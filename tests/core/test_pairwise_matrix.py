"""Tests for pairwise matrix summaries and exports."""

import numpy as np
import pandas as pd
import pytest

from pymaftools.core.PairwiseMatrix import SimilarityMatrix
from pymaftools.core.PivotTable import PivotTable


@pytest.fixture
def similarity_matrix():
    labels = ["s1", "s2", "s3", "s4"]
    return SimilarityMatrix(
        [
            [1.0, 0.2, 0.3, 0.4],
            [0.2, 1.0, 0.5, 0.6],
            [0.3, 0.5, 1.0, 0.8],
            [0.4, 0.6, 0.8, 1.0],
        ],
        index=labels,
        columns=labels,
    )


def test_group_means_exclude_self_similarity_and_mirrored_pairs(similarity_matrix):
    groups = np.array(["A", "A", "B", "B"])

    means = similarity_matrix.get_mean_group_similarity(groups, ["A", "B"])

    assert means.loc["A", "A"] == pytest.approx(0.2)
    assert means.loc["B", "B"] == pytest.approx(0.8)
    assert means.loc["A", "B"] == pytest.approx(0.45)
    assert means.loc["B", "A"] == pytest.approx(0.45)


def test_edges_are_named_and_emitted_once(similarity_matrix):
    edges = similarity_matrix.to_edges_dataframe("cosine", freq_threshold=0.25)

    assert list(edges.columns) == ["source", "target", "frequency", "label"]
    assert len(edges) == 5
    assert not (edges["source"] == edges["target"]).any()
    assert {tuple(edge) for edge in edges[["source", "target"]].to_numpy()} == {
        ("s1", "s3"),
        ("s1", "s4"),
        ("s2", "s3"),
        ("s2", "s4"),
        ("s3", "s4"),
    }


def test_permutation_tests_are_reproducible_and_never_return_zero(similarity_matrix):
    groups = pd.Series(["A", "A", "B", "B"])
    permutations1 = similarity_matrix.generate_permutation_list(
        groups, ["A", "B"], n_permutations=10, random_state=3
    )
    permutations2 = similarity_matrix.generate_permutation_list(
        groups, ["A", "B"], n_permutations=10, random_state=3
    )
    observed = similarity_matrix.get_mean_group_similarity(groups, ["A", "B"])
    pvalues = SimilarityMatrix.calculate_group_similarity_pvalues(
        observed, permutations1, ["A", "B"]
    )

    assert all(first.equals(second) for first, second in zip(permutations1, permutations2))
    assert (pvalues.to_numpy() >= 1 / 11).all()

    statistic1, pvalue1 = similarity_matrix.paired_similarity_permutation_test(
        groups, ("A", "A"), ("B", "B"), n_permutations=20, random_state=9
    )
    statistic2, pvalue2 = similarity_matrix.paired_similarity_permutation_test(
        groups, ("A", "A"), ("B", "B"), n_permutations=20, random_state=9
    )
    assert statistic1 == pytest.approx(-0.6)
    assert (statistic1, pvalue1) == (statistic2, pvalue2)
    assert pvalue1 >= 1 / 21


def test_analysis_accepts_array_groups_and_disabled_pair_test(monkeypatch):
    table = PivotTable(
        pd.DataFrame(
            [[1.0, 0.8, 0.2, 0.1], [0.2, 0.1, 0.8, 1.0]],
            columns=["s1", "s2", "s3", "s4"],
        )
    )
    monkeypatch.setattr(SimilarityMatrix, "plot_similarity", lambda *args, **kwargs: None)
    monkeypatch.setattr(SimilarityMatrix, "plot_heatmap", lambda *args, **kwargs: None)

    result = SimilarityMatrix.analyze_similarity(
        table,
        groups=np.array(["A", "A", "B", "B"]),
        group_order=["A", "B"],
        method="cosine",
        save_dir=None,
        utest_group_pairs=None,
        n_permutations=3,
        random_state=1,
    )

    assert result["pairwise_utest_p"] is None
    assert result["pair1"] is None
    assert result["pair2"] is None


def test_analysis_infers_colors_for_arbitrary_groups(monkeypatch):
    table = PivotTable(
        pd.DataFrame(
            [[1.0, 0.8, 0.2], [0.1, 0.2, 0.9]],
            columns=["s1", "s2", "s3"],
        )
    )
    captured = {}

    def capture_plot(*args, **kwargs):
        captured.update(kwargs["group_cmap"])

    monkeypatch.setattr(SimilarityMatrix, "plot_similarity", capture_plot)
    monkeypatch.setattr(SimilarityMatrix, "plot_heatmap", lambda *args, **kwargs: None)

    SimilarityMatrix.analyze_similarity(
        table,
        groups=np.array(["X", "Y", "Z"]),
        group_order=["X", "Y", "Z"],
        method="cosine",
        group_cmap={"X": "black"},
        save_dir=None,
        n_permutations=2,
        random_state=1,
    )

    assert set(captured) == {"X", "Y", "Z"}
    assert captured["X"] == "black"
