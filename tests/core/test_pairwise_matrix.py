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

    assert all(
        first.equals(second) for first, second in zip(permutations1, permutations2)
    )
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
    monkeypatch.setattr(
        SimilarityMatrix, "plot_similarity", lambda *args, **kwargs: None
    )
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


def test_group_pvalues_are_nan_when_observed_similarity_is_undefined():
    # NaN compares False, which used to give the minimum p = 1/(N+1).
    groups = ["A", "B"]
    observed = pd.DataFrame([[np.nan, 0.1], [0.1, 0.5]], index=groups, columns=groups)
    rng = np.random.default_rng(0)
    permutations = [
        pd.DataFrame(rng.random((2, 2)), index=groups, columns=groups)
        for _ in range(99)
    ]

    pvalues = SimilarityMatrix.calculate_group_similarity_pvalues(
        observed, permutations, groups
    )

    assert np.isnan(pvalues.loc["A", "A"])
    assert pvalues.loc["B", "B"] > 1 / 100


def test_group_pvalues_exclude_nan_permutations_from_denominator():
    groups = ["A"]
    observed = pd.DataFrame([[0.5]], index=groups, columns=groups)
    permutations = [
        pd.DataFrame([[v]], index=groups, columns=groups)
        for v in [0.9, np.nan, 0.1, np.nan]
    ]

    pvalues = SimilarityMatrix.calculate_group_similarity_pvalues(
        observed, permutations, groups, tail="two"
    )

    # Finite permutations 0.9 and 0.1 (mean 0.5); both are as extreme as 0.
    assert pvalues.loc["A", "A"] == pytest.approx((2 + 1) / (2 + 1))


def test_group_series_is_aligned_by_sample_id_not_position():
    samples = ["s1", "s2", "s3", "s4"]
    matrix = SimilarityMatrix(
        np.array(
            [
                [1.0, 0.9, 0.1, 0.1],
                [0.9, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.9],
                [0.1, 0.1, 0.9, 1.0],
            ]
        ),
        index=samples,
        columns=samples,
    )
    aligned = pd.Series(["A", "A", "B", "B"], index=samples)
    reordered = aligned.loc[["s3", "s1", "s4", "s2"]]

    result = matrix.get_mean_group_similarity(reordered, group_order=["A", "B"])

    assert result.loc["A", "A"] == pytest.approx(0.9)
    assert result.loc["A", "B"] == pytest.approx(0.1)


def test_group_series_missing_samples_raises_with_hint():
    samples = ["s1", "s2"]
    matrix = SimilarityMatrix(np.eye(2), index=samples, columns=samples)

    with pytest.raises(ValueError, match="to_numpy"):
        matrix.get_mean_group_similarity(pd.Series(["A", "B"], index=["s1", "x"]))


@pytest.mark.parametrize(
    ("tail", "expected"),
    [("right", (1 + 1) / (2 + 1)), ("left", (1 + 1) / (2 + 1))],
)
def test_group_pvalues_one_sided_tails_ignore_nan_permutations(tail, expected):
    groups = ["A"]
    observed = pd.DataFrame([[0.5]], index=groups, columns=groups)
    permutations = [
        pd.DataFrame([[v]], index=groups, columns=groups)
        for v in [0.9, np.nan, 0.1, np.nan]
    ]

    pvalues = SimilarityMatrix.calculate_group_similarity_pvalues(
        observed, permutations, groups, tail=tail
    )

    assert pvalues.loc["A", "A"] == pytest.approx(expected)


def test_group_pvalues_all_nan_permutations_give_nan_and_bad_tail_raises():
    groups = ["A"]
    observed = pd.DataFrame([[np.nan]], index=groups, columns=groups)
    permutations = [pd.DataFrame([[np.nan]], index=groups, columns=groups)] * 3
    finite_observed = pd.DataFrame([[0.5]], index=groups, columns=groups)

    pvalues = SimilarityMatrix.calculate_group_similarity_pvalues(
        finite_observed, permutations, groups
    )
    assert np.isnan(pvalues.loc["A", "A"])
    with pytest.raises(ValueError, match="tail"):
        SimilarityMatrix.calculate_group_similarity_pvalues(
            observed, permutations, groups, tail="both"
        )


def test_paired_similarity_permutation_test_holds_anchor_and_ignores_unrelated_groups():
    # Groups: A (anchor, 2 samples), B (2 samples), C (2 samples), D (unrelated, 2 samples)
    labels = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]
    groups = pd.Series(["A", "A", "B", "B", "C", "C", "D", "D"], index=labels)

    def _make_matrix(d_fill: float) -> SimilarityMatrix:
        mat = np.full((8, 8), 0.1, dtype=float)
        np.fill_diagonal(mat, 1.0)
        # A vs B = 0.8, A vs C = 0.2
        mat[0:2, 2:4] = mat[2:4, 0:2] = 0.8
        mat[0:2, 4:6] = mat[4:6, 0:2] = 0.2
        # Unrelated group D has arbitrary extreme similarities
        mat[6:8, :6] = mat[:6, 6:8] = d_fill
        mat[6, 7] = mat[7, 6] = d_fill
        return SimilarityMatrix(mat, index=labels, columns=labels)

    m_low = _make_matrix(-100.0)
    m_high = _make_matrix(100.0)

    # Shared-anchor (A, B) vs (A, C): unrelated D must have zero effect on stat or p-value
    stat_low, p_low = m_low.paired_similarity_permutation_test(
        groups, ("A", "B"), ("A", "C"), n_permutations=200, random_state=42
    )
    stat_high, p_high = m_high.paired_similarity_permutation_test(
        groups, ("A", "B"), ("A", "C"), n_permutations=200, random_state=42
    )
    assert stat_low == pytest.approx(0.6)
    assert (stat_low, p_low) == (stat_high, p_high)

    # Within-vs-within (A, A) vs (B, B): unrelated C and D must also have zero effect
    w_low = m_low.paired_similarity_permutation_test(
        groups, ("A", "A"), ("B", "B"), n_permutations=100, random_state=7
    )
    w_high = m_high.paired_similarity_permutation_test(
        groups, ("A", "A"), ("B", "B"), n_permutations=100, random_state=7
    )
    assert w_low == w_high

    # Anchor invariance check: every sample in B and C has identical similarity (0.5) to A,
    # while A-A is 0.99 and B-C is 0.0. Holding A fixed means every B<->C swap yields
    # exact difference 0.0 -> p must equal 1.0. If A were permuted into B/C, diffs would vary.
    mat_eq = np.zeros((6, 6), dtype=float)
    np.fill_diagonal(mat_eq, 1.0)
    mat_eq[0, 1] = mat_eq[1, 0] = 0.99
    mat_eq[0:2, 2:6] = mat_eq[2:6, 0:2] = 0.5
    eq_matrix = SimilarityMatrix(mat_eq, index=labels[:6], columns=labels[:6])
    stat_eq, p_eq = eq_matrix.paired_similarity_permutation_test(
        groups.iloc[:6], ("A", "B"), ("A", "C"), n_permutations=50, random_state=1
    )
    assert stat_eq == pytest.approx(0.0)
    assert p_eq == pytest.approx(1.0)


def test_get_pairs_subset_aligns_group_series_by_sample_id():
    samples = ["s1", "s2", "s3", "s4"]
    matrix = SimilarityMatrix(
        np.arange(16, dtype=float).reshape(4, 4), index=samples, columns=samples
    )
    groups = pd.Series(["A", "A", "B", "B"], index=samples).loc[
        ["s4", "s2", "s3", "s1"]
    ]

    pair1, _ = matrix.get_pairs_subset(groups, ("A", "B"), ("A", "A"))

    assert list(pair1.index) == ["s1", "s2"]
    assert list(pair1.columns) == ["s3", "s4"]


def test_group_series_with_duplicate_sample_ids_raises():
    samples = ["s1", "s2"]
    matrix = SimilarityMatrix(np.eye(2), index=samples, columns=samples)

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        matrix.get_mean_group_similarity(
            pd.Series(["A", "B", "B"], index=["s1", "s2", "s2"])
        )
