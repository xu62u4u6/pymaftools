"""Regression tests for statistical correctness (issues #18, #19, #20)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, fisher_exact

from pymaftools import ObservationMask, PivotTable
from pymaftools.plot.wes import plot_somatic_interactions, somatic_interactions

SAMPLES = list("abcdef")


def _interaction_table(rows, genes=("G1", "G2")) -> PivotTable:
    return PivotTable(rows, index=list(genes), columns=SAMPLES)


def _with_mask(table: PivotTable, observed_rows) -> PivotTable:
    mask = ObservationMask(
        pd.DataFrame(observed_rows, index=table.index, columns=table.columns)
    )
    return table.with_scientific_context(observation_mask=mask)


def _pair(stats: pd.DataFrame, gene1: str, gene2: str) -> pd.Series:
    match = stats[
        ((stats["gene1"] == gene1) & (stats["gene2"] == gene2))
        | ((stats["gene1"] == gene2) & (stats["gene2"] == gene1))
    ]
    assert len(match) == 1
    return match.iloc[0]


# --- #18: somatic_interactions respects ObservationMask ----------------------


def test_interactions_count_only_jointly_observed_samples():
    """Issue #18 reproduction: unobserved samples are not mutation absence."""
    table = _interaction_table([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    masked = _with_mask(table, [[True] * 4 + [False] * 2] * 2)

    masked_stats = somatic_interactions(masked)
    observed_only_stats = somatic_interactions(table.subset(samples=list("abcd")))

    columns = [
        "odds_ratio",
        "p_value",
        "both_mutated",
        "gene2_only",
        "gene1_only",
        "neither_mutated",
    ]
    pd.testing.assert_frame_equal(
        masked_stats[columns].reset_index(drop=True),
        observed_only_stats[columns].reset_index(drop=True),
    )
    row = masked_stats.iloc[0]
    assert row["neither_mutated"] == 1
    assert row["odds_ratio"] == pytest.approx(1.0)
    assert row["jointly_observed"] == 4
    assert bool(row["tested"])


def test_interactions_use_intersection_of_asymmetric_masks():
    """Each pair uses samples observed for both genes, not either one."""
    # G1 (3 events) ranks before G2 (2 events), so G1 is ``gene1``.
    table = _interaction_table([[1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0]])
    masked = _with_mask(
        table,
        [
            [True, True, True, True, False, False],  # G1 observed a-d
            [False, False, True, True, True, True],  # G2 observed c-f
        ],
    )

    row = somatic_interactions(masked).iloc[0]

    # Jointly observed samples are c and d only.
    assert row["jointly_observed"] == 2
    assert [
        row["both_mutated"],
        row["gene2_only"],
        row["gene1_only"],
        row["neither_mutated"],
    ] == [1, 0, 1, 0]
    expected_or, expected_p = fisher_exact([[1, 0], [1, 0]])
    assert row["p_value"] == pytest.approx(expected_p)


def test_interactions_without_joint_observation_are_not_tested():
    """A pair with no jointly observed sample gets no fabricated evidence."""
    table = _interaction_table(
        [[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0]],
        genes=("G1", "G2", "G3"),
    )
    masked = _with_mask(
        table,
        [
            [True, True, True, False, False, False],  # G1: a-c
            [False, False, False, True, True, True],  # G2: d-f
            [True] * 6,
        ],
    )

    stats = somatic_interactions(masked)
    untested = _pair(stats, "G1", "G2")

    assert not bool(untested["tested"])
    assert untested["jointly_observed"] == 0
    assert np.isnan(untested["odds_ratio"])
    assert np.isnan(untested["p_value"])
    assert np.isnan(untested["adjusted_p_value"])
    assert not bool(untested["is_significant"])
    assert untested["interaction"] == "Not tested"
    # BH family = the two testable unordered pairs only.
    assert (stats["tested_family_size"] == 2).all()
    assert stats["tested"].sum() == 2
    # Untested pairs sort last.
    assert not bool(stats.iloc[-1]["tested"])


def test_interactions_fully_observed_behavior_is_unchanged():
    """No mask and an explicit all-observed mask give identical results."""
    table = _interaction_table(
        [[1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]],
        genes=("G1", "G2", "G3"),
    )
    unmasked = somatic_interactions(table)
    masked = somatic_interactions(
        table.with_scientific_context(
            observation_mask=ObservationMask.fully_observed(table)
        )
    )

    pd.testing.assert_frame_equal(unmasked, masked)
    row = _pair(unmasked, "G1", "G2")
    assert row["jointly_observed"] == len(SAMPLES)
    g1 = table.loc["G1"].astype(bool)
    g2 = table.loc["G2"].astype(bool)
    expected_or, expected_p = fisher_exact(
        [
            [int((g1 & g2).sum()), int((~g1 & g2).sum())],
            [int((g1 & ~g2).sum()), int((~g1 & ~g2).sum())],
        ]
    )
    if row["gene1"] == "G1":
        assert row["odds_ratio"] == pytest.approx(expected_or)
    assert row["p_value"] == pytest.approx(expected_p)
    assert (unmasked["tested_family_size"] == 3).all()


def test_interactions_reject_present_event_marked_unobserved():
    table = _interaction_table([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    masked = _with_mask(
        table,
        [[False] + [True] * 5, [True] * 6],  # G1/a is mutated but "unobserved"
    )

    with pytest.raises(ValueError, match="present event"):
        somatic_interactions(masked)


def test_interaction_plot_reuses_corrected_stats_and_blanks_untested_pairs():
    table = _interaction_table(
        [[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0]],
        genes=("G1", "G2", "G3"),
    )
    masked = _with_mask(
        table,
        [
            [True, True, True, False, False, False],
            [False, False, False, True, True, True],
            [True] * 6,
        ],
    )

    fig, stats = plot_somatic_interactions(masked)

    pd.testing.assert_frame_equal(stats, somatic_interactions(masked))
    ax = fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    i_g1, i_g2, i_g3 = labels.index("G1"), labels.index("G2"), labels.index("G3")
    mesh_data = ax.collections[0].get_array().reshape(len(labels), len(labels))
    # Upper-triangle cell for untested (G1, G2) must be masked; tested (G1, G3) must be visible.
    row_u, col_u = min(i_g1, i_g2), max(i_g1, i_g2)
    row_t, col_t = min(i_g1, i_g3), max(i_g1, i_g3)
    assert np.ma.is_masked(mesh_data[row_u, col_u])
    assert not np.ma.is_masked(mesh_data[row_t, col_t])
    plt.close(fig)


# --- #19: empty enrichment inputs -------------------------------------------


def _group_table(rows, index) -> PivotTable:
    table = PivotTable(rows, index=index, columns=["A1", "A2", "B1", "B2"])
    table.sample_metadata["group"] = ["A", "A", "B", "B"]
    return table


def test_enrichment_on_zero_features_returns_documented_schema():
    empty = _group_table([], [])
    reference = _group_table([[True, False, True, False]], ["G"])

    result = empty.mutation_enrichment_test("group", "A", "B")
    expected_columns = reference.mutation_enrichment_test(
        "group", "A", "B", minimum_mutations=0
    ).columns

    assert result.empty
    assert list(result.columns) == list(expected_columns)
    for column in ["p_value", "adjusted_p_value", "tested", "tested_family_size"]:
        assert column in result.columns


def test_enrichment_after_filtering_to_zero_features_returns_empty():
    table = _group_table([[True, False, True, False]], ["G"])
    filtered = table.subset(features=[])

    result = filtered.mutation_enrichment_test("group", "A", "B")

    assert result.empty
    assert "adjusted_p_value" in result.columns


def test_enrichment_on_zero_features_still_validates_groups_and_parameters():
    empty = _group_table([], [])

    with pytest.raises(ValueError, match="at least one sample"):
        empty.mutation_enrichment_test("group", "A", "C")
    with pytest.raises(ValueError, match="different groups"):
        empty.mutation_enrichment_test("group", "A", "A")
    with pytest.raises(ValueError, match="alpha"):
        empty.mutation_enrichment_test("group", "A", "B", alpha=2)


# --- #20: degenerate chi-squared tables -------------------------------------


def test_chi2_marks_all_mutated_gene_untested():
    table = _group_table([[True, True, True, True]], ["G"])

    result = table.mutation_enrichment_test("group", "A", "B", method="chi2")

    assert not bool(result.loc["G", "tested"])
    assert result.loc["G", "not_tested_reason"] == "degenerate_contingency_table"
    assert np.isnan(result.loc["G", "p_value"])
    assert np.isnan(result.loc["G", "odds_ratio"])
    assert result.loc["G", "tested_family_size"] == 0
    assert result.loc["G", "test_method"] == "chi2"


def test_chi2_marks_all_absent_gene_untested_with_zero_minimum():
    table = _group_table([[False, False, False, False]], ["G"])

    result = table.mutation_enrichment_test(
        "group", "A", "B", method="chi2", minimum_mutations=0
    )

    assert not bool(result.loc["G", "tested"])
    assert result.loc["G", "not_tested_reason"] == "degenerate_contingency_table"
    assert np.isnan(result.loc["G", "p_value"])


def test_chi2_mixed_batch_tests_valid_rows_and_excludes_degenerate_from_fdr():
    table = _group_table(
        [
            [True, True, True, True],  # degenerate: all mutated
            [True, True, False, False],  # testable
            [True, False, True, True],  # testable
        ],
        ["ALL", "SPLIT", "MOSTLY"],
    )

    result = table.mutation_enrichment_test("group", "A", "B", method="chi2")

    assert result.loc["ALL", "not_tested_reason"] == "degenerate_contingency_table"
    assert result["tested"].tolist() == [False, True, True]
    assert (result["tested_family_size"] == 2).all()
    for gene, contingency in {
        "SPLIT": [[2, 0], [0, 2]],
        "MOSTLY": [[1, 1], [2, 0]],
    }.items():
        _, expected_p, _, _ = chi2_contingency(contingency)
        assert result.loc[gene, "p_value"] == pytest.approx(expected_p)
        assert pd.isna(result.loc[gene, "not_tested_reason"])
    assert np.isnan(result.loc["ALL", "adjusted_p_value"])


def test_fisher_results_unchanged_for_zero_margin_tables():
    """Fisher is defined on a zero-margin table, so it remains tested."""
    table = _group_table(
        [[True, True, True, True], [True, True, False, False]], ["ALL", "SPLIT"]
    )

    result = table.mutation_enrichment_test("group", "A", "B", method="fisher")

    assert result["tested"].all()
    assert result.loc["ALL", "p_value"] == pytest.approx(1.0)
    _, expected_p = fisher_exact([[2, 0], [0, 2]])
    assert result.loc["SPLIT", "p_value"] == pytest.approx(expected_p)
    assert (result["tested_family_size"] == 2).all()


def test_enrichment_reports_reason_for_below_minimum_rows():
    table = _group_table([[True, False, False, False]], ["G"])

    result = table.mutation_enrichment_test("group", "A", "B", minimum_mutations=2)

    assert result.loc["G", "not_tested_reason"] == "below_minimum_mutations"
