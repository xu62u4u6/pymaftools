"""Burden- and subtype-stratified somatic interactions (Mantel-Haenszel)."""

import numpy as np
import pandas as pd
import pytest

from pymaftools.core.SmallVariationTable import SmallVariationTable


def _burden_confounded_table() -> SmallVariationTable:
    """100 samples where GENE_A and GENE_B are independent within each burden group.

    Low-burden samples (0-49): A in 10, B in 10, both in 2 (= 10 * 10 / 50).
    High-burden samples (50-99): A in 40, B in 40, both in 32 (= 40 * 40 / 50),
    and 20 filler genes mutated in 45 of them. Pooled, A and B look co-mutated
    only because high-burden samples carry more mutations in every gene.
    """
    samples = [f"S{i:03d}" for i in range(100)]
    low, high = range(50), range(50, 100)
    a = np.zeros(100, dtype=bool)
    b = np.zeros(100, dtype=bool)
    a[list(low)[:10]] = True
    b[[i for i in low if i % 5 == 0]] = True
    a[list(high)[:40]] = True
    b[[i for i in high if (i - 50) % 5 != 4]] = True
    rows = {"GENE_A": a, "GENE_B": b}
    for g in range(20):
        filler = np.zeros(100, dtype=bool)
        filler[[i for i in high if (i + g) % 10 != 0]] = True
        rows[f"FILLER{g:02d}"] = filler
    frame = pd.DataFrame(rows, index=samples).T
    values = frame.replace({True: "Missense_Mutation", False: False})
    return SmallVariationTable(values)


def test_default_test_method_is_fisher():
    stats = _burden_confounded_table().somatic_interactions(top=2)
    row = stats.iloc[0]
    assert row["test_method"] == "fisher"
    assert row["n_strata"] == 1


def test_burden_strata_removes_burden_driven_cooccurrence():
    table = _burden_confounded_table()
    fisher = table.somatic_interactions(top=2)
    stratified = table.somatic_interactions(top=2, burden_strata=2)

    fisher_row = fisher.iloc[0]
    assert {fisher_row["gene1"], fisher_row["gene2"]} == {"GENE_A", "GENE_B"}
    assert fisher_row["interaction"] == "Co-occurrence"
    assert fisher_row["is_significant"]

    row = stratified.iloc[0]
    assert row["test_method"] == "mantel_haenszel"
    assert row["n_strata"] == 2
    assert row["odds_ratio"] == pytest.approx(1.0)
    assert not row["is_significant"]
    # pooled counts are still reported over all jointly observed samples
    assert row["both_mutated"] == fisher_row["both_mutated"] == 34


def test_strata_col_removes_subtype_driven_exclusivity():
    samples = [f"S{i:02d}" for i in range(40)]
    a = np.array([i < 10 for i in range(40)])  # only in subtype X (0-19)
    b = np.array([20 <= i < 30 for i in range(40)])  # only in subtype Y (20-39)
    values = pd.DataFrame({"GENE_A": a, "GENE_B": b}, index=samples).T.replace(
        {True: "Missense_Mutation", False: False}
    )
    table = SmallVariationTable(values)
    table.sample_metadata["subtype"] = ["X"] * 20 + ["Y"] * 20

    pooled = table.somatic_interactions(top=2).iloc[0]
    assert pooled["interaction"] == "Mutual exclusivity"

    within = table.somatic_interactions(top=2, strata_col="subtype").iloc[0]
    # Within each subtype one gene is never mutated: no informative stratum.
    assert not within["tested"]
    assert within["n_strata"] == 0
    assert not within["is_significant"]


def test_stratified_result_is_symmetric_in_gene_order():
    table = _burden_confounded_table()
    stats = table.somatic_interactions(top=2, burden_strata=2)
    swapped = table.subset(features=["GENE_B", "GENE_A"] + list(table.index[2:]))
    stats_swapped = swapped.somatic_interactions(top=2, burden_strata=2)
    assert stats.iloc[0]["p_value"] == pytest.approx(stats_swapped.iloc[0]["p_value"])
    assert stats.iloc[0]["odds_ratio"] == pytest.approx(
        stats_swapped.iloc[0]["odds_ratio"]
    )


def test_invalid_stratification_arguments_fail_loud():
    table = _burden_confounded_table()
    with pytest.raises(ValueError, match="burden_strata=4"):
        table.somatic_interactions(top=2, burden_strata=1)
    with pytest.raises(ValueError, match="Available columns"):
        table.somatic_interactions(top=2, strata_col="subtype")


def test_stratification_is_recorded_in_provenance():
    stats = _burden_confounded_table().somatic_interactions(top=2, burden_strata=2)
    params = stats.attrs["pymaftools_provenance"]["parameters"]
    assert params["burden_strata"] == 2
    assert params["test_method"] == "mantel_haenszel"


def test_burden_uses_mutations_count_so_feature_subsets_keep_full_burden():
    table = _burden_confounded_table()
    burden = table.to_binary_table().astype(bool).sum(axis=0)
    table.sample_metadata["mutations_count"] = burden.to_numpy() * 10
    drivers_only = table.subset(features=["GENE_A", "GENE_B"])

    stats = drivers_only.somatic_interactions(top=2, burden_strata=2)
    params = stats.attrs["pymaftools_provenance"]["parameters"]
    assert params["burden_source"] == "sample_metadata.mutations_count"
    # Burden still separates the two groups, so the confounded pair is null.
    assert stats.iloc[0]["n_strata"] == 2
    assert stats.iloc[0]["odds_ratio"] == pytest.approx(1.0)
