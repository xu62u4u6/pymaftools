"""Statistical helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


def mutation_enrichment_test(
    table,
    group_col: str,
    group1: str,
    group2: str,
    alpha: float = 0.05,
    minimum_mutations: int = 2,
    method: Literal["chi2", "fisher"] = "chi2",
) -> pd.DataFrame:
    """Perform mutation enrichment testing between two sample groups."""
    binary_pivot_table = table.to_binary_table()
    sample_metadata = binary_pivot_table.sample_metadata

    subset1 = binary_pivot_table.subset(samples=sample_metadata[group_col] == group1)
    subset2 = binary_pivot_table.subset(samples=sample_metadata[group_col] == group2)

    df = pd.DataFrame(
        index=binary_pivot_table.index,
        columns=[
            f"{group1}_True",
            f"{group1}_False",
            f"{group2}_True",
            f"{group2}_False",
        ],
    )

    df[f"{group1}_True"] = subset1.sum(axis=1)
    df[f"{group1}_False"] = len(subset1.columns) - df[f"{group1}_True"]

    df[f"{group2}_True"] = subset2.sum(axis=1)
    df[f"{group2}_False"] = len(subset2.columns) - df[f"{group2}_True"]

    df = df[
        (df[f"{group1}_True"] >= minimum_mutations)
        | (df[f"{group2}_True"] >= minimum_mutations)
    ]

    def get_p_value(row: pd.Series) -> float:
        contingency_table = row.values.astype(int).reshape(2, 2)
        if method == "chi2":
            _, p, _, _ = chi2_contingency(contingency_table)
        elif method == "fisher":
            _, p = fisher_exact(contingency_table)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return p

    df["p_value"] = df.apply(get_p_value, axis=1)

    p_values = df["p_value"].values
    reject, adjusted_p_value, _, _ = multipletests(
        p_values, method="fdr_bh", alpha=alpha
    )

    df["adjusted_p_value"] = adjusted_p_value
    df["is_significant"] = reject
    df["test_method"] = method

    return df
