"""Filtering helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests


def filter_by_freq(table, threshold: float = 0.05):
    """Filter features by feature_metadata['freq']."""
    if "freq" not in table.feature_metadata.columns:
        raise ValueError(
            "freq column not found in feature_metadata. Please perform table.add_freq() first"
        )
    pivot_table = table.copy()
    return pivot_table.subset(features=pivot_table.feature_metadata.freq >= threshold)


def filter_by_variance(
    table,
    threshold: float = None,
    method: Literal["var", "mad"] = "var",
    quantile: float = None,
):
    """Filter features by variance or median absolute deviation."""
    if threshold is None and quantile is None:
        raise ValueError("Either threshold or quantile must be specified.")
    if method not in ("var", "mad"):
        raise ValueError(f"Unsupported method '{method}'. Use 'var' or 'mad'.")

    pt = table.copy()
    numeric = pt.astype(float)
    if method == "var":
        scores = numeric.var(axis=1)
    else:
        scores = numeric.apply(lambda x: (x - x.median()).abs().median(), axis=1)

    pt.feature_metadata[method] = scores
    if quantile is not None:
        threshold = scores.quantile(quantile)
    return pt.subset(features=scores >= threshold)


def filter_by_statistical_test(
    table,
    group_col: str,
    method: Literal["ttest", "mann_whitney", "kruskal", "anova"] = "kruskal",
    alpha: float = 0.05,
):
    """Filter features by a group-wise statistical test with FDR correction."""
    test_funcs = {
        "ttest": lambda groups: ttest_ind(*groups),
        "mann_whitney": lambda groups: mannwhitneyu(*groups, alternative="two-sided"),
        "kruskal": lambda groups: kruskal(*groups),
        "anova": lambda groups: f_oneway(*groups),
    }
    if method not in test_funcs:
        raise ValueError(f"Unsupported method '{method}'. Choose from {list(test_funcs)}.")

    two_group_only = {"ttest", "mann_whitney"}
    pt = table.copy()
    numeric = pt.astype(float)
    groups_series = pt.sample_metadata[group_col]
    unique_groups = groups_series.dropna().unique()

    if method in two_group_only and len(unique_groups) != 2:
        raise ValueError(f"'{method}' requires exactly 2 groups, got {len(unique_groups)}.")

    p_values = []
    for feature in numeric.index:
        row = numeric.loc[feature]
        group_data = [row[groups_series == g].dropna().values for g in unique_groups]
        if any(len(g) < 2 for g in group_data):
            p_values.append(np.nan)
            continue
        _, pval = test_funcs[method](group_data)
        p_values.append(pval)

    p_values = np.array(p_values)
    valid = ~np.isnan(p_values)
    adjusted = np.full_like(p_values, np.nan)
    if valid.any():
        adjusted[valid] = multipletests(p_values[valid], method="fdr_bh")[1]

    pt.feature_metadata["p_value"] = p_values
    pt.feature_metadata["adjusted_p_value"] = adjusted
    return pt.subset(features=adjusted < alpha)
