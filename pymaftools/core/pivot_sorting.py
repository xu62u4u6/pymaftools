"""Sorting helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import List

import pandas as pd


def sort_features(
    table, by: str | List[str] = "freq", ascending: bool | List[bool] = False
):
    """Sort features by one or more feature metadata columns."""
    cols = [by] if isinstance(by, str) else list(by)
    missing = [c for c in cols if c not in table.feature_metadata.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in feature_metadata.")
    table = table.copy()
    sorted_index = table.feature_metadata.sort_values(by=by, ascending=ascending).index
    return table.subset(features=sorted_index)


def sort_samples_by_mutations(table, top: int = 10):
    """Sort samples by mutation patterns encoded from the top features."""

    def binary_sort_key(column: pd.Series) -> int:
        binary_str = "".join(column.astype(int).astype(str))
        return int(binary_str, 2)

    pivot_table = table.copy()
    binary_pivot_table = pivot_table.to_binary_table()
    mutations_weight = binary_pivot_table.head(top).apply(binary_sort_key, axis=0)
    pivot_table.sample_metadata["mutations_weight"] = mutations_weight
    sorted_samples = mutations_weight.sort_values(ascending=False).index.tolist()
    return pivot_table.subset(samples=sorted_samples)


def sort_samples_by_group(table, group_col: str, group_order: List[str], top: int = 10):
    """Sort listed groups first without dropping samples from other groups."""
    pivot_table = table.copy()

    if group_col not in pivot_table.sample_metadata.columns:
        raise ValueError(f"Column '{group_col}' not found in sample_metadata.")
    if len(group_order) != len(set(group_order)):
        raise ValueError("group_order must not contain duplicate values")

    sorted_samples = []
    for subtype in group_order:
        subtype_samples = pivot_table.sample_metadata[
            pivot_table.sample_metadata[group_col] == subtype
        ].index

        if len(subtype_samples) > 0:
            subtype_pivot = pivot_table.subset(samples=subtype_samples)
            sorted_subtype_pivot = subtype_pivot.sort_samples_by_mutations(top=top)
            sorted_samples.extend(sorted_subtype_pivot.columns)

    remaining_samples = pivot_table.columns[~pivot_table.columns.isin(sorted_samples)]
    sorted_samples.extend(remaining_samples)

    return pivot_table.subset(samples=sorted_samples)
