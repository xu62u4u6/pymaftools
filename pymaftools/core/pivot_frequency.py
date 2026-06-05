"""Frequency helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def calculate_feature_frequency(table) -> pd.Series:
    """Calculate feature frequency across samples."""
    binary_table = table.to_binary_table()
    return binary_table.sum(axis=1).astype(float) / binary_table.shape[1]


def add_freq(table, base_table_cls, groups: Optional[dict] = None, group_col: Optional[str] = None):
    """Add overall and optional group-specific frequency columns."""
    table._validate_metadata()

    if group_col is not None:
        if groups:
            raise ValueError("Pass either 'groups' or 'group_col', not both.")
        if group_col not in table.sample_metadata.columns:
            raise ValueError(f"group_col '{group_col}' not found in sample_metadata.")
        labels = table.sample_metadata[group_col]
        groups = {
            str(v): table.subset(samples=labels == v)
            for v in labels.dropna().unique()
        }
    groups = groups or {}

    pivot_table = table.copy()
    freq_data = pd.DataFrame(index=pivot_table.index)

    for group, group_table in groups.items():
        if not isinstance(group_table, base_table_cls):
            raise TypeError(
                f"Expected PivotTable for group '{group}', got {type(group_table)}."
            )
        freq_data[f"{group}_freq"] = group_table.calculate_feature_frequency()

    freq_data["freq"] = pivot_table.calculate_feature_frequency()
    pivot_table.feature_metadata[freq_data.columns] = freq_data
    return pivot_table
