"""Frequency helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def calculate_feature_frequency(table, observation_mask=None) -> pd.Series:
    """Calculate feature frequency across samples."""
    if observation_mask is None:
        observation_mask = table.observation_mask
    if observation_mask is not None:
        from .ObservationMask import ObservationMask

        if not isinstance(observation_mask, ObservationMask):
            raise TypeError("observation_mask must be an ObservationMask.")
        return observation_mask.calculate_feature_frequency(table)
    binary_table = table.to_binary_table()
    return binary_table.sum(axis=1).astype(float) / binary_table.shape[1]


def add_freq(
    table,
    base_table_cls,
    groups: Optional[dict] = None,
    group_col: Optional[str] = None,
    observation_mask=None,
):
    """Add overall and optional group-specific frequency columns."""
    table._validate_metadata()

    if group_col is not None:
        if groups:
            raise ValueError("Pass either 'groups' or 'group_col', not both.")
        if group_col not in table.sample_metadata.columns:
            raise ValueError(f"group_col '{group_col}' not found in sample_metadata.")
        labels = table.sample_metadata[group_col]
        groups = {
            str(v): table.subset(samples=labels == v) for v in labels.dropna().unique()
        }
    groups = groups or {}
    effective_mask = (
        table.observation_mask if observation_mask is None else observation_mask
    )

    pivot_table = table.copy()
    freq_data = pd.DataFrame(index=pivot_table.index)

    for group, group_table in groups.items():
        if not isinstance(group_table, base_table_cls):
            raise TypeError(
                f"Expected PivotTable for group '{group}', got {type(group_table)}."
            )
        group_mask = None
        if effective_mask is not None:
            group_mask = effective_mask.subset(
                features=group_table.index,
                samples=group_table.columns,
            )
        freq_data[f"{group}_freq"] = group_table.calculate_feature_frequency(
            observation_mask=group_mask
        )

    freq_data["freq"] = pivot_table.calculate_feature_frequency(
        observation_mask=effective_mask
    )
    pivot_table.feature_metadata[freq_data.columns] = freq_data
    return pivot_table
