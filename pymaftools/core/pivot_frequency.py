"""Frequency helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import Optional
import warnings

import pandas as pd


def _warn_if_manifest_is_incomplete(table) -> None:
    manifest = table.sample_manifest
    if manifest is None:
        return
    missing = manifest.eligible_samples.difference(table.columns)
    if missing.empty:
        return
    examples = missing[:3].tolist()
    warnings.warn(
        f"Frequency uses the {len(table.columns)} sample(s) in table.columns, "
        f"but {len(missing)} eligible SampleManifest sample(s) are missing "
        f"(examples: {examples}). If this is an intentional subgroup, use "
        "SampleManifest.subset(), for example "
        "table.sample_manifest = table.sample_manifest.subset(table.columns), "
        "before calculating frequencies.",
        UserWarning,
        stacklevel=3,
    )


def _calculate_feature_frequency(table, observation_mask=None) -> pd.Series:
    if observation_mask is None:
        observation_mask = table.observation_mask
    if observation_mask is not None:
        from .ObservationMask import ObservationMask

        if not isinstance(observation_mask, ObservationMask):
            raise TypeError("observation_mask must be an ObservationMask.")
        return observation_mask.calculate_feature_frequency(table)
    binary_table = table.to_binary_table()
    return binary_table.sum(axis=1).astype(float) / binary_table.shape[1]


def calculate_feature_frequency(table, observation_mask=None) -> pd.Series:
    """Calculate feature frequency across samples."""
    _warn_if_manifest_is_incomplete(table)
    return _calculate_feature_frequency(table, observation_mask=observation_mask)


def add_freq(
    table,
    base_table_cls,
    groups: Optional[dict] = None,
    group_col: Optional[str] = None,
    observation_mask=None,
):
    """Add overall and optional group-specific frequency columns."""
    table._validate_metadata()
    _warn_if_manifest_is_incomplete(table)

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
        freq_data[f"{group}_freq"] = _calculate_feature_frequency(
            group_table, observation_mask=group_mask
        )

    freq_data["freq"] = _calculate_feature_frequency(
        pivot_table, observation_mask=effective_mask
    )
    pivot_table.feature_metadata[freq_data.columns] = freq_data
    return pivot_table


def add_mutation_density(
    table,
    *,
    col: str = "mutation_density_per_kb",
    size_col: str = "exon_size",
    metric: str = "cds_length",
    observation_mask=None,
    force_download: bool = False,
):
    """Add length-normalized mutation density (mutated-sample frequency per kb)."""
    table._validate_metadata()
    _warn_if_manifest_is_incomplete(table)

    pivot_table = table.copy()
    if size_col not in pivot_table.feature_metadata.columns:
        pivot_table = pivot_table.add_exon_size(
            col=size_col,
            metric=metric,
            force_download=force_download,
        )

    effective_mask = (
        pivot_table.observation_mask if observation_mask is None else observation_mask
    )
    freq = _calculate_feature_frequency(pivot_table, observation_mask=effective_mask)
    sizes = pd.to_numeric(pivot_table.feature_metadata[size_col], errors="coerce")
    valid_kb = sizes.where(sizes > 0) / 1000.0
    pivot_table.feature_metadata[col] = (freq / valid_kb).astype(float)
    return pivot_table


def add_flags_annotation(
    table,
    *,
    col: str = "is_flag_gene",
    flags_genes=None,
):
    """Annotate whether each feature belongs to the FLAGS frequently mutated gene list."""
    from ..utils.geneinfo import load_flags_genes

    table._validate_metadata()
    flag_set = set(load_flags_genes() if flags_genes is None else flags_genes)
    pivot_table = table.copy()
    pivot_table.feature_metadata[col] = pd.Series(
        pivot_table.index.isin(flag_set),
        index=pivot_table.index,
        dtype=bool,
    )
    return pivot_table
