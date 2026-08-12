"""Statistical helpers for :class:`pymaftools.core.PivotTable.PivotTable`."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm
from statsmodels.stats.multitest import multipletests


def mutation_enrichment_test(
    table,
    group_col: str,
    group1: str,
    group2: str,
    alpha: float = 0.05,
    minimum_mutations: int = 2,
    method: Literal["chi2", "fisher"] = "fisher",
    analysis_unit: Literal["sample", "patient"] = "sample",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Perform denominator-aware mutation enrichment between two groups."""
    if method not in {"chi2", "fisher"}:
        raise ValueError(f"Unsupported method: {method}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if minimum_mutations < 0:
        raise ValueError("minimum_mutations must be non-negative.")
    if group1 == group2:
        raise ValueError("group1 and group2 must identify different groups.")
    if group_col not in table.sample_metadata.columns:
        raise ValueError(f"group_col '{group_col}' not found in sample_metadata.")

    table._validate_metadata()
    if table.sample_manifest is not None:
        missing_eligible = table.sample_manifest.eligible_samples.difference(
            table.columns
        )
        if not missing_eligible.empty:
            raise ValueError(
                "Mutation enrichment requires every eligible SampleManifest "
                "sample to be present in the table, including zero-event "
                f"samples; missing: {missing_eligible.tolist()}. Create and "
                "attach a new manifest to declare an intentional analysis "
                "subset."
            )

    if analysis_unit == "patient":
        if table.sample_manifest is None:
            raise ValueError(
                "analysis_unit='patient' requires an attached SampleManifest."
            )
        table.sample_manifest.assert_independent("patient")
    elif analysis_unit != "sample":
        raise ValueError("analysis_unit must be either 'sample' or 'patient'.")

    binary_pivot_table = table.to_binary_table()
    sample_metadata = binary_pivot_table.sample_metadata
    labels = sample_metadata[group_col]
    group1_samples = labels.index[labels.eq(group1)]
    group2_samples = labels.index[labels.eq(group2)]
    if group1_samples.empty or group2_samples.empty:
        raise ValueError("Both requested groups must contain at least one sample.")

    observation_mask = binary_pivot_table.observation_mask
    if observation_mask is None:
        from .ObservationMask import ObservationMask

        observation_mask = ObservationMask.fully_observed(binary_pivot_table)
    observation_mask.calculate_feature_frequency(binary_pivot_table)

    def group_counts(samples: pd.Index) -> tuple[pd.Series, pd.Series]:
        observed = observation_mask.subset(
            features=binary_pivot_table.index,
            samples=samples,
        ).to_frame()
        present = pd.DataFrame(binary_pivot_table.subset(samples=samples)).astype(bool)
        numerator = (present & observed).sum(axis=1).astype(int)
        denominator = observed.sum(axis=1).astype(int)
        return numerator, denominator

    group1_true, group1_denominator = group_counts(group1_samples)
    group2_true, group2_denominator = group_counts(group2_samples)

    df = pd.DataFrame(
        index=binary_pivot_table.index,
        columns=[
            f"{group1}_True",
            f"{group1}_False",
            f"{group2}_True",
            f"{group2}_False",
        ],
    )

    df[f"{group1}_True"] = group1_true
    df[f"{group1}_False"] = group1_denominator - group1_true
    df[f"{group2}_True"] = group2_true
    df[f"{group2}_False"] = group2_denominator - group2_true
    df[f"{group1}_denominator"] = group1_denominator
    df[f"{group2}_denominator"] = group2_denominator
    df["tested"] = (
        (df[f"{group1}_True"] + df[f"{group2}_True"] >= minimum_mutations)
        & df[f"{group1}_denominator"].gt(0)
        & df[f"{group2}_denominator"].gt(0)
    )

    z_value = norm.ppf(0.5 + confidence_level / 2)

    def calculate_row(row: pd.Series) -> pd.Series:
        if not row["tested"]:
            return pd.Series(
                {
                    "odds_ratio": np.nan,
                    "log2_odds_ratio": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                }
            )
        contingency_table = (
            row[
                [
                    f"{group1}_True",
                    f"{group1}_False",
                    f"{group2}_True",
                    f"{group2}_False",
                ]
            ]
            .to_numpy(dtype=int)
            .reshape(2, 2)
        )
        if method == "chi2":
            _, p, _, _ = chi2_contingency(contingency_table)
        else:
            _, p = fisher_exact(contingency_table)

        corrected = contingency_table.astype(float)
        if (corrected == 0).any():
            corrected += 0.5
        a, b, c, d = corrected.ravel()
        odds_ratio = (a * d) / (b * c)
        log_odds = np.log(odds_ratio)
        standard_error = np.sqrt((1 / corrected).sum())
        ci_low = np.exp(log_odds - z_value * standard_error)
        ci_high = np.exp(log_odds + z_value * standard_error)
        return pd.Series(
            {
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": np.log2(odds_ratio),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p,
            }
        )

    statistics = df.apply(calculate_row, axis=1)
    df[statistics.columns] = statistics
    df["adjusted_p_value"] = np.nan
    df["is_significant"] = False
    valid = df["tested"] & df["p_value"].notna()
    if valid.any():
        reject, adjusted_p_value, _, _ = multipletests(
            df.loc[valid, "p_value"].to_numpy(), method="fdr_bh", alpha=alpha
        )
        df.loc[valid, "adjusted_p_value"] = adjusted_p_value
        df.loc[valid, "is_significant"] = reject
    df["test_method"] = method
    df["analysis_unit"] = analysis_unit
    df["confidence_level"] = confidence_level
    df["minimum_total_mutations"] = minimum_mutations
    df["tested_family_size"] = int(valid.sum())
    df["adjustment_method"] = "fdr_bh"

    return df
