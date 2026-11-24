"""Shared test fixtures for pymaftools tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymaftools.core.Cohort import Cohort
from pymaftools.core.PivotTable import PivotTable


@pytest.fixture
def sample_mutation_data():
    """Create sample binary mutation data for testing."""
    data = {
        "sample1": [True, False, True, False],
        "sample2": [False, True, True, False],
        "sample3": [True, True, False, True],
        "sample4": [False, False, True, True],
    }
    return pd.DataFrame(data, index=["TP53", "KRAS", "EGFR", "PIK3CA"])


@pytest.fixture
def sample_pivot_table(sample_mutation_data):
    """Create a sample PivotTable with sample and feature metadata."""
    table = PivotTable(sample_mutation_data)

    table.sample_metadata["subtype"] = ["LUAD", "LUSC", "ASC", "LUAD"]
    table.sample_metadata["age"] = [65, 70, 55, 60]
    table.sample_metadata["sex"] = ["M", "F", "M", "F"]

    table.feature_metadata["chromosome"] = ["17p13.1", "12p12.1", "7p11.2", "3q26.32"]
    table.feature_metadata["gene_type"] = [
        "tumor_suppressor",
        "oncogene",
        "oncogene",
        "oncogene",
    ]
    return table


@pytest.fixture
def sample_cnv_data():
    """Create sample CNV data (log2 copy number ratios) for testing."""
    return pd.DataFrame(
        {
            "sample1": [0.5, -0.8, 0.2, 1.2],
            "sample2": [-0.3, 0.9, -0.5, 0.7],
            "sample3": [1.1, -0.2, 0.8, -0.9],
            "sample4": [-0.6, 0.4, -0.1, 0.3],
        },
        index=["TP53", "KRAS", "EGFR", "PIK3CA"],
    )


@pytest.fixture
def sample_cnv_table(sample_cnv_data):
    """Create a sample CNV PivotTable for testing."""
    cnv_table = PivotTable(sample_cnv_data)

    cnv_table.sample_metadata["subtype"] = ["LUAD", "LUSC", "ASC", "LUAD"]
    cnv_table.sample_metadata["age"] = [65, 70, 55, 60]
    cnv_table.sample_metadata["sex"] = ["M", "F", "M", "F"]

    cnv_table.feature_metadata["chromosome"] = [
        "17p13.1",
        "12p12.1",
        "7p11.2",
        "3q26.32",
    ]
    cnv_table.feature_metadata["gene_type"] = [
        "tumor_suppressor",
        "oncogene",
        "oncogene",
        "oncogene",
    ]
    cnv_table.feature_metadata["cytoband"] = [
        "17p13.1",
        "12p12.1",
        "7p11.2",
        "3q26.32",
    ]
    return cnv_table


@pytest.fixture
def sample_cohort(sample_pivot_table, sample_cnv_table):
    """Create a sample Cohort with SNV and CNV tables."""
    cohort = Cohort("test_cohort", "Test cohort for unit tests")
    cohort.add_table(sample_pivot_table, "SNV")
    cohort.add_table(sample_cnv_table, "CNV")
    return cohort


@pytest.fixture
def sample_model_metrics():
    """Create sample CV metric data with generic model labels."""
    rng = np.random.default_rng(42)
    rows = []
    for model in ("ModelA", "ModelB", "ModelC"):
        for fold in range(10):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "acc": float(rng.normal(0.75, 0.05)),
                    "f1": float(rng.normal(0.70, 0.06)),
                    "auc": float(rng.normal(0.80, 0.04)),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "sample_mutation_data",
    "sample_pivot_table",
    "sample_cnv_data",
    "sample_cnv_table",
    "sample_cohort",
    "sample_model_metrics",
]
