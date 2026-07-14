"""Tests for expression-specific table behavior."""

import builtins

import pandas as pd
import pytest

from pymaftools.core.ExpressionTable import ExpressionTable


def test_deseq2_missing_dependency_has_install_guidance(monkeypatch):
    table = ExpressionTable(
        pd.DataFrame({"s1": [10, 4], "s2": [12, 3]}, index=["G1", "G2"])
    )
    table.sample_metadata["group"] = ["target", "control"]
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("pydeseq2"):
            raise ImportError("No module named 'pydeseq2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match=r"pymaftools\[expression\]"):
        table.deseq2("group", target="target", control="control")


def test_deseq2_validates_groups_and_raw_counts_before_import():
    table = ExpressionTable(
        pd.DataFrame({"s1": [10.5, 4], "s2": [12, 3]}, index=["G1", "G2"])
    )
    table.sample_metadata["group"] = ["target", "control"]

    with pytest.raises(ValueError, match="integer counts"):
        table.deseq2("group", target="target", control="control")

    table.iloc[0, 0] = 10
    with pytest.raises(ValueError, match="not found"):
        table.deseq2("group", target="missing", control="control")


def test_expression_filters_and_deg_selection_preserve_metadata():
    table = ExpressionTable(
        pd.DataFrame({"s1": [10, 1, 4], "s2": [10, 1, 8]}, index=["G1", "G2", "G3"])
    )
    table.sample_metadata["group"] = ["A", "B"]
    table.feature_metadata["padj"] = [0.01, 0.2, 0.03]
    table.feature_metadata["log2FoldChange"] = [2.0, -3.0, 0.5]

    filtered = table.filter_low_expression(min_total=10)

    assert list(filtered.index) == ["G1", "G3"]
    assert filtered.sample_metadata.equals(table.sample_metadata)
    assert list(table.find_deg().index) == ["G1"]


def test_to_cluster_table_does_not_create_a_fake_sample_column():
    table = ExpressionTable(
        pd.DataFrame({"s1": [2.0, 4.0, 99.0], "s2": [6.0, 8.0, 99.0]}, index=["G1", "G2", "G3"])
    )
    table.sample_metadata["group"] = ["A", "B"]
    table.feature_metadata["cluster"] = ["C1", "C1", pd.NA]

    clustered = table.to_cluster_table()

    assert list(clustered.columns) == ["s1", "s2"]
    assert clustered.loc["C1"].to_dict() == {"s1": 3.0, "s2": 7.0}
    assert clustered.feature_metadata.loc["C1", "features"] == ["G1", "G2"]
    assert clustered.sample_metadata.equals(table.sample_metadata)
