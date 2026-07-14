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
