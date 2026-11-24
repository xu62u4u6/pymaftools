"""Signature exposure orientation and plotting regressions."""

import pandas as pd
import pytest

from pymaftools.core.SignatureTable import SignatureTable


def test_read_signature_defaults_to_samples_by_signatures_sigprofiler_layout(tmp_path):
    path = tmp_path / "Assignment_Solution_Activities.txt"
    pd.DataFrame(
        {"SBS1": [0.7, 0.2], "SBS5": [0.3, 0.8]},
        index=["S1", "S2"],
    ).to_csv(path, sep="\t")

    table = SignatureTable.read_signature(path)

    assert list(table.index) == ["SBS1", "SBS5"]
    assert list(table.columns) == ["S1", "S2"]
    assert table.loc["SBS5", "S2"] == pytest.approx(0.8)


def test_read_signature_accepts_explicit_signatures_by_samples_orientation(tmp_path):
    path = tmp_path / "exposures.tsv"
    pd.DataFrame(
        {"S1": [0.7, 0.3], "S2": [0.2, 0.8]},
        index=["SBS1", "SBS5"],
    ).to_csv(path, sep="\t")

    table = SignatureTable.read_signature(path, orientation="signatures_by_samples")

    assert list(table.index) == ["SBS1", "SBS5"]
    assert list(table.columns) == ["S1", "S2"]
    assert table.loc["SBS5", "S2"] == pytest.approx(0.8)


def test_read_signature_rejects_ambiguous_orientation_name(tmp_path):
    path = tmp_path / "exposures.tsv"
    pd.DataFrame({"SBS1": [1.0]}, index=["S1"]).to_csv(path, sep="\t")

    with pytest.raises(ValueError, match="signatures_by_samples"):
        SignatureTable.read_signature(path, orientation="auto")
