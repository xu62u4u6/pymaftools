"""Tests for the case-level TCGA table builders."""

from pathlib import Path

import pandas as pd
import pytest

from pymaftools.core.PivotTable import PivotTable
from pymaftools.io.tcga.base import TCGATableBuilder


class DummyBuilder(TCGATableBuilder):
    file_pattern = "*.dummy"

    def __init__(self, files, sample_type="Primary Tumor"):
        mapping = pd.DataFrame(columns=["filename", "case_id"])
        super().__init__(".", mapping, sample_type=sample_type)
        self.files = files

    def resolve_files(self):
        return self.files

    def read_and_merge(self, files):
        return PivotTable(
            {file_info["case_id"]: [file_info["file_id"]] for file_info in files},
            index=["source_file"],
        )


def _file(case_id, sample_type, file_id):
    return {
        "case_id": case_id,
        "sample_type": sample_type,
        "data_type": "dummy",
        "file_id": file_id,
        "filepath": Path(f"/{file_id}.dummy"),
    }


def test_builder_filters_before_building_values_and_metadata():
    files = [
        _file("case-1", "Solid Tissue Normal", "normal-1"),
        _file("case-1", "Primary Tumor", "tumor-1"),
        _file("case-2", "Primary Tumor", "tumor-2"),
    ]

    table = DummyBuilder(files).build()

    assert table.loc["source_file"].to_dict() == {
        "case-1": "tumor-1",
        "case-2": "tumor-2",
    }
    assert table.sample_metadata["file_id"].to_dict() == {
        "case-1": "tumor-1",
        "case-2": "tumor-2",
    }
    assert set(table.sample_metadata["sample_type"]) == {"Primary Tumor"}


def test_builder_selects_duplicate_files_deterministically():
    files = [
        _file("case-1", "Primary Tumor", "tumor-z"),
        _file("case-1", "Primary Tumor", "tumor-a"),
    ]

    with pytest.warns(UserWarning, match="has 2 matching files"):
        table = DummyBuilder(files).build()

    assert table.loc["source_file", "case-1"] == "tumor-a"
    assert table.sample_metadata.loc["case-1", "file_id"] == "tumor-a"


def test_builder_requires_explicit_type_for_mixed_case_samples():
    files = [
        _file("case-1", "Primary Tumor", "tumor-1"),
        _file("case-1", "Solid Tissue Normal", "normal-1"),
    ]

    with pytest.raises(ValueError, match="set sample_type explicitly"):
        DummyBuilder(files, sample_type=None).build()
