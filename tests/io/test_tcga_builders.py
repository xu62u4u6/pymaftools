"""Tests for the case-level TCGA table builders."""

from pathlib import Path

import pandas as pd
import pytest
import pymaftools

from pymaftools.core.PivotTable import PivotTable
from pymaftools.io.tcga.base import TCGATableBuilder
from pymaftools.io.tcga.expression import TCGAExpressionBuilder
from pymaftools.io.tcga.mutation import TCGAMutationBuilder


def test_tcga_builders_are_available_from_public_namespaces():
    from pymaftools import io

    assert pymaftools.TCGAExpressionBuilder is TCGAExpressionBuilder
    assert pymaftools.TCGAMutationBuilder is TCGAMutationBuilder
    assert io.TCGAExpressionBuilder is TCGAExpressionBuilder
    assert io.TCGAMutationBuilder is TCGAMutationBuilder


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


def test_expression_builder_handles_missing_qc_rows(tmp_path):
    expression_path = tmp_path / "sample.tsv"
    pd.DataFrame(
        {
            "gene_id": ["N_unmapped", "ENSG000001.1"],
            "gene_name": [None, "TP53"],
            "gene_type": [None, "protein_coding"],
            "unstranded": [5, 95],
        }
    ).to_csv(expression_path, sep="\t", index=False)
    builder = TCGAExpressionBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        enrich_coordinates=False,
    )

    table = builder.read_and_merge(
        [{"filepath": expression_path, "case_id": "case-1"}]
    )

    assert table.sample_metadata.loc["case-1", "mapping_rate"] == pytest.approx(0.95)
    assert pd.isna(table.sample_metadata.loc["case-1", "N_multimapping"])


def test_mutation_builder_reports_empty_tumor_selection(tmp_path):
    maf_path = tmp_path / "normal.maf.gz"
    pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53"],
            "Chromosome": ["17"],
            "Start_Position": [1],
            "Tumor_Seq_Allele2": ["A"],
        }
    ).to_csv(maf_path, sep="\t", index=False, compression="gzip")
    builder = TCGAMutationBuilder(
        tmp_path, pd.DataFrame(columns=["filename", "case_id"])
    )

    with pytest.raises(ValueError, match="No mutation files remained"):
        builder.read_and_merge(
            [
                {
                    "filepath": maf_path,
                    "case_id": "case-1",
                    "sample_type": "Solid Tissue Normal",
                }
            ]
        )
