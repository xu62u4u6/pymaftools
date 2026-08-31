"""Tests for the TCGA table builders and sample identity contract."""

from pathlib import Path

import pandas as pd
import pytest
import pymaftools

from pymaftools.core.PivotTable import PivotTable
from pymaftools.io.tcga.base import TCGATableBuilder
from pymaftools.io.tcga.cnv_gene import TCGACNVGeneBuilder
from pymaftools.io.tcga.cnv_segment import TCGACNVSegmentBuilder
from pymaftools.io.tcga.expression import TCGAExpressionBuilder
from pymaftools.io.tcga.methylation import TCGAMethylationBuilder
from pymaftools.io.tcga.mutation import TCGAMutationBuilder


def test_tcga_builders_are_available_from_public_namespaces():
    from pymaftools import io

    assert pymaftools.TCGAExpressionBuilder is TCGAExpressionBuilder
    assert pymaftools.TCGAMutationBuilder is TCGAMutationBuilder
    assert io.TCGAExpressionBuilder is TCGAExpressionBuilder
    assert io.TCGAMutationBuilder is TCGAMutationBuilder


class DummyBuilder(TCGATableBuilder):
    file_pattern = "*.dummy"

    def __init__(self, files, sample_type="Primary Tumor", sample_key="case_id"):
        mapping = pd.DataFrame(columns=["filename", "case_id"])
        super().__init__(
            ".", mapping, sample_type=sample_type, sample_key=sample_key
        )
        self.files = files

    def resolve_files(self):
        return self.files

    def read_and_merge(self, files):
        return PivotTable(
            {
                self.sample_identifier(file_info): [file_info["file_id"]]
                for file_info in files
            },
            index=["source_file"],
        )


def _file(case_id, sample_type, file_id, sample_id=None):
    return {
        "case_id": case_id,
        "sample_id": sample_id or f"{case_id}-01A",
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


def test_builder_can_preserve_exact_specimen_ids():
    files = [
        _file("case-1", "Primary Tumor", "tumor-1", "case-1-01A"),
        _file("case-2", "Primary Tumor", "tumor-2", "case-2-01A"),
    ]

    table = DummyBuilder(files, sample_key="sample_id").build()

    assert list(table.columns) == ["case-1-01A", "case-2-01A"]
    assert list(table.sample_metadata.index) == ["case-1-01A", "case-2-01A"]
    assert table.sample_metadata.loc["case-1-01A", "case_id"] == "case-1"


def test_builder_requires_specimen_id_when_exact_identity_is_requested():
    files = [_file("case-1", "Primary Tumor", "tumor-1", sample_id=None)]
    files[0]["sample_id"] = None

    with pytest.raises(ValueError, match="missing 'sample_id'"):
        DummyBuilder(files, sample_key="sample_id").build()


def test_builder_rejects_duplicate_exact_specimen_ids():
    files = [
        _file("case-1", "Primary Tumor", "tumor-1", "shared-01A"),
        _file("case-2", "Primary Tumor", "tumor-2", "shared-01A"),
    ]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        DummyBuilder(files, sample_key="sample_id").build()


def test_builder_rejects_duplicate_files_instead_of_choosing_by_uuid():
    files = [
        _file("case-1", "Primary Tumor", "tumor-z"),
        _file("case-1", "Primary Tumor", "tumor-a"),
    ]

    with pytest.raises(ValueError, match="resolve duplicate files explicitly"):
        DummyBuilder(files).build()


def test_builder_rejects_multiple_primary_tumor_specimens():
    files = [
        _file("case-1", "Primary Tumor", "tumor-a", "case-1-01A"),
        _file("case-1", "Primary Tumor", "tumor-b", "case-1-01B"),
    ]

    with pytest.raises(ValueError, match="multiple specimens"):
        DummyBuilder(files).build()


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


def test_expression_builder_uses_exact_specimen_id_when_requested(tmp_path):
    expression_path = tmp_path / "sample.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG000001.1"],
            "gene_name": ["TP53"],
            "gene_type": ["protein_coding"],
            "unstranded": [95],
        }
    ).to_csv(expression_path, sep="\t", index=False)
    builder = TCGAExpressionBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        enrich_coordinates=False,
        sample_key="sample_id",
    )

    table = builder.read_and_merge(
        [
            {
                "filepath": expression_path,
                "case_id": "case-1",
                "sample_id": "case-1-01A",
            }
        ]
    )

    assert list(table.columns) == ["case-1-01A"]
    assert list(table.sample_metadata.index) == ["case-1-01A"]


def test_cnv_gene_builder_uses_exact_specimen_id_when_requested(tmp_path):
    path = tmp_path / "sample.gene_level_copy_number.v36.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG000001.1"],
            "gene_name": ["TP53"],
            "chromosome": ["17"],
            "start": [1],
            "end": [2],
            "copy_number": [2.0],
        }
    ).to_csv(path, sep="\t", index=False)
    builder = TCGACNVGeneBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        sample_key="sample_id",
    )

    table = builder.read_and_merge(
        [
            {
                "filepath": path,
                "case_id": "case-1",
                "sample_id": "case-1-01A",
            }
        ]
    )

    assert list(table.columns) == ["case-1-01A"]


def test_methylation_builder_uses_exact_specimen_id_when_requested(tmp_path):
    path = tmp_path / "sample.methylation_array.sesame.level3betas.txt"
    path.write_text("cg00000029\t0.25\n", encoding="utf-8")
    builder = TCGAMethylationBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        sample_key="sample_id",
    )

    table = builder.read_and_merge(
        [
            {
                "filepath": path,
                "case_id": "case-1",
                "sample_id": "case-1-01A",
            }
        ]
    )

    assert list(table.columns) == ["case-1-01A"]
    assert table.loc["cg00000029", "case-1-01A"] == pytest.approx(0.25)


def test_methylation_builder_aligns_probe_order_mismatch(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("cg00000029\t0.25\ncg00000108\t0.75\n", encoding="utf-8")
    second.write_text("cg00000108\t0.80\ncg00000029\t0.20\n", encoding="utf-8")
    builder = TCGAMethylationBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
    )

    table = builder.read_and_merge(
        [
            {"filepath": first, "case_id": "case-1"},
            {"filepath": second, "case_id": "case-2"},
        ]
    )

    assert list(table.index) == ["cg00000029", "cg00000108"]
    assert table.loc["cg00000029", "case-2"] == pytest.approx(0.20)
    assert table.loc["cg00000108", "case-2"] == pytest.approx(0.80)


def test_mutation_builder_writes_exact_specimen_id(tmp_path):
    path = tmp_path / "sample.maf.gz"
    pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53"],
            "Chromosome": ["17"],
            "Start_Position": [1],
            "End_Position": [1],
            "Reference_Allele": ["C"],
            "Tumor_Seq_Allele1": ["C"],
            "Tumor_Seq_Allele2": ["T"],
            "Variant_Classification": ["Missense_Mutation"],
            "Variant_Type": ["SNP"],
        }
    ).to_csv(path, sep="\t", index=False, compression="gzip")
    builder = TCGAMutationBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        sample_key="sample_id",
    )

    maf = builder.read_and_merge(
        [
            {
                "filepath": path,
                "case_id": "case-1",
                "sample_id": "case-1-01A",
                "sample_type": "Primary Tumor",
            }
        ]
    )

    assert maf["sample_ID"].tolist() == ["case-1-01A"]


def test_cnv_segment_cytoband_overlap_is_sample_keyed_and_weighted(tmp_path):
    cytoband_path = tmp_path / "cytoBand.txt"
    cytoband_path.write_text(
        "chr1\t0\t100\tp1\tgneg\nchr1\t100\t200\tq1\tgpos25\n",
        encoding="utf-8",
    )
    builder = TCGACNVSegmentBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        sample_key="sample_id",
    )
    segments = pd.DataFrame(
        {
            "Chromosome": ["1", "1"],
            "Start": [0, 50],
            "End": [100, 150],
            "Segment_Mean": [0.0, 1.0],
            "sample_ID": ["case-1-01A", "case-1-01A"],
            "case_id": ["case-1", "case-1"],
            "sample_type": ["Primary Tumor", "Primary Tumor"],
        }
    )

    table = builder.build_cytoband_table(segments, cytoband_path)

    assert list(table.columns) == ["case-1-01A"]
    assert table.loc["chr1p1", "case-1-01A"] == pytest.approx(1 / 3)
    assert table.loc["chr1q1", "case-1-01A"] == pytest.approx(1.0)


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
