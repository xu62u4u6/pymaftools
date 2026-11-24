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
        super().__init__(".", mapping, sample_type=sample_type, sample_key=sample_key)
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

    table = builder.read_and_merge([{"filepath": expression_path, "case_id": "case-1"}])

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


def _nan_segment_inputs(tmp_path):
    cytoband_path = tmp_path / "cytoBand.txt"
    cytoband_path.write_text(
        "chr1\t0\t100\tp1\tgneg\nchrY\t0\t100\tp11.31\tgneg\n", encoding="utf-8"
    )
    segments = pd.DataFrame(
        {
            "Chromosome": ["1", "1", "Y", "1", "Y"],
            "Start": [0, 50, 0, 0, 0],
            "End": [50, 100, 100, 100, 100],
            "Segment_Mean": [2.0, float("nan"), -0.5, 1.0, float("nan")],
            "sample_ID": [
                "male-01A",
                "male-01A",
                "male-01A",
                "female-01A",
                "female-01A",
            ],
            "case_id": ["male", "male", "male", "female", "female"],
            "sample_type": ["Primary Tumor"] * 5,
        }
    )
    return cytoband_path, segments


def test_cnv_segment_nan_segments_do_not_dilute_or_fabricate_diploid(tmp_path):
    # NaN segment length used to count in the denominator: the partly
    # measured band read 1.0 instead of 2.0 and female chrY read 0.0.
    cytoband_path, segments = _nan_segment_inputs(tmp_path)
    builder = TCGACNVSegmentBuilder(
        tmp_path, pd.DataFrame(columns=["filename", "case_id"]), sample_key="sample_id"
    )

    table = builder.build_cytoband_table(segments, cytoband_path)

    assert table.loc["chr1p1", "male-01A"] == pytest.approx(2.0)
    assert table.loc["chrYp11.31", "male-01A"] == pytest.approx(-0.5)
    assert pd.isna(table.loc["chrYp11.31", "female-01A"])
    assert list(table.columns) == ["male-01A", "female-01A"]


def test_legacy_seg_to_cytoband_table_ignores_nan_segments(tmp_path):
    from pymaftools.io.tcga_readers import seg_to_cytoband_table

    cytoband_path, segments = _nan_segment_inputs(tmp_path)

    table = seg_to_cytoband_table(segments, cytoband_path)

    assert table.loc["chr1p1", "male"] == pytest.approx(2.0)
    assert pd.isna(table.loc["chrYp11.31", "female"])
    assert table.value_scale == "log2_ratio"


def test_cnv_gene_builder_tags_absolute_cn_and_binarizes_diploid_two_as_false(tmp_path):
    filename = "sample1.gene_level_copy_number.v36.tsv"
    path = tmp_path / filename
    pd.DataFrame(
        {
            "gene_id": ["ENSG000001.1", "ENSG000002.1", "ENSG000003.1"],
            "gene_name": ["CDKN2A", "PTEN", "TP53"],
            "chromosome": ["9", "10", "17"],
            "start": [1, 1, 1],
            "end": [2, 2, 2],
            "copy_number": [0, 1, 2],
        }
    ).to_csv(path, sep="\t", index=False)
    mapping = pd.DataFrame(
        {
            "filename": [filename],
            "case_id": ["TCGA-01-0001"],
            "sample_id": ["TCGA-01-0001-01A"],
            "sample_type": ["Primary Tumor"],
            "file_id": ["file-cnv-1"],
        }
    )

    table = TCGACNVGeneBuilder(tmp_path, mapping).build()

    assert table.value_scale == "absolute_cn"
    binary = table.to_binary_table()
    assert binary.loc["ENSG000001", "TCGA-01-0001"]
    assert binary.loc["ENSG000002", "TCGA-01-0001"]
    assert not binary.loc["ENSG000003", "TCGA-01-0001"]


def test_expression_builder_build_preserves_star_qc_and_uses_raw_counts_for_tpm(
    tmp_path,
):
    filename = "sample1.rna_seq.augmented_star_gene_counts.tsv"
    path = tmp_path / filename
    pd.DataFrame(
        {
            "gene_id": [
                "N_unmapped",
                "N_multimapping",
                "N_noFeature",
                "N_ambiguous",
                "ENSG000001.1",
                "ENSG000002.1",
            ],
            "gene_name": [None, None, None, None, "TP53", "EGFR"],
            "gene_type": [None, None, None, None, "protein_coding", "protein_coding"],
            "unstranded": [1000, 500, 200, 100, 3000, 2000],
            "tpm_unstranded": [None, None, None, None, 15.5, 42.0],
        }
    ).to_csv(path, sep="\t", index=False)
    mapping = pd.DataFrame(
        {
            "filename": [filename],
            "case_id": ["TCGA-01-0001"],
            "sample_id": ["TCGA-01-0001-01A"],
            "sample_type": ["Primary Tumor"],
        },
        index=pd.Index(["file-expr-1"], name="file_id"),
    )

    counts_table = TCGAExpressionBuilder(
        tmp_path, mapping, count_column="unstranded", enrich_coordinates=False
    ).build()
    tpm_table = TCGAExpressionBuilder(
        tmp_path, mapping, count_column="tpm_unstranded", enrich_coordinates=False
    ).build()

    expected_rate = 5000 / (5000 + 1000 + 500 + 200 + 100)
    for table in (counts_table, tpm_table):
        meta = table.sample_metadata
        assert meta.loc["TCGA-01-0001", "case_id"] == "TCGA-01-0001"
        assert meta.loc["TCGA-01-0001", "file_id"] == "file-expr-1"
        assert meta.loc["TCGA-01-0001", "N_unmapped"] == 1000
        assert meta.loc["TCGA-01-0001", "mapped_reads"] == 5000
        assert meta.loc["TCGA-01-0001", "mapping_rate"] == pytest.approx(expected_rate)

    assert tpm_table.loc["ENSG000001", "TCGA-01-0001"] == pytest.approx(15.5)


def test_expression_builder_does_not_fabricate_mapping_rate_without_qc_or_raw_counts(
    tmp_path,
):
    tpm_only_path = tmp_path / "tpm_only.rna_seq.augmented_star_gene_counts.tsv"
    pd.DataFrame(
        {
            "gene_id": ["N_unmapped", "ENSG000001.1"],
            "gene_name": [None, "TP53"],
            "gene_type": [None, "protein_coding"],
            "tpm_unstranded": [None, 12.5],
        }
    ).to_csv(tpm_only_path, sep="\t", index=False)

    no_qc_path = tmp_path / "no_qc.rna_seq.augmented_star_gene_counts.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG000001.1"],
            "gene_name": ["TP53"],
            "gene_type": ["protein_coding"],
            "unstranded": [500],
        }
    ).to_csv(no_qc_path, sep="\t", index=False)

    builder_tpm = TCGAExpressionBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        count_column="tpm_unstranded",
        enrich_coordinates=False,
    )
    tpm_table = builder_tpm.read_and_merge(
        [{"filepath": tpm_only_path, "case_id": "case-1"}]
    )
    assert pd.isna(tpm_table.sample_metadata.loc["case-1", "mapping_rate"])
    assert pd.isna(tpm_table.sample_metadata.loc["case-1", "mapped_reads"])

    builder_raw = TCGAExpressionBuilder(
        tmp_path,
        pd.DataFrame(columns=["filename", "case_id"]),
        count_column="unstranded",
        enrich_coordinates=False,
    )
    raw_table = builder_raw.read_and_merge(
        [{"filepath": no_qc_path, "case_id": "case-2"}]
    )
    assert raw_table.sample_metadata.loc["case-2", "mapped_reads"] == 500
    assert pd.isna(raw_table.sample_metadata.loc["case-2", "mapping_rate"])


def test_legacy_read_star_counts_tpm_reads_qc_from_raw_counts(tmp_path, monkeypatch):
    from pymaftools.io import tcga_readers

    path = tmp_path / "star.augmented_star_gene_counts.tsv"
    pd.DataFrame(
        {
            "gene_id": ["N_unmapped", "ENSG000001.1"],
            "gene_name": [None, "TP53"],
            "gene_type": [None, "protein_coding"],
            "unstranded": [250, 750],
            "tpm_unstranded": [None, 33.3],
        }
    ).to_csv(path, sep="\t", index=False)

    monkeypatch.setattr(
        tcga_readers,
        "resolve_files_to_cases",
        lambda *_args, **_kwargs: [("TCGA-01-0001", path)],
    )
    table = tcga_readers.read_star_counts(
        tmp_path, tmp_path / "manifest.tsv", value_column="tpm_unstranded"
    )

    assert table.loc["ENSG000001", "TCGA-01-0001"] == pytest.approx(33.3)
    assert table.sample_metadata.loc["TCGA-01-0001", "N_unmapped"] == 250
