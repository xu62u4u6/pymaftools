"""
Tests for TCGA data readers.

Unit tests use mocked data to avoid network/file dependencies.
Integration tests use the 10-case sample dataset at data/sample/.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pymaftools.core.ExpressionTable import ExpressionTable
from pymaftools.core.MAF import MAF
from pymaftools.core.PivotTable import PivotTable
from pymaftools.io.tcga_readers import (
    build_uuid_to_case_mapping,
    read_clinical,
    read_maf_files,
    read_manifest,
    read_methylation_betas,
    read_seg_files,
    read_star_counts,
    scan_gdc_directory,
)


@pytest.fixture
def mock_manifest(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        "uuid-1\tfile1.tsv\tabc\t1000\treleased\n"
        "uuid-2\tfile2.tsv\tdef\t2000\treleased\n"
    )
    return manifest


@pytest.fixture
def mock_star_dir(tmp_path):
    gene_lines = [
        "# gene-model: GENCODE v36",
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded",
        "N_unmapped\t\t\t100\t100\t100\t\t\t",
        "N_multimapping\t\t\t200\t200\t200\t\t\t",
        "N_noFeature\t\t\t300\t300\t300\t\t\t",
        "N_ambiguous\t\t\t400\t400\t400\t\t\t",
    ]
    genes = [
        ("ENSG00000000003.15", "TSPAN6", "protein_coding"),
        ("ENSG00000000005.6", "TNMD", "protein_coding"),
        ("ENSG00000000419.13", "DPM1", "protein_coding"),
    ]

    for i, uuid in enumerate(["uuid-aaa", "uuid-bbb", "uuid-ccc"]):
        directory = tmp_path / uuid
        directory.mkdir()
        lines = gene_lines.copy()
        for gid, gname, gtype in genes:
            count = (i + 1) * 100
            lines.append(f"{gid}\t{gname}\t{gtype}\t{count}\t50\t50\t1.0\t0.5\t0.4")
        (directory / f"{uuid}.rna_seq.augmented_star_gene_counts.tsv").write_text(
            "\n".join(lines)
        )

    return tmp_path


@pytest.fixture
def mock_manifest_for_star(tmp_path):
    manifest = tmp_path / "manifest_expr.tsv"
    manifest.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        "uuid-aaa\tfile_a.tsv\tabc\t1000\treleased\n"
        "uuid-bbb\tfile_b.tsv\tdef\t1000\treleased\n"
        "uuid-ccc\tfile_c.tsv\tghi\t1000\treleased\n"
    )
    return manifest


@pytest.fixture
def mock_seg_dir(tmp_path):
    for uuid in ["uuid-seg-1", "uuid-seg-2", "uuid-seg-3"]:
        directory = tmp_path / uuid
        directory.mkdir()
        rows = ["GDC_Aliquot\tChromosome\tStart\tEnd\tNum_Probes\tSegment_Mean"]
        for i in range(5):
            rows.append(
                f"aliquot-{uuid}\t1\t{i * 100 + 1}\t{i * 100 + 50}\t{i + 10}\t{0.1 * i:.3f}"
            )
        (directory / f"{uuid}.nocnv_grch38.seg.v2.txt").write_text("\n".join(rows))
    return tmp_path


@pytest.fixture
def mock_manifest_for_seg(tmp_path):
    manifest = tmp_path / "manifest_seg.tsv"
    manifest.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        "uuid-seg-1\tseg1.txt\ta\t1\treleased\n"
        "uuid-seg-2\tseg2.txt\tb\t1\treleased\n"
        "uuid-seg-3\tseg3.txt\tc\t1\treleased\n"
    )
    return manifest


@pytest.fixture
def mock_maf_dir(tmp_path):
    for uuid, sample in [("uuid-maf-1", "TCGA-01-0001"), ("uuid-maf-2", "TCGA-01-0002")]:
        directory = tmp_path / uuid
        directory.mkdir()
        content = "\n".join(
            [
                "#version gdc-1.0.0",
                "#annotation.spec gdc-2.0.0-aliquot-merged-masked",
                "Hugo_Symbol\tStart_Position\tEnd_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tVariant_Classification\tTumor_Sample_Barcode\tcase_id",
                f"TP53\t100\t100\tA\tA\tT\tMissense_Mutation\t{sample}-01A\t{sample}",
                f"KRAS\t200\t200\tG\tG\tA\tSilent\t{sample}-01A\t{sample}",
                f"EGFR\t300\t300\tC\tC\tG\tNonsense_Mutation\t{sample}-01A\t{sample}",
            ]
        )
        with gzip.open(directory / f"{uuid}.wxs.aliquot_ensemble_masked.maf.gz", "wt") as handle:
            handle.write(content)
    return tmp_path


@pytest.fixture
def mock_manifest_for_maf(tmp_path):
    manifest = tmp_path / "manifest_maf.tsv"
    manifest.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        "uuid-maf-1\tmaf1.maf.gz\ta\t1\treleased\n"
        "uuid-maf-2\tmaf2.maf.gz\tb\t1\treleased\n"
    )
    return manifest


@pytest.fixture
def mock_methylation_dir(tmp_path):
    probes = ["cg00000029", "cg00000108", "cg00000109", "cg00000165", "cg00000236"]
    values = {
        "uuid-met-1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "uuid-met-2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "uuid-met-3": [0.9, 0.8, 0.7, 0.6, 0.5],
    }
    for uuid, beta_values in values.items():
        directory = tmp_path / uuid
        directory.mkdir()
        rows = [f"{probe}\t{beta}" for probe, beta in zip(probes, beta_values)]
        (directory / f"{uuid}.methylation_array.sesame.level3betas.txt").write_text(
            "\n".join(rows)
        )
    return tmp_path


@pytest.fixture
def mock_manifest_for_methylation(tmp_path):
    manifest = tmp_path / "manifest_methylation.tsv"
    manifest.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        "uuid-met-1\tmet1.txt\ta\t1\treleased\n"
        "uuid-met-2\tmet2.txt\tb\t1\treleased\n"
        "uuid-met-3\tmet3.txt\tc\t1\treleased\n"
    )
    return manifest


@pytest.fixture
def mock_clinical_dir(tmp_path):
    header = "bcr_patient_barcode\tgender\tvital_status"
    aliases = "bcr_patient_barcode\tgender\tvital_status"
    cde = "CDE_ID:2003301\tCDE_ID:2200604\tCDE_ID:3224275"

    first = tmp_path / "uuid-clin-1"
    first.mkdir()
    (first / "nationwidechildrens.org_clinical_patient_luad.txt").write_text(
        "\n".join(
            [
                header,
                aliases,
                cde,
                "TCGA-01-0001\tMALE\tAlive",
                "TCGA-01-0002\tFEMALE\tDead",
            ]
        )
    )

    second = tmp_path / "uuid-clin-2"
    second.mkdir()
    (second / "nationwidechildrens.org_clinical_patient_luad.txt").write_text(
        "\n".join(
            [
                header,
                aliases,
                cde,
                "TCGA-01-0002\tFEMALE\tDead",
                "TCGA-01-0003\tMALE\tAlive",
            ]
        )
    )
    return tmp_path


class TestReadManifest:
    def test_basic(self, mock_manifest):
        df = read_manifest(mock_manifest)
        assert len(df) == 2
        assert df.index.name == "file_id"
        assert "uuid-1" in df.index

    def test_columns(self, mock_manifest):
        df = read_manifest(mock_manifest)
        assert "filename" in df.columns


class TestScanGdcDirectory:
    def test_finds_files(self, mock_star_dir):
        result = scan_gdc_directory(mock_star_dir, "*.augmented_star_gene_counts.tsv")
        assert len(result) == 3
        assert "uuid-aaa" in result
        assert result["uuid-aaa"].exists()

    def test_empty_dir(self, tmp_path):
        result = scan_gdc_directory(tmp_path, "*.tsv")
        assert len(result) == 0

    def test_skips_logs(self, tmp_path):
        logs_dir = tmp_path / "uuid-1" / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "file.tsv").write_text("test")
        result = scan_gdc_directory(tmp_path, "*.tsv")
        assert "logs" not in result


class TestBuildUuidToCaseMapping:
    @patch("pymaftools.io.tcga_readers.requests.post")
    def test_mapping(self, mock_post, mock_manifest):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "hits": [
                    {"file_id": "uuid-1", "cases": [{"submitter_id": "TCGA-AA-0001"}]},
                    {"file_id": "uuid-2", "cases": [{"submitter_id": "TCGA-AA-0002"}]},
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        mapping = build_uuid_to_case_mapping(mock_manifest)
        assert mapping == {"uuid-1": "TCGA-AA-0001", "uuid-2": "TCGA-AA-0002"}


class TestReadStarCounts:
    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_basic(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0002",
            "uuid-ccc": "TCGA-01-0003",
        }

        table = read_star_counts(mock_star_dir, mock_manifest_for_star)

        assert isinstance(table, ExpressionTable)
        assert table.shape == (3, 3)
        assert "TCGA-01-0001" in table.columns
        assert "ENSG00000000003" in table.index

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_qc_rows_removed(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0002",
            "uuid-ccc": "TCGA-01-0003",
        }

        table = read_star_counts(mock_star_dir, mock_manifest_for_star)
        assert "N_unmapped" not in table.index
        assert "N_multimapping" not in table.index

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_feature_metadata(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0002",
            "uuid-ccc": "TCGA-01-0003",
        }

        table = read_star_counts(mock_star_dir, mock_manifest_for_star)
        assert "gene_name" in table.feature_metadata.columns
        assert "gene_type" in table.feature_metadata.columns
        assert table.feature_metadata.loc["ENSG00000000003", "gene_name"] == "TSPAN6"

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_sample_metadata(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0002",
            "uuid-ccc": "TCGA-01-0003",
        }

        table = read_star_counts(mock_star_dir, mock_manifest_for_star)
        assert "case_id" in table.sample_metadata.columns
        assert len(table.sample_metadata) == 3

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_value_column_tpm(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0002",
            "uuid-ccc": "TCGA-01-0003",
        }

        table = read_star_counts(
            mock_star_dir, mock_manifest_for_star, value_column="tpm_unstranded"
        )
        assert (table.values == 1.0).all()

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_no_version_strip(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0002",
            "uuid-ccc": "TCGA-01-0003",
        }

        table = read_star_counts(
            mock_star_dir, mock_manifest_for_star, strip_gene_version=False
        )
        assert "ENSG00000000003.15" in table.index

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_deduplication(self, mock_mapping, mock_star_dir, mock_manifest_for_star):
        mock_mapping.return_value = {
            "uuid-aaa": "TCGA-01-0001",
            "uuid-bbb": "TCGA-01-0001",
            "uuid-ccc": "TCGA-01-0002",
        }

        table = read_star_counts(mock_star_dir, mock_manifest_for_star)
        assert table.shape[1] == 2

    def test_empty_dir_raises(self, tmp_path, mock_manifest_for_star):
        with pytest.raises(FileNotFoundError, match="No STAR count files found"):
            read_star_counts(tmp_path, mock_manifest_for_star)


class TestReadSegFiles:
    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_basic(self, mock_mapping, mock_seg_dir, mock_manifest_for_seg):
        mock_mapping.return_value = {
            "uuid-seg-1": "TCGA-01-0001",
            "uuid-seg-2": "TCGA-01-0002",
            "uuid-seg-3": "TCGA-01-0003",
        }

        df = read_seg_files(mock_seg_dir, mock_manifest_for_seg)
        assert len(df) == 15
        assert df["case_id"].nunique() == 3
        assert list(df.columns) == [
            "case_id",
            "Chromosome",
            "Start",
            "End",
            "Num_Probes",
            "Segment_Mean",
        ]
        assert "GDC_Aliquot" not in df.columns

    def test_empty_dir_raises(self, tmp_path, mock_manifest_for_seg):
        with pytest.raises(FileNotFoundError, match="No SEG files found"):
            read_seg_files(tmp_path, mock_manifest_for_seg)


class TestReadMafFiles:
    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_basic(self, mock_mapping, mock_maf_dir, mock_manifest_for_maf):
        mock_mapping.return_value = {
            "uuid-maf-1": "TCGA-01-0001",
            "uuid-maf-2": "TCGA-01-0002",
        }

        result = read_maf_files(mock_maf_dir, mock_manifest_for_maf)
        assert isinstance(result, MAF)
        assert len(result) == 6
        assert set(result["sample_ID"]) == {"TCGA-01-0001", "TCGA-01-0002"}

    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_nonsynonymous_filter(self, mock_mapping, mock_maf_dir, mock_manifest_for_maf):
        mock_mapping.return_value = {
            "uuid-maf-1": "TCGA-01-0001",
            "uuid-maf-2": "TCGA-01-0002",
        }

        result = read_maf_files(
            mock_maf_dir, mock_manifest_for_maf, nonsynonymous_only=True
        )
        assert len(result) == 4
        assert set(result["Variant_Classification"]) == {
            "Missense_Mutation",
            "Nonsense_Mutation",
        }

    def test_empty_dir_raises(self, tmp_path, mock_manifest_for_maf):
        with pytest.raises(FileNotFoundError, match="No MAF files found"):
            read_maf_files(tmp_path, mock_manifest_for_maf)


class TestReadMethylationBetas:
    @patch("pymaftools.io.tcga_readers.build_uuid_to_case_mapping")
    def test_basic(
        self, mock_mapping, mock_methylation_dir, mock_manifest_for_methylation
    ):
        mock_mapping.return_value = {
            "uuid-met-1": "TCGA-01-0001",
            "uuid-met-2": "TCGA-01-0002",
            "uuid-met-3": "TCGA-01-0003",
        }

        table = read_methylation_betas(
            mock_methylation_dir, mock_manifest_for_methylation
        )
        assert isinstance(table, PivotTable)
        assert table.shape == (5, 3)
        assert "cg00000029" in table.index
        assert "TCGA-01-0001" in table.columns
        assert table.loc["cg00000029", "TCGA-01-0001"] == pytest.approx(0.1)

    def test_empty_dir_raises(self, tmp_path, mock_manifest_for_methylation):
        with pytest.raises(FileNotFoundError, match="No methylation beta files found"):
            read_methylation_betas(tmp_path, mock_manifest_for_methylation)


class TestReadClinical:
    def test_basic(self, mock_clinical_dir):
        df = read_clinical(mock_clinical_dir)
        assert len(df) == 3
        assert df.index.name == "bcr_patient_barcode"
        assert "bcr_patient_barcode" in df.columns
        assert df.loc["TCGA-01-0001", "gender"] == "MALE"

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No clinical patient files found"):
            read_clinical(tmp_path)


SAMPLE_DATA = Path(__file__).resolve().parents[2] / "data" / "sample"


@pytest.mark.integration
class TestReadStarCountsIntegration:
    @pytest.fixture(autouse=True)
    def check_sample_data(self):
        if not (SAMPLE_DATA / "expression").exists():
            pytest.skip("Sample data not available at data/sample/expression")

    def test_load_10_cases(self):
        table = read_star_counts(
            SAMPLE_DATA / "expression",
            SAMPLE_DATA / "manifests" / "manifest_expression.tsv",
        )
        assert isinstance(table, ExpressionTable)
        assert table.shape[0] > 60000
        assert table.shape[1] == 10
        assert table.isna().sum().sum() == 0

    def test_load_tpm(self):
        table = read_star_counts(
            SAMPLE_DATA / "expression",
            SAMPLE_DATA / "manifests" / "manifest_expression.tsv",
            value_column="tpm_unstranded",
        )
        assert table.shape[1] == 10
        assert (table.values >= 0).all()


@pytest.mark.integration
class TestReadSegFilesIntegration:
    @pytest.fixture(autouse=True)
    def check_sample_data(self):
        if not (SAMPLE_DATA / "cnv").exists():
            pytest.skip("Sample data not available at data/sample/cnv")

    def test_load_segments(self):
        df = read_seg_files(
            SAMPLE_DATA / "cnv",
            SAMPLE_DATA / "manifests" / "manifest_cnv.tsv",
        )
        assert df["case_id"].nunique() == 10
        assert {
            "case_id",
            "Chromosome",
            "Start",
            "End",
            "Num_Probes",
            "Segment_Mean",
        } <= set(df.columns)
        assert pd.api.types.is_float_dtype(df["Segment_Mean"])


@pytest.mark.integration
class TestReadMafFilesIntegration:
    @pytest.fixture(autouse=True)
    def check_sample_data(self):
        if not (SAMPLE_DATA / "mutation").exists():
            pytest.skip("Sample data not available at data/sample/mutation")

    def test_load_maf(self):
        result = read_maf_files(
            SAMPLE_DATA / "mutation",
            SAMPLE_DATA / "manifests" / "manifest_mutation.tsv",
        )
        assert isinstance(result, MAF)
        assert len(result) > 0
        assert {"Hugo_Symbol", "Variant_Classification"} <= set(result.columns)


@pytest.mark.integration
class TestReadMethylationBetasIntegration:
    @pytest.fixture(autouse=True)
    def check_sample_data(self):
        if not (SAMPLE_DATA / "methylation").exists():
            pytest.skip("Sample data not available at data/sample/methylation")

    def test_load_methylation(self):
        table = read_methylation_betas(
            SAMPLE_DATA / "methylation",
            SAMPLE_DATA / "manifests" / "manifest_methylation.tsv",
        )
        assert isinstance(table, PivotTable)
        assert table.shape[0] > 400000
        assert table.shape[1] == 10
        non_na = table.stack()
        assert len(non_na) > 0
        assert ((non_na >= 0) & (non_na <= 1)).all()


@pytest.mark.integration
class TestReadClinicalIntegration:
    @pytest.fixture(autouse=True)
    def check_sample_data(self):
        if not (SAMPLE_DATA / "clinical").exists():
            pytest.skip("Sample data not available at data/sample/clinical")

    def test_load_patient_table(self):
        df = read_clinical(SAMPLE_DATA / "clinical")
        assert len(df) > 500
        assert "bcr_patient_barcode" in df.columns
