"""
Tests for GDC / TCGA IO module.

Tests are split into:
- Unit tests (offline, mocked): barcode parsing, manifest generation logic
- Integration tests (online, requires network): actual GDC API queries

Integration tests are marked with @pytest.mark.integration and skipped by default.
Run with: pytest -m integration
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import hashlib

from pymaftools.io.tcga import GDCClient, parse_tcga_barcode, DATA_TYPE_CONFIGS
from pymaftools.io.tcga.client import _extract_biospecimen_metadata


# ------------------------------------------------------------------ #
#  Unit tests (offline)
# ------------------------------------------------------------------ #


class TestParseTcgaBarcode:
    """Test TCGA barcode parsing."""

    def test_full_barcode(self):
        result = parse_tcga_barcode("TCGA-44-2655-01A-01D-0182-01")
        assert result["project"] == "TCGA"
        assert result["tss"] == "44"
        assert result["participant"] == "2655"
        assert result["case_id"] == "TCGA-44-2655"
        assert result["sample_type"] == 1
        assert result["is_tumor"] is True
        assert result["vial"] == "A"
        assert result["portion"] == "01"
        assert result["analyte"] == "D"
        assert result["plate"] == "0182"
        assert result["center"] == "01"

    def test_case_level_barcode(self):
        result = parse_tcga_barcode("TCGA-44-2655")
        assert result["case_id"] == "TCGA-44-2655"
        assert "sample_type" not in result

    def test_sample_level_barcode(self):
        result = parse_tcga_barcode("TCGA-44-2655-11A")
        assert result["sample_type"] == 11
        assert result["is_tumor"] is False
        assert result["vial"] == "A"

    def test_tumor_types(self):
        assert parse_tcga_barcode("TCGA-XX-0001-01A")["is_tumor"] is True  # Primary
        assert parse_tcga_barcode("TCGA-XX-0001-06A")["is_tumor"] is True  # Metastatic
        assert parse_tcga_barcode("TCGA-XX-0001-10A")["is_tumor"] is False  # Blood Normal
        assert parse_tcga_barcode("TCGA-XX-0001-11A")["is_tumor"] is False  # Solid Normal


class TestGDCClientOffline:
    """Test GDCClient methods with mocked API calls."""

    def test_init_no_token(self):
        client = GDCClient()
        assert client.token is None

    def test_init_with_token(self, tmp_path):
        token_file = tmp_path / "token.txt"
        token_file.write_text("my-secret-token\n")
        client = GDCClient(token_path=str(token_file))
        assert client.token == "my-secret-token"
        assert client.token_path == token_file

    def test_data_type_configs(self):
        """Verify all expected data types are configured."""
        expected = {"expression", "mutation", "cnv_seg", "cnv_gene", "methylation"}
        assert set(DATA_TYPE_CONFIGS.keys()) == expected
        for key, config in DATA_TYPE_CONFIGS.items():
            assert "data_type" in config
            assert "label" in config

    def test_extracts_file_associated_tumor_aliquot_not_first_case_sample(self):
        hit = {
            "cases": [
                {
                    "submitter_id": "TCGA-44-6147",
                    "project": {"project_id": "TCGA-LUAD"},
                    "samples": [
                        {
                            "submitter_id": "TCGA-44-6147-11A",
                            "sample_id": "normal-sample-uuid",
                            "sample_type": "Solid Tissue Normal",
                        },
                        {
                            "submitter_id": "TCGA-44-6147-01A",
                            "sample_id": "tumor-sample-uuid",
                            "sample_type": "Primary Tumor",
                        },
                    ],
                }
            ],
            "associated_entities": [
                {
                    "entity_type": "aliquot",
                    "entity_submitter_id": "TCGA-44-6147-11A-01D-1111-01",
                    "entity_id": "normal-aliquot-uuid",
                },
                {
                    "entity_type": "aliquot",
                    "entity_submitter_id": "TCGA-44-6147-01A-11R-1755-07",
                    "entity_id": "tumor-aliquot-uuid",
                },
            ],
        }

        result = _extract_biospecimen_metadata(hit)

        assert result["case_id"] == "TCGA-44-6147"
        assert result["sample_id"] == "TCGA-44-6147-01A"
        assert result["sample_uuid"] == "tumor-sample-uuid"
        assert result["sample_type"] == "Primary Tumor"
        assert result["portion_id"] == "TCGA-44-6147-01A-11"
        assert result["analyte_id"] == "TCGA-44-6147-01A-11R"
        assert result["aliquot_id"] == "TCGA-44-6147-01A-11R-1755-07"
        assert result["paired_normal_aliquot_ids"] == (
            "TCGA-44-6147-11A-01D-1111-01"
        )
        assert result["mapping_status"] == "resolved_tumor_aliquot"

    def test_marks_multiple_tumor_aliquots_ambiguous(self):
        hit = {
            "cases": [{"submitter_id": "TCGA-XX-0001", "samples": []}],
            "associated_entities": [
                {
                    "entity_type": "aliquot",
                    "entity_submitter_id": "TCGA-XX-0001-01A-01D-0000-01",
                    "entity_id": "a",
                },
                {
                    "entity_type": "aliquot",
                    "entity_submitter_id": "TCGA-XX-0001-01B-01D-0000-01",
                    "entity_id": "b",
                },
            ],
        }

        result = _extract_biospecimen_metadata(hit)

        assert result["mapping_status"] == "ambiguous_tumor_aliquot"
        assert result["sample_id"] is None
        assert result["tumor_aliquot_count"] == 2

    def test_align_specimens_reports_attrition_without_cross_vial_join(self):
        rows = []

        def add(case, modality, sample, file_id, status="resolved_tumor_aliquot"):
            rows.append(
                {
                    "file_id": file_id,
                    "case_id": case,
                    "project": "TCGA-LUAD",
                    "data_type": modality,
                    "sample_id": sample,
                    "sample_type": "Primary Tumor",
                    "mapping_status": status,
                }
            )

        add("C1", "expression", "C1-01A", "C1-e")
        add("C1", "mutation", "C1-01A", "C1-m")
        add("C2", "expression", "C2-01A", "C2-e")
        add("C2", "mutation", "C2-01B", "C2-m")
        add("C3", "expression", "C3-01A", "C3-e")
        add("C4", "expression", "C4-01A", "C4-e1")
        add("C4", "expression", "C4-01A", "C4-e2")
        add("C4", "mutation", "C4-01A", "C4-m")
        add("C5", "expression", "C5-01A", "C5-e")
        add("C5", "mutation", "C5-01A", "C5-m")
        rows.append(
            {
                "file_id": "C5-normal-expression",
                "case_id": "C5",
                "project": "TCGA-LUAD",
                "data_type": "expression",
                "sample_id": None,
                "sample_type": None,
                "mapping_status": "no_tumor_aliquot",
            }
        )

        selected, report = GDCClient.align_specimens(
            pd.DataFrame(rows), ["expression", "mutation"]
        )

        assert selected["file_id"].tolist() == ["C1-e", "C1-m", "C5-e", "C5-m"]
        assert report.set_index("case_id")["status"].to_dict() == {
            "C1": "selected",
            "C2": "specimen_mismatch",
            "C3": "missing_modality",
            "C4": "duplicate_files",
            "C5": "selected",
        }

    def test_build_file_mapping_emits_exact_provenance_columns(self, monkeypatch):
        hit = {
            "file_id": "file-1",
            "data_type": "Gene Expression Quantification",
            "md5sum": "abc",
            "file_size": 123,
            "state": "released",
            "analysis": {
                "workflow_type": "STAR - Counts",
                "updated_datetime": "2026-01-01T00:00:00Z",
            },
            "cases": [
                {
                    "submitter_id": "TCGA-XX-0001",
                    "project": {"project_id": "TCGA-LUAD"},
                    "samples": [
                        {
                            "submitter_id": "TCGA-XX-0001-01A",
                            "sample_id": "sample-uuid",
                            "sample_type": "Primary Tumor",
                        }
                    ],
                }
            ],
            "associated_entities": [
                {
                    "entity_type": "aliquot",
                    "entity_submitter_id": "TCGA-XX-0001-01A-01R-0000-01",
                    "entity_id": "aliquot-uuid",
                }
            ],
        }
        client = GDCClient()
        monkeypatch.setattr(client, "_batch_query_metadata", lambda _: [hit])

        mapping = client.build_file_mapping(
            [{"file_id": "file-1", "filename": "counts.tsv", "dtype": "expression"}]
        )

        row = mapping.iloc[0]
        assert row["data_type"] == "expression"
        assert row["gdc_data_type"] == "Gene Expression Quantification"
        assert row["workflow_type"] == "STAR - Counts"
        assert row["sample_id"] == "TCGA-XX-0001-01A"
        assert row["aliquot_id"] == "TCGA-XX-0001-01A-01R-0000-01"
        assert row["mapping_status"] == "resolved_tumor_aliquot"

    def test_align_manifests_uses_exact_shared_sample(self, tmp_path):
        data_types = {
            "expression": {"data_type": "expression"},
            "mutation": {"data_type": "mutation"},
        }
        full_dir = tmp_path / "full"
        aligned_dir = tmp_path / "aligned"
        full_dir.mkdir()
        mapping_rows = []
        manifests = {"expression": [], "mutation": []}

        def add(case, modality, sample, file_id):
            mapping_rows.append(
                {
                    "file_id": file_id,
                    "filename": f"{file_id}.txt",
                    "data_type": modality,
                    "case_id": case,
                    "project": "TCGA-LUAD",
                    "sample_id": sample,
                    "sample_type": "Primary Tumor",
                    "mapping_status": "resolved_tumor_aliquot",
                }
            )
            manifests[modality].append(
                {
                    "id": file_id,
                    "filename": f"{file_id}.txt",
                    "md5": "abc",
                    "size": 10,
                    "state": "released",
                }
            )

        add("C1", "expression", "C1-01A", "C1-e")
        add("C1", "mutation", "C1-01A", "C1-m")
        add("C2", "expression", "C2-01A", "C2-e")
        add("C2", "mutation", "C2-01B", "C2-m")
        for modality, rows in manifests.items():
            pd.DataFrame(rows).to_csv(
                full_dir / f"manifest_{modality}.tsv", sep="\t", index=False
            )
        mapping_path = tmp_path / "file_mapping.tsv"
        pd.DataFrame(mapping_rows).to_csv(mapping_path, sep="\t", index=False)

        GDCClient(data_types=data_types).align_manifests(
            full_manifest_dir=full_dir,
            mapping_path=mapping_path,
            outdir=aligned_dir,
            aligned_cases_path=tmp_path / "aligned_cases.tsv",
            alignment_report_path=tmp_path / "alignment_report.tsv",
        )

        aligned_expression = pd.read_csv(
            aligned_dir / "manifest_expression.tsv", sep="\t"
        )
        report = pd.read_csv(tmp_path / "alignment_report.tsv", sep="\t")
        cases = pd.read_csv(tmp_path / "aligned_cases.tsv", sep="\t")
        assert aligned_expression["id"].tolist() == ["C1-e"]
        assert cases[["submitter_id", "sample_id"]].to_dict("records") == [
            {"submitter_id": "C1", "sample_id": "C1-01A"}
        ]
        assert report.set_index("case_id").loc["C2", "status"] == (
            "specimen_mismatch"
        )

    def test_download_resume_skips_only_checksum_verified_files(self, tmp_path):
        content = b"verified public payload\n"
        digest = hashlib.md5(content).hexdigest()
        manifest = tmp_path / "manifest.tsv"
        pd.DataFrame(
            [
                {
                    "id": "good-id",
                    "filename": "good.txt",
                    "md5": digest,
                    "size": len(content),
                    "state": "released",
                },
                {
                    "id": "bad-id",
                    "filename": "bad.txt",
                    "md5": digest,
                    "size": len(content),
                    "state": "released",
                },
            ]
        ).to_csv(manifest, sep="\t", index=False)
        good_dir = tmp_path / "downloads" / "good-id"
        bad_dir = tmp_path / "downloads" / "bad-id"
        good_dir.mkdir(parents=True)
        bad_dir.mkdir(parents=True)
        (good_dir / "good.txt").write_bytes(content)
        (bad_dir / "bad.txt").write_bytes(b"wrong payload")

        report = GDCClient.verify_manifest_downloads(
            manifest, tmp_path / "downloads"
        )
        filtered, total, skipped = GDCClient._filter_manifest_skip_existing(
            manifest, tmp_path / "downloads"
        )

        assert report.set_index("file_id")["status"].to_dict() == {
            "good-id": "verified",
            "bad-id": "size_mismatch",
        }
        assert (total, skipped) == (2, 1)
        remaining = pd.read_csv(filtered, sep="\t")
        assert remaining["id"].tolist() == ["bad-id"]

    def test_download_fails_closed_when_client_does_not_produce_file(
        self, tmp_path, monkeypatch
    ):
        manifest = tmp_path / "manifest.tsv"
        pd.DataFrame(
            [
                {
                    "id": "missing-id",
                    "filename": "missing.txt",
                    "md5": "abc",
                    "size": 10,
                    "state": "released",
                }
            ]
        ).to_csv(manifest, sep="\t", index=False)
        client = GDCClient()
        monkeypatch.setattr(client, "_find_gdc_client", lambda: "gdc-client")
        monkeypatch.setattr(
            "pymaftools.io.tcga.client.subprocess.run", lambda *args, **kwargs: None
        )

        with pytest.raises(RuntimeError, match="verification failed"):
            client._download_gdc_client(
                {"mutation": manifest}, tmp_path / "downloads"
            )

    @patch("pymaftools.io.tcga.client.requests.post")
    def test_get_cases(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "hits": [
                    {"cases": [{"submitter_id": "TCGA-44-2655"}]},
                    {"cases": [{"submitter_id": "TCGA-44-2656"}]},
                    {"cases": [{"submitter_id": "TCGA-44-2655"}]},  # duplicate
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = GDCClient()
        cases = client.get_cases("TCGA-LUAD", "expression")

        assert cases == {"TCGA-44-2655", "TCGA-44-2656"}
        mock_post.assert_called_once()

    @patch("pymaftools.io.tcga.client.requests.get")
    def test_get_status(self, mock_get):
        response = MagicMock()
        response.json.return_value = {
            "status": "OK",
            "data_release": "Data Release 46.0",
        }
        mock_get.return_value = response

        status = GDCClient().get_status()

        assert status["data_release"] == "Data Release 46.0"
        response.raise_for_status.assert_called_once()

    @patch("pymaftools.io.tcga.client.requests.post")
    def test_align_cases(self, mock_post):
        """Test alignment returns intersection of case sets."""
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count[0] == 1:  # expression
                resp.json.return_value = {"data": {"hits": [
                    {"cases": [{"submitter_id": "A"}]},
                    {"cases": [{"submitter_id": "B"}]},
                    {"cases": [{"submitter_id": "C"}]},
                ]}}
            else:  # mutation
                resp.json.return_value = {"data": {"hits": [
                    {"cases": [{"submitter_id": "B"}]},
                    {"cases": [{"submitter_id": "C"}]},
                    {"cases": [{"submitter_id": "D"}]},
                ]}}
            return resp

        mock_post.side_effect = side_effect

        client = GDCClient()
        aligned = client.align_cases("TCGA-LUAD", ["expression", "mutation"])

        assert aligned == ["B", "C"]

    @patch("pymaftools.io.tcga.client.requests.post")
    def test_generate_manifests(self, mock_post, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "hits": [
                    {
                        "file_id": "uuid-1",
                        "file_name": "file1.tsv",
                        "file_size": 1000,
                        "md5sum": "abc123",
                        "state": "released",
                        "cases": [{"submitter_id": "TCGA-44-2655"}],
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = GDCClient()
        manifests = client.generate_manifests(
            ["TCGA-44-2655"], "TCGA-LUAD",
            data_types=["mutation"],
            outdir=str(tmp_path),
        )

        assert "mutation" in manifests
        manifest_path = manifests["mutation"]
        assert manifest_path.exists()

        content = manifest_path.read_text()
        assert "uuid-1" in content
        assert "file1.tsv" in content

    @patch("pymaftools.io.tcga.client.requests.post")
    def test_fetch_clinical_table(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "hits": [
                    {
                        "submitter_id": "TCGA-44-2655",
                        "project": {"project_id": "TCGA-LUAD"},
                        "demographic": {
                            "gender": "male",
                            "vital_status": "Dead",
                            "days_to_death": 365,
                            "age_at_index": 67,
                        },
                        "diagnoses": [{
                            "primary_diagnosis": "Adenocarcinoma, NOS",
                            "ajcc_pathologic_stage": "Stage IIA",
                            "ajcc_pathologic_t": "T2a",
                            "ajcc_pathologic_n": "N1",
                            "ajcc_pathologic_m": "M0",
                            "morphology": "8140/3",
                            "tissue_or_organ_of_origin": "Upper lobe, lung",
                            "days_to_last_follow_up": None,
                        }],
                        "exposures": [{
                            "tobacco_smoking_status": "Current smoker",
                            "pack_years_smoked": 40,
                        }],
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = GDCClient()
        df = client.fetch_clinical_table(["TCGA-44-2655"])

        assert len(df) == 1
        assert df.index[0] == "TCGA-44-2655"
        assert df.loc["TCGA-44-2655", "gender"] == "male"
        assert df.loc["TCGA-44-2655", "stage"] == "Stage IIA"
        assert df.loc["TCGA-44-2655", "primary_diagnosis"] == "Adenocarcinoma, NOS"
        assert df.loc["TCGA-44-2655", "smoking_status"] == "Current smoker"


# ------------------------------------------------------------------ #
#  Integration tests (online, requires network)
# ------------------------------------------------------------------ #


@pytest.mark.integration
class TestGDCClientIntegration:
    """Integration tests that hit the real GDC API. Run with: pytest -m integration"""

    def test_get_cases_luad_expression(self):
        client = GDCClient()
        cases = client.get_cases("TCGA-LUAD", "expression")
        assert len(cases) > 400  # Should be ~518

    def test_align_cases_luad(self):
        client = GDCClient()
        aligned = client.align_cases("TCGA-LUAD", ["expression", "mutation"])
        assert len(aligned) > 400  # Should be ~507+

    def test_fetch_clinical_table_small(self):
        client = GDCClient()
        # Use a known TCGA case
        df = client.fetch_clinical_table(["TCGA-44-2655"])
        assert len(df) == 1
        assert "gender" in df.columns
        assert "stage" in df.columns
