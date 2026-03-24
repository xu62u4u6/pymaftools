"""
Tests for GDC / TCGA IO module.

Tests are split into:
- Unit tests (offline, mocked): barcode parsing, manifest generation logic
- Integration tests (online, requires network): actual GDC API queries

Integration tests are marked with @pytest.mark.integration and skipped by default.
Run with: pytest -m integration
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from pymaftools.io.tcga import GDCClient, parse_tcga_barcode, DATA_TYPE_CONFIGS


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

    def test_data_type_configs(self):
        """Verify all expected data types are configured."""
        expected = {"expression", "mutation", "cnv", "methylation", "clinical"}
        assert set(DATA_TYPE_CONFIGS.keys()) == expected
        for key, config in DATA_TYPE_CONFIGS.items():
            assert "data_type" in config
            assert "label" in config

    @patch("pymaftools.io.tcga.requests.post")
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

    @patch("pymaftools.io.tcga.requests.post")
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

    @patch("pymaftools.io.tcga.requests.post")
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

    @patch("pymaftools.io.tcga.requests.post")
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
