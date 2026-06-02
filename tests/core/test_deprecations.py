"""Deprecated aliases keep working but warn and delegate to the canonical name.

These guard backward compatibility for the snake_case / pandas-convention API
rename (calculate_TMB -> calculate_tmb, to_MAF/write_maf -> to_maf,
write_SigProfilerMatrixGenerator_format -> to_sigprofiler,
VCF.from_file -> VCF.read_vcf).
"""

from unittest.mock import patch

import pandas as pd
import pytest

from pymaftools import PivotTable
from pymaftools.core.MAF import MAF
from pymaftools.core.VCF import VCF


def test_calculate_TMB_alias_warns_and_delegates():
    pt = PivotTable(
        pd.DataFrame({"s1": [True, False], "s2": [False, True]}, index=["TP53", "KRAS"])
    )
    pt.sample_metadata["mutations_count"] = [3, 5]

    with pytest.warns(DeprecationWarning, match="calculate_tmb"):
        out = pt.calculate_TMB(default_capture_size=40)

    assert "TMB" in out.sample_metadata.columns


def test_to_sigprofiler_alias_warns_and_delegates(tmp_path):
    maf = MAF(
        pd.DataFrame(
            {
                "sample_ID": ["s1", "s2"],
                "Chromosome": ["chr1", "chr2"],
                "Start_Position": [1, 2],
                "End_Position": [1, 2],
                "Reference_Allele": ["C", "G"],
                "Tumor_Seq_Allele2": ["T", "A"],
                "Variant_Type": ["SNP", "SNP"],
            }
        )
    )
    out_path = tmp_path / "sig.tsv"

    with pytest.warns(DeprecationWarning, match="to_sigprofiler"):
        maf.write_SigProfilerMatrixGenerator_format(out_path)

    assert out_path.exists()


def test_vcf_from_file_alias_warns_and_delegates():
    with patch.object(VCF, "read_vcf", return_value="sentinel") as stub:
        with pytest.warns(DeprecationWarning, match="read_vcf"):
            result = VCF.from_file("x.vcf", "mutect2")

    assert result == "sentinel"
    stub.assert_called_once_with("x.vcf", "mutect2")
