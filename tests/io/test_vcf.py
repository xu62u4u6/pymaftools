"""VCF parser and caller-merging regression tests."""

import gzip

import pytest

import pymaftools
from pymaftools.core.VCF import VCF


def _write_vcf(path, tumor_value="0/1:30,5,10:0.111,0.222:45"):
    content = "\n".join(
        [
            "##fileformat=VCFv4.2",
            "##tumor_sample=SAMPLE_T",
            "##normal_sample=SAMPLE_N",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE_N\tSAMPLE_T",
            "chr1\t101\t.\tA\tC,G\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:40,0,0:0,0:40\t"
            + tumor_value,
        ]
    )
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        path.write_text(content + "\n", encoding="utf-8")


def test_vcf_is_available_from_top_level_api():
    assert pymaftools.VCF is VCF


def test_read_vcf_expands_multiallelic_gzip_records(tmp_path):
    path = tmp_path / "calls.vcf.gz"
    _write_vcf(path)

    vcf = VCF.read_vcf(path, caller="mutect2")

    assert list(vcf.index) == ["chr1|101|A|C", "chr1|101|A|G"]
    assert vcf["tumor_ad"].tolist() == [5, 10]
    assert vcf["tumor_af"].tolist() == pytest.approx([0.111, 0.222])
    assert vcf["normal_ad"].tolist() == [0, 0]


def test_read_vcf_reports_incomplete_format_values(tmp_path):
    path = tmp_path / "invalid.vcf"
    _write_vcf(path, tumor_value="0/1:30,5")

    with pytest.raises(ValueError, match="Failed to parse FORMAT"):
        VCF.read_vcf(path, caller="mutect2")


def test_merge_callers_tracks_distinct_support(tmp_path):
    path = tmp_path / "calls.vcf"
    _write_vcf(path)
    first = VCF.read_vcf(path, caller="mutect2")
    second = VCF(first.copy())
    second["caller"] = "muse"
    second.header = first.header.copy()

    merged = VCF.merge_callers([first, second], min_callers=2)

    assert len(merged) == 2
    assert set(merged["callers"]) == {"mutect2;muse"}
    assert merged.header == first.header
