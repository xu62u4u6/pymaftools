"""VCF caller-adapter, consensus-merging, and serialization tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import pymaftools
from pymaftools.core.VCF import VCF
from pymaftools.io.vcf import (
    BaseCallerAdapter,
    SampleMetrics,
    VCF_RECORD_COLUMNS,
    get_caller_adapter,
    list_caller_adapters,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "vcf_callers"


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


# ---------------------------------------------------------------------------
# Backwards compatibility & basic behavior
# ---------------------------------------------------------------------------


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
    assert vcf.header["multiallelic_sites_expanded"] == 1


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

    with pytest.warns(UserWarning, match="bcftools norm"):
        merged = VCF.merge_callers([first, second], min_callers=2)

    assert len(merged) == 2
    assert set(merged["callers"]) == {"mutect2;muse"}
    assert merged.header == first.header


# ---------------------------------------------------------------------------
# Built-in caller adapters & header auto-detection
# ---------------------------------------------------------------------------


def test_builtin_caller_registry_lists_core_and_modern_callers():
    callers = VCF.available_callers()
    assert callers == list_caller_adapters()
    for expected in (
        "deepsomatic",
        "dragen",
        "lofreq",
        "muse",
        "mutect2",
        "pindel",
        "strelka2",
        "varscan2",
    ):
        assert expected in callers


@pytest.mark.parametrize(
    "fixture_name,expected_caller,expected_len",
    [
        ("mutect2.vcf", "mutect2", 3),
        ("muse.vcf", "muse", 2),
        ("varscan2_snp.vcf", "varscan2", 2),
        ("varscan2_indel.vcf", "varscan2", 1),
        ("pindel.vcf", "pindel", 2),
        ("strelka2_snvs.vcf", "strelka2", 2),
        ("strelka2_indels.vcf", "strelka2", 1),
        ("deepsomatic.vcf", "deepsomatic", 2),
        ("dragen.vcf", "dragen", 2),
        ("lofreq.vcf", "lofreq", 2),
    ],
)
def test_auto_detects_all_builtin_caller_fixtures(
    fixture_name, expected_caller, expected_len
):
    vcf = VCF.read_vcf(FIXTURE_DIR / fixture_name, caller="auto")
    assert len(vcf) == expected_len
    assert set(vcf["caller"]) == {expected_caller}
    assert vcf.header["caller"] == expected_caller


def test_strelka2_reads_snv_and_indel_files_together():
    vcf = VCF.read_vcf(
        [FIXTURE_DIR / "strelka2_snvs.vcf", FIXTURE_DIR / "strelka2_indels.vcf"],
        caller="strelka",  # alias
    )
    assert list(vcf.index) == [
        "chr17|7675088|C|T",
        "chr7|55191822|T|G",
        "chr12|25245350|C|CTG",
    ]
    assert vcf.loc["chr17|7675088|C|T", "tumor_ref"] == 60
    assert vcf.loc["chr17|7675088|C|T", "tumor_ad"] == 40
    assert vcf.loc["chr17|7675088|C|T", "tumor_af"] == pytest.approx(0.4)
    assert vcf.loc["chr12|25245350|C|CTG", "tumor_ref"] == 45
    assert vcf.loc["chr12|25245350|C|CTG", "tumor_ad"] == 15
    assert vcf.loc["chr12|25245350|C|CTG", "tumor_af"] == pytest.approx(0.25)
    assert vcf.loc["chr12|25245350|C|CTG", "variant_type"] == "INS"


def test_varscan2_reads_snp_and_indel_files_and_accepts_dot_filter():
    vcf = VCF.read_vcf(
        [FIXTURE_DIR / "varscan2_snp.vcf", FIXTURE_DIR / "varscan2_indel.vcf"],
        caller="varscan",  # alias
    )
    assert len(vcf) == 3
    # chr7 row in varscan2_snp.vcf has FILTER='.'
    assert "chr7|55191822|T|G" in vcf.index
    assert vcf.loc["chr7|55191822|T|G", "tumor_af"] == pytest.approx(0.2959)
    assert vcf.loc["chr12|25245350|C|CTG", "variant_type"] == "INS"


def test_deepsomatic_tumor_only_single_sample_and_vaf_field():
    vcf = VCF.read_vcf(FIXTURE_DIR / "deepsomatic.vcf", caller="deepsomatic")
    assert vcf.header["tumor_sample"] == "CASE01"
    assert vcf.header["normal_sample"] is None
    assert vcf.loc["chr17|7675088|C|T", "tumor_af"] == pytest.approx(0.39)
    assert vcf["normal_dp"].isna().all()


def test_lofreq_site_only_vcf_parses_info_dp4():
    vcf = VCF.read_vcf(FIXTURE_DIR / "lofreq.vcf", caller="lofreq")
    assert vcf.loc["chr17|7675088|C|T", "tumor_dp"] == 100
    assert vcf.loc["chr17|7675088|C|T", "tumor_ref"] == 60
    assert vcf.loc["chr17|7675088|C|T", "tumor_ad"] == 40
    assert vcf.loc["chr17|7675088|C|T", "tumor_af"] == pytest.approx(0.4)
    assert vcf.loc["chr7|55191822|T|G", "tumor_ad"] == 20


def test_muse_custom_accepted_filters_includes_tier1():
    default_vcf = VCF.read_vcf(FIXTURE_DIR / "muse.vcf", caller="muse")
    tier1_vcf = VCF.read_vcf(
        FIXTURE_DIR / "muse.vcf",
        caller="muse",
        accepted_filters={"PASS", "Tier1"},
    )
    assert "chr7|55191822|T|G" not in default_vcf.index
    assert "chr7|55191822|T|G" in tier1_vcf.index


def test_custom_caller_adapter_instance_works_without_subclassing_vcf(tmp_path):
    class CustomToolAdapter(BaseCallerAdapter):
        name = "custom_tool"
        supported_variant_types = frozenset({"SNV"})

        def parse_record_metrics(
            self,
            *,
            ref,
            alt,
            format_str,
            sample_value,
            info_str,
            alt_index=0,
        ):
            parts = dict(zip(format_str.split(":"), sample_value.split(":")))
            ref_c = int(parts["RC"])
            alt_c = int(parts["AC"])
            dp = ref_c + alt_c
            return SampleMetrics(dp=dp, ref_count=ref_c, alt_count=alt_c, af=alt_c / dp)

    vcf_path = tmp_path / "custom.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPT_X",
                "chr1\t500\t.\tA\tG\t.\tPASS\t.\tRC:AC\t80:20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    vcf = VCF.read_vcf(vcf_path, caller=CustomToolAdapter())
    assert len(vcf) == 1
    assert vcf.loc["chr1|500|A|G", "caller"] == "custom_tool"
    assert vcf.loc["chr1|500|A|G", "tumor_af"] == pytest.approx(0.2)


def test_unknown_caller_raises_helpful_error():
    with pytest.raises(ValueError, match="Available callers:"):
        get_caller_adapter("not_a_real_caller")


def test_read_vcf_rejects_mixing_different_callers_in_one_call():
    with pytest.raises(ValueError, match="same caller"):
        VCF.read_vcf(
            [FIXTURE_DIR / "mutect2.vcf", FIXTURE_DIR / "muse.vcf"],
            caller="auto",
        )


def test_explicit_sample_override_for_ambiguous_headers(tmp_path):
    vcf_path = tmp_path / "custom_ids.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS01\tS02",
                "1\t1000\t.\tC\tA\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:50,0:0.0:50\t0/1:40,10:0.2:50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pass tumor_sample="):
        VCF.read_vcf(vcf_path, caller="mutect2")

    vcf = VCF.read_vcf(
        vcf_path, caller="mutect2", tumor_sample="S02", normal_sample="S01"
    )
    assert len(vcf) == 1
    assert vcf.loc["1|1000|C|A", "tumor_ad"] == 10
    assert vcf.loc["1|1000|C|A", "normal_dp"] == 50


# ---------------------------------------------------------------------------
# ISS-024 & ISS-011 regressions
# ---------------------------------------------------------------------------


def test_bare_chromosomes_and_empty_vcf_schema_preserved(tmp_path):
    bare_path = tmp_path / "bare.vcf"
    bare_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNORMAL\tTUMOR",
                "10\t200\t.\tA\tG\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:50,0:0.0:50\t0/1:40,10:0.2:50",
                "X\t300\t.\tC\tT\t.\t.\t.\tGT:AD:AF:DP\t0/0:50,0:0.0:50\t0/1:30,20:0.4:50",
                "2\t100\t.\tG\tA\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:50,0:0.0:50\t0/1:35,15:0.3:50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    vcf = VCF.read_vcf(bare_path, caller="mutect2").sort_variants()
    assert list(vcf["chrom"]) == ["2", "10", "X"]

    empty_path = tmp_path / "empty.vcf"
    empty_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNORMAL\tTUMOR",
                "1\t100\t.\tA\tG\t.\tweak_evidence\t.\tGT:AD:AF:DP\t0/0:50,0:0.0:50\t0/1:45,5:0.1:50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    empty_vcf = VCF.read_vcf(empty_path, caller="mutect2")
    assert empty_vcf.empty
    assert list(empty_vcf.columns) == list(VCF_RECORD_COLUMNS)
    # Must not raise KeyError on empty VCF
    assert empty_vcf.filter_variants().empty
    assert empty_vcf.sort_variants().empty


def test_to_file_handles_missing_normal_real_normal_af_and_gzip_roundtrip(tmp_path):
    df = VCF(
        {
            "chrom": ["chr1", "chr1"],
            "pos": [100, 200],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "variant_type": ["SNV", "SNV"],
            "filter": ["PASS", "PASS"],
            "caller": ["mutect2", "mutect2"],
            "tumor_dp": [50, 60],
            "tumor_ref": [30, 40],
            "tumor_ad": [20, 20],
            "tumor_af": [0.4, 0.3333],
            "normal_dp": [40.0, np.nan],
            "normal_ref": [38.0, np.nan],
            "normal_ad": [2.0, np.nan],
            "callers": ["mutect2;strelka2", "mutect2;deepsomatic"],
            "n_callers": [2, 2],
        },
        index=["chr1|100|A|G", "chr1|200|C|T"],
    )
    df.header = {"tumor_sample": "T1", "normal_sample": "N1"}

    out_vcf = tmp_path / "harmonized.vcf"
    df.to_file(out_vcf)
    lines = [
        line
        for line in out_vcf.read_text(encoding="utf-8").splitlines()
        if not line.startswith("##")
    ]
    assert lines[0].split("\t")[-2:] == ["N1", "T1"]

    row1 = lines[1].split("\t")
    row2 = lines[2].split("\t")
    assert len(row1) == 11
    assert len(row2) == 11
    # Row 1 must compute real normal AF = 2/40 = 0.05 (not hardcoded 0.0)
    assert row1[9] == "0/0:40:38,2:0.05"
    assert "CALLERS=mutect2;strelka2;N_CALLERS=2" in row1[7]
    # Row 2 missing normal must write '.' in N1 column and keep tumor in T1 column
    assert row2[9] == "."
    assert row2[10] == "0/1:60:40,20:0.3333"

    # Gzip round-trip test for paired records
    paired_only = VCF(df.iloc[[0]].copy())
    paired_only.header = df.header.copy()
    gz_out = tmp_path / "paired.vcf.gz"
    paired_only.to_file(gz_out)
    reloaded = VCF.read_vcf(gz_out, caller="mutect2")
    assert len(reloaded) == 1
    assert reloaded.loc["chr1|100|A|G", "tumor_ad"] == 20
    assert reloaded.loc["chr1|100|A|G", "normal_ad"] == 2


# ---------------------------------------------------------------------------
# Multi-caller consensus, SNV/INDEL thresholds, concordance, cohort merging
# ---------------------------------------------------------------------------


def test_merge_callers_snv_indel_split_and_snv_only_warning():
    mutect2 = VCF.read_vcf(FIXTURE_DIR / "mutect2.vcf", caller="mutect2")
    muse = VCF.read_vcf(FIXTURE_DIR / "muse.vcf", caller="muse")
    varscan2 = VCF.read_vcf(
        [FIXTURE_DIR / "varscan2_snp.vcf", FIXTURE_DIR / "varscan2_indel.vcf"],
        caller="varscan2",
    )

    # With min_callers=3 and SNV-only MuSE, INDELs have only 2 capable callers -> warn
    with pytest.warns(UserWarning, match="SNV-only caller"):
        strict_all = VCF.merge_callers([mutect2, varscan2, muse], min_callers=3)
    assert "chr12|25245350|C|CTG" not in strict_all.index
    assert list(strict_all.index) == ["chr17|7675088|C|T"]

    # Explicit min_callers_snv=3, min_callers_indel=2 retains the 2-caller INDEL cleanly
    split = VCF.merge_callers(
        [mutect2, varscan2, muse],
        min_callers_snv=3,
        min_callers_indel=2,
        sort=True,
    )
    assert list(split.index) == ["chr12|25245350|C|CTG", "chr17|7675088|C|T"]
    assert split.loc["chr12|25245350|C|CTG", "callers"] == "mutect2;varscan2"


def test_merge_callers_aggregation_modes_and_vaf_spread_warning():
    mutect2 = VCF.read_vcf(FIXTURE_DIR / "mutect2.vcf", caller="mutect2")
    dragen = VCF.read_vcf(FIXTURE_DIR / "dragen.vcf", caller="dragen")
    strelka2 = VCF.read_vcf(
        [FIXTURE_DIR / "strelka2_snvs.vcf", FIXTURE_DIR / "strelka2_indels.vcf"],
        caller="strelka2",
    )

    mean_merged = VCF.merge_callers(
        [mutect2, dragen, strelka2], min_callers=2, aggregation="mean"
    )
    # chr17: mutect2=0.40, dragen=0.41, strelka2=0.40 -> mean ~ 0.403333
    assert mean_merged.loc["chr17|7675088|C|T", "tumor_af"] == pytest.approx(
        (0.40 + 0.41 + 0.40) / 3
    )

    median_merged = VCF.merge_callers(
        [mutect2, dragen, strelka2], min_callers=2, aggregation="median"
    )
    assert median_merged.loc["chr17|7675088|C|T", "tumor_af"] == pytest.approx(0.40)

    # Discordant VAF warning
    discordant = VCF(dragen.copy())
    discordant.loc["chr17|7675088|C|T", "tumor_af"] = 0.05
    with pytest.warns(UserWarning, match="cross-caller VAF spread"):
        merged_disc = VCF.merge_callers(
            [mutect2, discordant], min_callers=2, vaf_spread_warn=0.2
        )
    assert merged_disc.loc["chr17|7675088|C|T", "vaf_spread"] == pytest.approx(0.35)


def test_merge_callers_rejects_mixed_chr_prefixes(tmp_path):
    mutect2 = VCF.read_vcf(FIXTURE_DIR / "mutect2.vcf", caller="mutect2")
    bare_path = tmp_path / "bare.vcf"
    bare_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNORMAL\tTUMOR",
                "17\t7675088\t.\tC\tT\t.\tPASS\t.\tGT:AD:AF:DP\t0/0:50,0:0.0:50\t0/1:60,40:0.4:100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bare = VCF.read_vcf(bare_path, caller="dragen")
    with pytest.raises(ValueError, match="Inconsistent chromosome naming"):
        VCF.merge_callers([mutect2, bare], min_callers=2)


def test_concordance_summary_computes_exact_exclusive_subsets():
    mutect2 = VCF.read_vcf(FIXTURE_DIR / "mutect2.vcf", caller="mutect2")
    muse = VCF.read_vcf(FIXTURE_DIR / "muse.vcf", caller="muse")
    varscan2 = VCF.read_vcf(
        [FIXTURE_DIR / "varscan2_snp.vcf", FIXTURE_DIR / "varscan2_indel.vcf"],
        caller="varscan2",
    )

    summary = VCF.concordance_summary(
        [mutect2, muse, varscan2], min_callers_snv=2, min_callers_indel=2
    )
    # Exclusive subset counts MUST sum exactly to total_unique_variants (no double counting)
    assert (
        sum(summary["exclusive_subsets"].values()) == summary["total_unique_variants"]
    )
    assert summary["exclusive_subsets"]["mutect2;muse;varscan2"] == 1  # chr17
    assert summary["exclusive_subsets"]["mutect2;varscan2"] == 2  # chr7 + chr12
    assert summary["exclusive_subsets"]["muse"] == 1  # chr2
    assert summary["snv_consensus_count"] == 2
    assert summary["indel_consensus_count"] == 1
    assert summary["consensus_count"] == 3
    # Serializable through pymaftools.to_json
    payload = pymaftools.to_json(summary)
    assert '"schema_version"' in payload


def test_merge_cohort_vcfs_never_combines_single_caller_hits_across_samples():
    # Sample A has chr17 ONLY in mutect2; Sample B has chr17 ONLY in muse.
    # Neither sample has 2-caller support -> BOTH samples must yield 0 consensus hits.
    mutect2_full = VCF.read_vcf(FIXTURE_DIR / "mutect2.vcf", caller="mutect2")
    muse_full = VCF.read_vcf(FIXTURE_DIR / "muse.vcf", caller="muse")

    sample_a_mutect2 = VCF(mutect2_full.loc[["chr17|7675088|C|T"]].copy())
    sample_a_mutect2.header = mutect2_full.header.copy()
    sample_a_muse = VCF(muse_full.iloc[0:0].copy())
    sample_a_muse.header = muse_full.header.copy()

    sample_b_mutect2 = VCF(mutect2_full.iloc[0:0].copy())
    sample_b_mutect2.header = mutect2_full.header.copy()
    sample_b_muse = VCF(muse_full.loc[["chr17|7675088|C|T"]].copy())
    sample_b_muse.header = muse_full.header.copy()

    merged, summary_df = VCF.merge_cohort_vcfs(
        {
            "SAMPLE_A": [sample_a_mutect2, sample_a_muse],
            "SAMPLE_B": [sample_b_mutect2, sample_b_muse],
        },
        min_callers=2,
    )
    assert merged["SAMPLE_A"].empty
    assert merged["SAMPLE_B"].empty
    assert summary_df.loc["SAMPLE_A", "consensus_count"] == 0
    assert summary_df.loc["SAMPLE_B", "consensus_count"] == 0


def test_sort_variants_orders_chromosomes_and_positions_numerically():
    df = VCF(
        {
            "chrom": ["chr10", "chr2", "chr2", "chr1", "chrX"],
            "pos": [50, 10000, 2000, 300, 10],
            "ref": ["A", "C", "G", "T", "A"],
            "alt": ["G", "T", "A", "C", "T"],
            "variant_type": ["SNV"] * 5,
            "filter": ["PASS"] * 5,
            "caller": ["mutect2"] * 5,
            "tumor_dp": [50] * 5,
            "tumor_ref": [40] * 5,
            "tumor_ad": [10] * 5,
            "tumor_af": [0.2] * 5,
            "normal_dp": [50] * 5,
            "normal_ref": [50] * 5,
            "normal_ad": [0] * 5,
        },
        index=[
            "chr10|50|A|G",
            "chr2|10000|C|T",
            "chr2|2000|G|A",
            "chr1|300|T|C",
            "chrX|10|A|T",
        ],
    )
    sorted_vcf = df.sort_variants()
    assert list(sorted_vcf.index) == [
        "chr1|300|T|C",
        "chr2|2000|G|A",
        "chr2|10000|C|T",
        "chr10|50|A|G",
        "chrX|10|A|T",
    ]
