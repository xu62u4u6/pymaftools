import gzip

import pandas as pd
import pytest

from pymaftools.core.MAF import MAF
from pymaftools.core.PivotTable import PivotTable


# Column header + a few data rows shared by every header-format variant below.
# Only the leading comment lines differ between files, so a correct reader must
# return identical content regardless of how many comment lines precede it.
_MAF_HEADER = "\t".join(
    [
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "Variant_Type",
        "Protein_position",
    ]
)
_MAF_ROWS = [
    "TP53\tchr17\t7577120\t7577120\tC\tC\tT\tMissense_Mutation\tSNP\t273/393",
    "TP53\tchr17\t7578406\t7578406\tG\tG\tA\tNonsense_Mutation\tSNP\t196/393",
    "EGFR\tchr7\t55259515\t55259515\tT\tT\tG\tMissense_Mutation\tSNP\t858/1210",
    "KRAS\tchr12\t25398284\t25398284\tC\tC\tA\tMissense_Mutation\tSNP\t12/189",
]


def _write_maf(path, comment_lines):
    """Write the shared MAF body, prefixed with the given comment lines."""
    lines = list(comment_lines) + [_MAF_HEADER] + _MAF_ROWS
    path.write_text("\n".join(lines) + "\n")
    return path


# Header-format variants: (id, list of leading comment lines).
# "multi_comment" reproduces the exact failure Allison reported
# (ParserError: Expected 1 fields ..., saw N) on the old skiprows=1 reader.
_HEADER_FORMATS = [
    ("no_comment", []),
    ("one_comment", ["#version 2.4"]),
    (
        "multi_comment",
        ["#version 2.4", "#filedate 20240101", "#annotator vcf2maf"],
    ),
]


def _build_maf_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53", "TP53", "EGFR", "EGFR", "KRAS", "KRAS", "PIK3CA", "PIK3CA"],
            "Start_Position": [100, 101, 200, 201, 300, 301, 400, 401],
            "End_Position": [100, 101, 200, 201, 300, 301, 400, 401],
            "Reference_Allele": ["C", "G", "A", "A", "G", "G", "T", "T"],
            "Tumor_Seq_Allele1": ["C", "G", "A", "A", "G", "G", "T", "T"],
            "Tumor_Seq_Allele2": ["T", "A", "T", "C", "A", "T", "C", "G"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Frame_Shift_Del",
                "Splice_Site",
                "Silent",
                "Missense_Mutation",
                "Missense_Mutation",
                "Frame_Shift_Ins",
            ],
            "Variant_Type": ["SNP", "SNP", "DEL", "SNP", "SNP", "SNP", "INS", "SNP"],
            "Tumor_Sample_Barcode": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "sample_ID": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "Protein_position": ["100/400", "120/400", "50/300", "60/300", "70/250", "71/250", "30/150", "31/150"],
        }
    ).reset_index(drop=True)


def test_to_pivot_table_returns_expected_axes_and_metadata():
    maf = MAF(_build_maf_df())

    table = maf.to_pivot_table()

    assert isinstance(table, PivotTable)
    assert set(table.index) == {"TP53", "EGFR", "KRAS", "PIK3CA"}
    assert set(table.columns) == {"s1", "s2", "s3", "s4"}
    assert "mutations_count" in table.sample_metadata.columns
    assert table.loc["TP53", "s1"] == "Multi_Hit"


def test_to_gene_table_matches_to_pivot_table_alias():
    """to_gene_table is canonical; to_pivot_table is a backward-compatible alias."""
    maf = MAF(_build_maf_df())

    gene_table = maf.to_gene_table()
    aliased = maf.to_pivot_table()

    assert isinstance(gene_table, PivotTable)
    assert set(gene_table.index) == set(aliased.index)
    assert set(gene_table.columns) == set(aliased.columns)


def test_to_maf_canonical_and_deprecated_aliases(tmp_path):
    """to_maf is canonical; to_MAF/write_maf warn but produce identical output."""
    maf = MAF(_build_maf_df())

    canonical = tmp_path / "canonical.maf"
    maf.to_maf(canonical)
    expected = canonical.read_text()

    legacy_camel = tmp_path / "camel.maf"
    with pytest.warns(DeprecationWarning, match="to_maf"):
        maf.to_MAF(legacy_camel)
    assert legacy_camel.read_text() == expected

    legacy_write = tmp_path / "write.maf"
    with pytest.warns(DeprecationWarning, match="to_maf"):
        maf.write_maf(legacy_write)
    assert legacy_write.read_text() == expected


def test_merge_mutations_covers_false_single_and_multi_hit():
    assert MAF.merge_mutations(pd.Series([False, False])) is False
    assert MAF.merge_mutations(pd.Series(["Splice_Site", False], index=[0, 10])) == "Splice_Site"
    assert MAF.merge_mutations(pd.Series(["Missense_Mutation", "Nonsense_Mutation"])) == "Multi_Hit"


def test_filter_maf_and_select_samples():
    maf = MAF(_build_maf_df())

    filtered = maf.filter_maf(MAF.nonsynonymous_types)
    selected = maf.select_samples(["s1", "s2"])

    assert set(filtered["Variant_Classification"]).issubset(set(MAF.nonsynonymous_types))
    assert set(selected["sample_ID"]) == {"s1", "s2"}


def test_change_index_level_builds_composite_index():
    maf = MAF(_build_maf_df())

    reindexed = maf.change_index_level()

    expected = "TP53|100|100|C|C|T"
    assert expected in reindexed.index


def test_to_base_change_pivot_table_computes_ti_tv():
    base_change_df = pd.DataFrame(
        {
            "Hugo_Symbol": [f"G{i}" for i in range(12)],
            "Start_Position": list(range(1, 13)),
            "End_Position": list(range(1, 13)),
            "Reference_Allele": ["A", "C", "G", "T", "A", "A", "C", "C", "G", "G", "T", "T"],
            "Tumor_Seq_Allele1": ["A"] * 12,
            "Tumor_Seq_Allele2": ["G", "T", "A", "C", "C", "T", "A", "G", "C", "T", "A", "G"],
            "Variant_Classification": ["Missense_Mutation"] * 12,
            "Variant_Type": ["SNP"] * 12,
            "Tumor_Sample_Barcode": ["s1"] * 12,
            "sample_ID": ["s1"] * 12,
            "Protein_position": ["1/100"] * 12,
        }
    ).reset_index(drop=True)
    maf = MAF(base_change_df)

    base_change = maf.to_base_change_pivot_table()

    assert isinstance(base_change, PivotTable)
    assert "ti" in base_change.sample_metadata.columns
    assert "tv" in base_change.sample_metadata.columns
    assert "ti/tv" in base_change.sample_metadata.columns


def test_read_csv_reindex_and_sigprofiler_export(tmp_path):
    maf_df = _build_maf_df()
    csv_path = tmp_path / "maf.tsv"
    maf_df.to_csv(csv_path, sep="\t", index=False)

    maf = MAF.read_csv(csv_path, reindex=True)
    out_path = tmp_path / "sigprofiler.tsv"
    maf.to_sigprofiler(out_path)
    written = pd.read_csv(out_path, sep="\t")

    assert "sample_ID" in written.columns
    assert set(written["Variant_Type"].unique()).issubset({"SNP", "INS", "DEL"})


def test_get_protein_info_returns_mutation_summary():
    maf = MAF(_build_maf_df())

    aa_length, mutations = maf.get_protein_info("TP53")

    assert aa_length == 400
    assert isinstance(mutations, list)
    assert all({"position", "type", "count"}.issubset(m.keys()) for m in mutations)


@pytest.mark.parametrize("fmt_id, comment_lines", _HEADER_FORMATS, ids=[f[0] for f in _HEADER_FORMATS])
def test_read_maf_handles_varying_comment_lines(tmp_path, fmt_id, comment_lines):
    """read_maf must parse files with 0, 1, or many leading comment lines."""
    maf_path = _write_maf(tmp_path / f"{fmt_id}.maf", comment_lines)

    maf = MAF.read_maf(maf_path, sample_ID="s1")

    # All required columns present and correctly aligned (no header drift).
    assert set(MAF.index_col).issubset(maf.columns)
    assert "Variant_Classification" in maf.columns
    # 4 data rows, every row tagged with the requested sample_ID.
    assert len(maf) == 4
    assert (maf["sample_ID"] == "s1").all()
    # Composite index built from index_col.
    assert "TP53|7577120|7577120|C|C|T" in maf.index


def test_read_maf_is_identical_across_header_formats(tmp_path):
    """Same body + different comment headers => identical parsed MAF."""
    frames = []
    for fmt_id, comment_lines in _HEADER_FORMATS:
        path = _write_maf(tmp_path / f"{fmt_id}.maf", comment_lines)
        maf = MAF.read_maf(path, sample_ID="s1")
        frames.append(pd.DataFrame(maf).reset_index(drop=True))

    for other in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], other)


def test_read_maf_multi_comment_regression(tmp_path):
    """Regression for the reported 'Expected 1 fields ... saw N' ParserError.

    The old reader hardcoded skiprows=1, so multiple comment lines left a
    '#'-comment as the header (1 field) and the real header (N fields) on a
    later line, raising pandas.errors.ParserError. It must now read cleanly.
    """
    maf_path = _write_maf(
        tmp_path / "multi.maf",
        ["#version 2.4", "#filedate 20240101", "#annotator vcf2maf"],
    )

    maf = MAF.read_maf(maf_path, sample_ID="s1")

    assert list(maf["Hugo_Symbol"]) == ["TP53", "TP53", "EGFR", "KRAS"]


def test_count_leading_comment_lines(tmp_path):
    for fmt_id, comment_lines in _HEADER_FORMATS:
        path = _write_maf(tmp_path / f"{fmt_id}.maf", comment_lines)
        assert MAF._count_leading_comment_lines(path) == len(comment_lines)


def test_read_maf_supports_gzip_with_leading_comments(tmp_path):
    path = tmp_path / "mutations.maf.gz"
    lines = ["#version 2.4", "#filedate 20240101", _MAF_HEADER] + _MAF_ROWS
    with gzip.open(path, mode="wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    maf = MAF.read_maf(path, sample_ID="sample-gz")

    assert MAF._count_leading_comment_lines(path) == 2
    assert len(maf) == 4
    assert (maf["sample_ID"] == "sample-gz").all()
    assert "TP53|7577120|7577120|C|C|T" in maf.index


def test_read_maf_to_pivot_table_and_tmb_end_to_end(tmp_path):
    """Full flow: read two single-sample MAFs -> merge -> pivot -> TMB."""
    path_a = _write_maf(tmp_path / "a.maf", ["#version 2.4"])
    path_b = _write_maf(
        tmp_path / "b.maf",
        ["#version 2.4", "#filedate 20240101"],
    )

    maf = MAF.merge_mafs(
        [
            MAF.read_maf(path_a, sample_ID="sample_A"),
            MAF.read_maf(path_b, sample_ID="sample_B"),
        ]
    )
    pivot = maf.to_pivot_table()

    # to_pivot_table provides mutations_count but NOT TMB.
    assert "mutations_count" in pivot.sample_metadata.columns
    assert "TMB" not in pivot.sample_metadata.columns

    # calculate_tmb returns a NEW table; the original is left untouched.
    with_tmb = pivot.calculate_tmb(default_capture_size=40)
    assert "TMB" not in pivot.sample_metadata.columns
    assert "TMB" in with_tmb.sample_metadata.columns

    expected = (
        with_tmb.sample_metadata["mutations_count"] / 40
    )
    pd.testing.assert_series_equal(
        with_tmb.sample_metadata["TMB"], expected, check_names=False
    )


# --- sample_ID resolution ---------------------------------------------------
# A standard multi-sample MAF carries per-row sample identity in
# Tumor_Sample_Barcode. read_maf must preserve those samples by default,
# instead of collapsing the whole file into one sample.
_MULTI_SAMPLE_HEADER = "\t".join(
    [
        "Hugo_Symbol",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele1",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "Tumor_Sample_Barcode",
    ]
)
_MULTI_SAMPLE_ROWS = [
    "TP53\t1\t1\tC\tC\tT\tMissense_Mutation\tpatient_1",
    "EGFR\t2\t2\tA\tA\tG\tMissense_Mutation\tpatient_1",
    "KRAS\t3\t3\tG\tG\tA\tMissense_Mutation\tpatient_2",
]


def _write_multi_sample_maf(path):
    path.write_text(
        "#version 2.4\n"
        + _MULTI_SAMPLE_HEADER
        + "\n"
        + "\n".join(_MULTI_SAMPLE_ROWS)
        + "\n"
    )
    return path


def test_read_maf_defaults_sample_id_to_tumor_sample_barcode(tmp_path):
    """Without sample_ID, per-row Tumor_Sample_Barcode keeps samples distinct."""
    maf_path = _write_multi_sample_maf(tmp_path / "cohort.maf")

    maf = MAF.read_maf(maf_path)

    assert list(maf["sample_ID"]) == ["patient_1", "patient_1", "patient_2"]
    # The whole point: a multi-sample MAF yields multiple pivot columns.
    pivot = maf.to_pivot_table()
    assert set(pivot.columns) == {"patient_1", "patient_2"}


def test_read_maf_explicit_sample_id_overrides_barcode(tmp_path):
    """Explicit sample_ID still collapses the file to one sample (back-compat)."""
    maf_path = _write_multi_sample_maf(tmp_path / "cohort.maf")

    maf = MAF.read_maf(maf_path, sample_ID="s1")

    assert (maf["sample_ID"] == "s1").all()


def test_read_maf_prefix_suffix_applied_per_row(tmp_path):
    maf_path = _write_multi_sample_maf(tmp_path / "cohort.maf")

    maf = MAF.read_maf(maf_path, preffix="C_", suffix="_T")

    assert list(maf["sample_ID"]) == ["C_patient_1_T", "C_patient_1_T", "C_patient_2_T"]


def test_read_maf_raises_when_no_sample_id_and_no_barcode(tmp_path):
    """Fail loud rather than silently mislabel when sample identity is unknown."""
    maf_path = _write_maf(tmp_path / "no_barcode.maf", ["#version 2.4"])

    with pytest.raises(ValueError, match="Tumor_Sample_Barcode"):
        MAF.read_maf(maf_path)
