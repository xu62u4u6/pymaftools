import pandas as pd

from pymaftools.core.MAF import MAF
from pymaftools.core.PivotTable import PivotTable


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
    maf.write_SigProfilerMatrixGenerator_format(out_path)
    written = pd.read_csv(out_path, sep="\t")

    assert "sample_ID" in written.columns
    assert set(written["Variant_Type"].unique()).issubset({"SNP", "INS", "DEL"})


def test_get_protein_info_returns_mutation_summary():
    maf = MAF(_build_maf_df())

    aa_length, mutations = maf.get_protein_info("TP53")

    assert aa_length == 400
    assert isinstance(mutations, list)
    assert all({"position", "type", "count"}.issubset(m.keys()) for m in mutations)
