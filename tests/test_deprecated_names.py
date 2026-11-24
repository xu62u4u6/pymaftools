"""0.6.0 name unification: every old name still works, warns, and names 1.0.

Old names must return exactly the object/result of the new name, so code
written against earlier releases (including published analysis scripts) keeps
producing the same numbers until pymaftools 1.0.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pandas as pd
import pytest

import pymaftools
from pymaftools import Cohort, PivotTable
from pymaftools.core import CopyNumberVariationTable as cnv_module
from pymaftools.core.MAF import MAF
from pymaftools.core.VCF import VCF
from pymaftools.utils import geneinfo


def _table() -> PivotTable:
    table = PivotTable(
        pd.DataFrame(
            {
                "s1": [True, False, True],
                "s2": [False, True, True],
                "s3": [True, True, False],
            },
            index=["TP53", "KRAS", "EGFR"],
        )
    )
    table.sample_metadata["group"] = ["a", "b", "a"]
    return table


def test_top_level_deprecated_pca_cca_warns_and_resolves():
    from pymaftools.utils.reduction import PCA_CCA as canonical

    assert "PCA_CCA" not in pymaftools.__all__
    with pytest.warns(DeprecationWarning, match=r"pymaftools\.PCA_CCA.*1\.0"):
        resolved = pymaftools.PCA_CCA
    assert resolved is canonical

    with pytest.warns(DeprecationWarning, match=r"pymaftools\.PCA_CCA.*1\.0"):
        from pymaftools import PCA_CCA as imported

    assert imported is canonical


@pytest.mark.parametrize(
    ("name", "hint"),
    [
        ("ASCStackingModel", "predict_proba"),
        ("read_GMT", "gseapy.read_gmt"),
        ("read_gmt", "gseapy.read_gmt"),
        ("fetch_msigdb_geneset", "gseapy.Msigdb"),
    ],
)
def test_removed_names_raise_with_replacement(name, hint):
    with pytest.raises(AttributeError, match=rf"removed in 0\.6\.0.*{hint}"):
        getattr(pymaftools, name)
    assert name not in pymaftools.__all__


def test_removed_modules_are_gone():
    import importlib.util

    import pymaftools.model.StackingModel as stacking

    assert importlib.util.find_spec("pymaftools.utils.geneset") is None
    assert not hasattr(stacking, "ASCStackingModel")


def test_cohort_hdf5_aliases_round_trip_identically(tmp_path):
    cohort = Cohort(name="demo")
    cohort.add_table(_table(), "mutation")

    canonical = tmp_path / "canonical.h5"
    legacy = tmp_path / "legacy.h5"
    cohort.to_h5(canonical)
    with pytest.warns(DeprecationWarning, match=r"1\.0.*Cohort\.to_h5\(\)"):
        cohort.to_hdf5(legacy)

    with pytest.warns(DeprecationWarning, match=r"1\.0.*Cohort\.read_h5\(\)"):
        from_legacy = Cohort.read_hdf5(legacy)
    from_canonical = Cohort.read_h5(canonical)
    pd.testing.assert_frame_equal(
        pd.DataFrame(from_legacy.mutation), pd.DataFrame(from_canonical.mutation)
    )


def test_pivot_sqlite_warns_in_favor_of_hdf5(tmp_path):
    table = PivotTable(pd.DataFrame({"s1": [True]}, index=["TP53"]))
    db_path = tmp_path / "pivot.db"

    with pytest.warns(DeprecationWarning, match=r"1\.0.*to_h5"):
        table.to_sqlite(db_path)
    with pytest.warns(DeprecationWarning, match=r"1\.0.*read_h5"):
        restored = PivotTable.read_sqlite(db_path)

    pd.testing.assert_frame_equal(
        pd.DataFrame(restored), pd.DataFrame(table), check_names=False
    )


def test_pivot_pca_alias_matches_pca():
    table = _table()
    expected_df, expected_var, _ = table.pca(to_binary=True)
    with pytest.warns(DeprecationWarning, match=r"1\.0.*PivotTable\.pca\(\)"):
        legacy_df, legacy_var, _ = table.PCA(to_binary=True)
    pd.testing.assert_frame_equal(legacy_df, expected_df)
    assert list(legacy_var) == list(expected_var)


def test_calculate_TMB_alias_warns_and_matches_calculate_tmb():
    table = _table()
    table.sample_metadata["mutations_count"] = [3, 5, 2]

    with pytest.warns(FutureWarning, match="legacy normalized count"):
        expected = table.calculate_tmb(default_capture_size=40)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = table.calculate_TMB(default_capture_size=40)
    dep_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("calculate_tmb" in m and "1.0" in m for m in dep_msgs)
    pd.testing.assert_series_equal(
        legacy.sample_metadata["TMB"], expected.sample_metadata["TMB"]
    )


def test_to_sigprofiler_alias_warns_and_matches_canonical(tmp_path):
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
    canonical_path = tmp_path / "canonical.tsv"
    legacy_path = tmp_path / "legacy.tsv"

    maf.to_sigprofiler(canonical_path)
    with pytest.warns(DeprecationWarning, match=r"1\.0.*to_sigprofiler"):
        maf.write_SigProfilerMatrixGenerator_format(legacy_path)

    assert legacy_path.read_text() == canonical_path.read_text()


def test_vcf_from_file_alias_warns_and_delegates():
    with patch.object(VCF, "read_vcf", return_value="sentinel") as stub:
        with pytest.warns(DeprecationWarning, match=r"1\.0.*read_vcf"):
            result = VCF.from_file("x.vcf", "mutect2")

    assert result == "sentinel"
    stub.assert_called_once_with("x.vcf", "mutect2")


def test_misspelled_variant_classification_alias_warns_on_class_and_instance():
    with pytest.warns(DeprecationWarning, match=r"1\.0.*valid_variant_classification"):
        assert MAF.vaild_variant_classfication is MAF.valid_variant_classification
    maf = MAF(pd.DataFrame({"Hugo_Symbol": ["TP53"]}))
    with pytest.warns(DeprecationWarning, match=r"1\.0.*valid_variant_classification"):
        assert maf.vaild_variant_classfication is MAF.valid_variant_classification


def test_tcga_cnv_helpers_have_snake_case_names():
    barcode = "TCGA-05-4389-01A"
    with pytest.warns(DeprecationWarning, match=r"1\.0.*tcga_sample_type\(\)"):
        assert cnv_module.TCGA_sample_type(barcode) == cnv_module.tcga_sample_type(
            barcode
        )
    assert cnv_module.read_TCGA_ASCAT3_CNV_file_sheet.__wrapped__ is (
        cnv_module.read_tcga_ascat3_cnv_file_sheet
    )
    assert (
        cnv_module.get_target_sample_ID.__wrapped__ is cnv_module.get_target_sample_id
    )


def test_geneinfo_aliases_delegate():
    assert geneinfo.get_ncbi_gene_ID.__wrapped__ is geneinfo.get_ncbi_gene_id
    assert geneinfo.get_ncbi_gene_IDs.__wrapped__ is geneinfo.get_ncbi_gene_ids


def test_maf_stats_plot_aliases_warn_and_match_canonical():
    maf_df = pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53", "KRAS", "EGFR"],
            "Tumor_Sample_Barcode": ["S1", "S1", "S2"],
            "sample_ID": ["S1", "S1", "S2"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Missense_Mutation",
            ],
            "Variant_Type": ["SNP", "SNP", "SNP"],
            "Reference_Allele": ["C", "G", "C"],
            "Tumor_Seq_Allele2": ["T", "A", "A"],
        }
    )
    maf = MAF(maf_df)
    other = MAF(maf_df.iloc[[2]].copy())

    expected_summary = maf.summary_stats()
    with pytest.warns(DeprecationWarning, match=r"1\.0.*MAF\.summary_stats\(\)"):
        legacy_summary = maf.plot.summary_stats()
    assert legacy_summary == expected_summary

    expected_cmp = maf.compare_cohorts(other, cohort1_name="A", cohort2_name="B")
    with pytest.warns(DeprecationWarning, match=r"1\.0.*MAF\.compare_cohorts\(\)"):
        legacy_cmp = maf.plot.compare_cohorts(other, cohort1_name="A", cohort2_name="B")
    pd.testing.assert_frame_equal(legacy_cmp, expected_cmp)


def test_pivot_stats_and_plot_aliases_warn_and_match_canonical():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = _table()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        expected_interactions = table.somatic_interactions(top=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_interactions = table.plot.somatic_interactions_stats(top=3)
    dep_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any(
        "PivotTable.somatic_interactions()" in m and "1.0" in m for m in dep_msgs
    )
    pd.testing.assert_frame_equal(legacy_interactions, expected_interactions)

    expected_pca_df, expected_var, _ = table.plot.pca(color_col="group")
    plt.close("all")
    with pytest.warns(DeprecationWarning, match=r"1\.0.*PivotTable\.plot\.pca\(\)"):
        legacy_pca_df, legacy_var, _ = table.plot.plot_pca_samples(color_col="group")
    plt.close("all")
    pd.testing.assert_frame_equal(legacy_pca_df, expected_pca_df)
    assert list(legacy_var) == list(expected_var)

    data = pd.DataFrame({"group": ["a", "a", "b", "b"], "val": [1.0, 2.0, 5.0, 6.0]})
    _, ax1 = plt.subplots()
    table.plot.boxplot(data, group_col="group", test_col="val", ax=ax1, verbose=False)
    _, ax2 = plt.subplots()
    with pytest.warns(DeprecationWarning, match=r"1\.0.*PivotTable\.plot\.boxplot\(\)"):
        table.plot.plot_boxplot_with_annot(
            data, group_col="group", test_col="val", ax=ax2, verbose=False
        )
    assert [t.get_text() for t in ax2.get_xticklabels()] == [
        t.get_text() for t in ax1.get_xticklabels()
    ]
    plt.close("all")
