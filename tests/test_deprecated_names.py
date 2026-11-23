"""0.6.0 name unification: every old name still works, warns, and names 1.0.

Old names must return exactly the object/result of the new name, so code
written against earlier releases (including published analysis scripts) keeps
producing the same numbers until pymaftools 1.0.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import pymaftools
from pymaftools import Cohort, PivotTable
from pymaftools._deprecation import REMOVAL_VERSION
from pymaftools.core import CopyNumberVariationTable as cnv_module
from pymaftools.core.MAF import MAF
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


def test_removal_version_is_one_point_zero():
    assert REMOVAL_VERSION == "1.0"


@pytest.mark.parametrize("name", ["PCA_CCA"])
def test_project_specific_names_left_top_level_namespace(name):
    assert name not in pymaftools.__all__
    with pytest.warns(DeprecationWarning, match=rf"pymaftools\.{name}.*1\.0"):
        value = getattr(pymaftools, name)
    assert value is not None


def test_top_level_deprecated_names_resolve_to_submodule_objects():
    from pymaftools.utils.reduction import PCA_CCA

    with pytest.warns(DeprecationWarning):
        assert pymaftools.PCA_CCA is PCA_CCA


def test_from_import_of_deprecated_top_level_name_warns():
    with pytest.warns(DeprecationWarning, match="PCA_CCA"):
        from pymaftools import PCA_CCA  # noqa: F401


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
    with pytest.warns(DeprecationWarning, match=r"Cohort\.to_h5\(\).*") as record:
        cohort.to_hdf5(legacy)
    assert "1.0" in str(record[0].message)

    with pytest.warns(DeprecationWarning, match=r"Cohort\.read_h5\(\)"):
        from_legacy = Cohort.read_hdf5(legacy)
    from_canonical = Cohort.read_h5(canonical)
    pd.testing.assert_frame_equal(
        pd.DataFrame(from_legacy.mutation), pd.DataFrame(from_canonical.mutation)
    )


def test_pivot_pca_alias_matches_pca():
    table = _table()
    expected_df, expected_var, _ = table.pca(to_binary=True)
    with pytest.warns(DeprecationWarning, match=r"PivotTable\.pca\(\)"):
        legacy_df, legacy_var, _ = table.PCA(to_binary=True)
    pd.testing.assert_frame_equal(legacy_df, expected_df)
    assert list(legacy_var) == list(expected_var)


def test_misspelled_variant_classification_alias_warns_on_class_and_instance():
    with pytest.warns(DeprecationWarning, match="valid_variant_classification"):
        assert MAF.vaild_variant_classfication is MAF.valid_variant_classification
    maf = MAF(pd.DataFrame({"Hugo_Symbol": ["TP53"]}))
    with pytest.warns(DeprecationWarning):
        assert maf.vaild_variant_classfication is MAF.valid_variant_classification


def test_tcga_cnv_helpers_have_snake_case_names():
    barcode = "TCGA-05-4389-01A"
    with pytest.warns(DeprecationWarning, match=r"tcga_sample_type\(\)"):
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


@pytest.mark.parametrize(
    "call",
    [
        lambda: _table().calculate_TMB(),
        lambda: pymaftools.PCA_CCA,
    ],
)
def test_every_deprecation_message_names_the_removal_release(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            call()
        except (
            Exception
        ):  # the call itself may need more input; only the warning matters
            pass
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations
    assert all(f"pymaftools {REMOVAL_VERSION}" in str(w.message) for w in deprecations)
