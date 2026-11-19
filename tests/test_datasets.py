"""Tests for the bundled example datasets.

These guard the promise that an installed user can read the documented example
MAFs with no external download (the gap the doc-only-user agent surfaced)."""

import pytest

from pymaftools import (
    PivotTable,
    example_maf_path,
    example_table_path,
    load_example_maf,
    load_example_table,
)
from pymaftools.core.MAF import MAF


def test_example_maf_paths_exist():
    for name in ("multisample", "single_sample"):
        path = example_maf_path(name)
        assert path.exists(), f"bundled example {name} missing at {path}"
        assert path.suffix == ".maf"


def test_example_maf_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown example MAF"):
        example_maf_path("does_not_exist")


def test_load_example_multisample_is_readable():
    """The documented entry point returns a MAF that builds a multi-sample
    pivot table — i.e. the README/getting-started workflow actually runs."""
    maf = load_example_maf("multisample")
    assert isinstance(maf, MAF)

    table = maf.to_pivot_table()
    assert table.shape[0] > 0  # genes
    # the multisample example keeps its samples distinct (not collapsed to one)
    assert table.shape[1] > 1


def test_load_example_single_sample_is_readable():
    maf = load_example_maf("single_sample")
    assert isinstance(maf, MAF)
    assert maf.to_pivot_table().shape[0] > 0


def test_bundled_hdf5_table_has_installed_package_entry_points():
    path = example_table_path()

    assert path.exists()
    assert path.suffix == ".h5"
    table = load_example_table()
    assert isinstance(table, PivotTable)
    assert table.shape[0] > 0
    assert table.shape[1] > 0
