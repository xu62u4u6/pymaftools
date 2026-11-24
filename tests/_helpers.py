"""Shared test assertion helpers (modeled after anndata.tests.helpers)."""

from __future__ import annotations

import pandas as pd

from pymaftools.core.PivotTable import PivotTable


def assert_pivot_equal(
    actual: PivotTable,
    expected: PivotTable,
    *,
    check_type: bool = True,
    check_names: bool = False,
    check_dtype: bool = True,
    check_exact: bool = False,
    check_context: bool = True,
) -> None:
    """Assert full container equality for PivotTable and its subclasses.

    Compares subclass type, data matrix, ``feature_metadata``,
    ``sample_metadata``, and (when ``check_context=True``)
    ``observation_mask``, ``sample_manifest``, and ``value_scale``.
    """
    if check_type:
        assert type(actual) is type(expected), (
            f"Expected {type(expected).__name__}, got {type(actual).__name__}"
        )

    pd.testing.assert_frame_equal(
        pd.DataFrame(actual),
        pd.DataFrame(expected),
        check_names=check_names,
        check_dtype=check_dtype,
        check_exact=check_exact,
    )
    pd.testing.assert_frame_equal(
        actual.feature_metadata,
        expected.feature_metadata,
        check_names=check_names,
        check_dtype=check_dtype,
        check_exact=check_exact,
    )
    pd.testing.assert_frame_equal(
        actual.sample_metadata,
        expected.sample_metadata,
        check_names=check_names,
        check_dtype=check_dtype,
        check_exact=check_exact,
    )

    if not check_context:
        return

    assert getattr(actual, "value_scale", None) == getattr(
        expected, "value_scale", None
    )

    if expected.observation_mask is None:
        assert actual.observation_mask is None
    else:
        assert actual.observation_mask is not None
        pd.testing.assert_frame_equal(
            actual.observation_mask.to_frame(),
            expected.observation_mask.to_frame(),
            check_names=check_names,
        )

    if expected.sample_manifest is None:
        assert actual.sample_manifest is None
    else:
        assert actual.sample_manifest is not None
        pd.testing.assert_frame_equal(
            actual.sample_manifest.to_frame(),
            expected.sample_manifest.to_frame(),
            check_names=check_names,
        )
