"""Common contract and invariance tests across PivotTable and all subclasses.

Modeled after scikit-learn common estimator checks and scverse/anndata
representation-invariance contracts: every PivotTable subclass must satisfy
the same structural, metadata, persistence, and order-invariance guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable
from pymaftools.core.ExpressionTable import ExpressionTable
from pymaftools.core.ObservationMask import ObservationMask
from pymaftools.core.PivotTable import PivotTable
from pymaftools.core.SampleManifest import SampleManifest
from pymaftools.core.SignatureTable import SignatureTable
from pymaftools.core.SmallVariationTable import SmallVariationTable
from tests._helpers import assert_pivot_equal


def _make_table(table_cls: type[PivotTable]) -> PivotTable:
    samples = ["S1", "S2", "S3", "S4"]
    features = ["F1", "F2", "F3"]

    if table_cls is SmallVariationTable:
        data = pd.DataFrame(
            {
                "S1": ["Missense_Mutation", False, "Nonsense_Mutation"],
                "S2": [False, "Frame_Shift_Del", False],
                "S3": ["Missense_Mutation", "Splice_Site", False],
                "S4": [False, False, False],
            },
            index=features,
        )
        table = table_cls(data)
    elif table_cls is CopyNumberVariationTable:
        data = pd.DataFrame(
            {
                "S1": [0, 2, 4],
                "S2": [1, 2, 2],
                "S3": [2, 0, 3],
                "S4": [2, 2, 2],
            },
            index=features,
        )
        table = table_cls(data, value_scale="absolute_cn")
    elif table_cls is ExpressionTable:
        data = pd.DataFrame(
            {
                "S1": [10.5, 0.0, 42.1],
                "S2": [5.2, 3.1, 18.0],
                "S3": [0.0, 12.4, 9.5],
                "S4": [7.8, 1.5, 0.0],
            },
            index=features,
        )
        table = table_cls(data)
    elif table_cls is SignatureTable:
        data = pd.DataFrame(
            {
                "S1": [0.6, 0.3, 0.1],
                "S2": [0.2, 0.5, 0.3],
                "S3": [0.1, 0.1, 0.8],
                "S4": [0.4, 0.4, 0.2],
            },
            index=["SBS1", "SBS2", "SBS4"],
        )
        table = table_cls(data)
    else:
        data = pd.DataFrame(
            {
                "S1": [True, False, True],
                "S2": [False, True, False],
                "S3": [True, True, False],
                "S4": [False, False, False],
            },
            index=features,
        )
        table = table_cls(data)

    table.sample_metadata = pd.DataFrame(
        {"subtype": ["LUAD", "LUSC", "LUAD", "LUSC"], "age": [61, 70, 55, 64]},
        index=table.columns,
    )
    table.feature_metadata = pd.DataFrame(
        {"chrom": ["17", "12", "7"], "weight": [1.0, 2.0, 3.0]},
        index=table.index,
    )

    mask_df = pd.DataFrame(True, index=table.index, columns=table.columns)
    mask_df.iloc[2, 3] = False
    manifest = SampleManifest(
        pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3", "P4"],
                "eligible": [True, True, True, True],
                "exclusion_reason": [pd.NA, pd.NA, pd.NA, pd.NA],
                "assay": ["WES"] * 4,
            },
            index=samples,
        )
    )
    return table.with_scientific_context(
        observation_mask=ObservationMask(mask_df),
        sample_manifest=manifest,
    )


ALL_TABLE_CLASSES = [
    PivotTable,
    SmallVariationTable,
    CopyNumberVariationTable,
    ExpressionTable,
    SignatureTable,
]


@pytest.mark.parametrize("table_cls", ALL_TABLE_CLASSES, ids=lambda cls: cls.__name__)
def test_copy_preserves_subclass_context_and_is_independent(table_cls):
    original = _make_table(table_cls)
    copied = original.copy()

    assert_pivot_equal(copied, original)
    copied.sample_metadata.loc["S1", "subtype"] = "MUTATED"
    assert original.sample_metadata.loc["S1", "subtype"] == "LUAD"


@pytest.mark.parametrize("table_cls", ALL_TABLE_CLASSES, ids=lambda cls: cls.__name__)
def test_subset_and_reindex_preserve_subclass_and_align_context(table_cls):
    table = _make_table(table_cls)
    keep_features = [table.index[2], table.index[0]]
    keep_samples = ["S3", "S1"]

    sub = table.subset(features=keep_features, samples=keep_samples)
    assert type(sub) is table_cls
    assert list(sub.index) == keep_features
    assert list(sub.columns) == keep_samples
    assert list(sub.feature_metadata.index) == keep_features
    assert list(sub.sample_metadata.index) == keep_samples
    assert sub.observation_mask is not None
    assert list(sub.observation_mask.to_frame().index) == keep_features
    assert list(sub.observation_mask.to_frame().columns) == keep_samples
    assert sub.sample_manifest is not None
    assert getattr(sub, "value_scale", None) == getattr(table, "value_scale", None)

    reordered = table.reindex(
        index=list(reversed(table.index)),
        columns=list(reversed(table.columns)),
    )
    assert type(reordered) is table_cls
    assert list(reordered.feature_metadata.index) == list(reordered.index)
    assert list(reordered.sample_metadata.index) == list(reordered.columns)
    assert reordered.observation_mask is not None
    assert reordered.observation_mask.to_frame().loc[table.index[2], "S4"] == False  # noqa: E712
    assert getattr(reordered, "value_scale", None) == getattr(
        table, "value_scale", None
    )


@pytest.mark.parametrize("table_cls", ALL_TABLE_CLASSES, ids=lambda cls: cls.__name__)
def test_add_metadata_is_order_invariant_and_preserves_subclass(table_cls):
    table = _make_table(table_cls)
    shuffled_sample_meta = pd.DataFrame(
        {"stage": ["III", "I", "IV", "II"]},
        index=["S3", "S1", "S4", "S2"],
    )
    shuffled_feature_meta = pd.DataFrame(
        {"pathway": ["RTK", "TP53", "PI3K"]},
        index=[table.index[2], table.index[0], table.index[1]],
    )

    updated = table.add_sample_metadata(shuffled_sample_meta).add_feature_metadata(
        shuffled_feature_meta
    )

    assert type(updated) is table_cls
    assert updated.sample_metadata["stage"].to_dict() == {
        "S1": "I",
        "S2": "II",
        "S3": "III",
        "S4": "IV",
    }
    assert updated.feature_metadata["pathway"].to_dict() == {
        table.index[0]: "TP53",
        table.index[1]: "PI3K",
        table.index[2]: "RTK",
    }
    assert updated.observation_mask is not None
    assert updated.sample_manifest is not None
    assert getattr(updated, "value_scale", None) == getattr(table, "value_scale", None)


@pytest.mark.parametrize("table_cls", ALL_TABLE_CLASSES, ids=lambda cls: cls.__name__)
def test_hdf5_round_trip_preserves_full_container(table_cls, tmp_path):
    table = _make_table(table_cls)
    h5_path = tmp_path / f"{table_cls.__name__}.h5"

    table.to_h5(h5_path)
    reloaded = PivotTable.read_h5(h5_path)

    assert_pivot_equal(reloaded, table)


@pytest.mark.parametrize(
    "table_cls",
    [PivotTable, SmallVariationTable, CopyNumberVariationTable],
    ids=lambda cls: cls.__name__,
)
def test_frequency_and_similarity_are_sample_order_invariant(table_cls):
    table = _make_table(table_cls)
    permuted = table.reindex(columns=["S4", "S2", "S1", "S3"])

    freq_orig = table.calculate_feature_frequency()
    freq_perm = permuted.calculate_feature_frequency()
    pd.testing.assert_series_equal(freq_orig, freq_perm)

    binary_orig = table.to_binary_table()
    binary_perm = permuted.to_binary_table()
    sim_orig = binary_orig.compute_similarity(method="cosine")
    sim_perm = binary_perm.compute_similarity(method="cosine").loc[
        sim_orig.index, sim_orig.columns
    ]
    np.testing.assert_allclose(sim_orig.values, sim_perm.values, atol=1e-12)
