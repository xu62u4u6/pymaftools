"""Each table subclass exposes its own ``.plot`` accessor type.

The accessors share the common plotters (notably ``oncoplot``) via inheritance
from PivotStatsPlot, and add type-specific entry points. This guards the
contract that ``table.plot`` is a per-type namespace, not a single class.
"""

import numpy as np
import pandas as pd
import pytest

from pymaftools import (
    PivotTable,
    MAF,
    CopyNumberVariationTable,
    SmallVariationTable,
    SignatureTable,
    ExpressionTable,
)


def _frame():
    return pd.DataFrame(
        np.random.rand(4, 3),
        index=["g1", "g2", "g3", "g4"],
        columns=["s1", "s2", "s3"],
    )


@pytest.mark.parametrize(
    "table_cls, accessor_name",
    [
        (PivotTable, "PivotStatsPlot"),
        (CopyNumberVariationTable, "CopyNumberVariationTablePlot"),
        (SmallVariationTable, "SmallVariationTablePlot"),
        (SignatureTable, "SignatureTablePlot"),
        (ExpressionTable, "ExpressionTablePlot"),
    ],
)
def test_plot_accessor_is_type_specific_and_shares_oncoplot(table_cls, accessor_name):
    table = table_cls(_frame())
    accessor = table.plot

    # Each subclass gets its own accessor type (a per-type namespace)...
    assert type(accessor).__name__ == accessor_name
    # ...yet the shared oncoplot entry point is reachable on all of them.
    assert hasattr(accessor, "oncoplot")


def test_maf_plot_accessor_is_maf_specific():
    maf = MAF(
        {
            "Hugo_Symbol": ["TP53"],
            "sample_ID": ["S1"],
            "Variant_Classification": ["Missense_Mutation"],
        }
    )

    assert type(maf.plot).__name__ == "MafPlot"
    assert hasattr(maf.plot, "summary")


def test_cnv_accessor_exposes_band_ratio():
    cnv = CopyNumberVariationTable(_frame())
    assert hasattr(cnv.plot, "band_ratio")


def test_plot_cnv_band_ratio_is_deprecated_alias():
    """The old table-level method warns and delegates to .plot.band_ratio()."""
    cnv = CopyNumberVariationTable(_frame())
    with pytest.warns(DeprecationWarning, match="plot.band_ratio"):
        # Missing required feature_metadata columns -> raises after warning fires.
        with pytest.raises(ValueError):
            cnv.plot_cnv_band_ratio(cluster_id="C1")
