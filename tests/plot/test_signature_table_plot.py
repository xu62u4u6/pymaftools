"""Signature exposure plot colour sampling regressions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from pymaftools.core.SignatureTable import SignatureTable


def _signature_table(n_signatures: int) -> SignatureTable:
    return SignatureTable(
        pd.DataFrame(
            np.ones((n_signatures, 2)),
            index=[f"SBS{i + 1}" for i in range(n_signatures)],
            columns=["S1", "S2"],
        )
    )


def test_stacked_bar_samples_continuous_colormap_across_full_range():
    table = _signature_table(20)
    fig, ax = plt.subplots()

    table.plot.stacked_bar(ax=ax, colormap="viridis", legend=False)

    colors = [to_hex(patch.get_facecolor()) for patch in ax.patches]
    assert len(set(colors)) == 20
    assert colors[0] == to_hex(plt.get_cmap("viridis")(0.0))
    assert colors[-1] == to_hex(plt.get_cmap("viridis")(1.0))
    plt.close(fig)


def test_stacked_bar_cycles_small_qualitative_palette_with_warning():
    table = _signature_table(12)
    fig, ax = plt.subplots()

    with pytest.warns(UserWarning, match=r"tab10.*10.*12.*tab20"):
        table.plot.stacked_bar(ax=ax, colormap="tab10", legend=False)

    # Two sample patches per signature: signature 11 cycles to tab10 colour 1.
    colors = [to_hex(patch.get_facecolor()) for patch in ax.patches]
    assert colors[0] == colors[20]
    plt.close(fig)
