"""Tests for the shared plotting base class."""

import matplotlib.pyplot as plt
import pytest

from pymaftools.plot.BasePlot import BasePlot


@pytest.mark.plot
def test_save_accepts_pathlike(tmp_path):
    """The public save method accepts pathlib paths."""
    plot = BasePlot()
    plot.fig, _ = plt.subplots()
    save_path = tmp_path / "pathlike_plot.png"

    plot.save(save_path)

    assert save_path.is_file()
    assert save_path.stat().st_size > 0
    plot.close()
