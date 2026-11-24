"""Tests for PivotStatsPlot, ModelPlot, and BasePlot saving/validation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pymaftools.plot.BasePlot import BasePlot
from pymaftools.plot.ModelPlot import ModelPlot
from pymaftools.plot.PivotStatsPlot import PivotStatsPlot


class TestPivotStatsPlot:
    """Behavioral tests for PivotStatsPlot boxplot and PCA methods."""

    @pytest.mark.plot
    def test_boxplot_draws_boxes_custom_labels_and_annotations(
        self, sample_pivot_table
    ):
        sample_pivot_table.sample_metadata["TMB"] = [2.5, 3.1, 1.8, 2.9]
        plot = PivotStatsPlot(sample_pivot_table)

        ax = plot.boxplot(
            test_col="TMB",
            group_col="subtype",
            test="Mann-Whitney",
            xlabel="Cancer Subtype",
            ylabel="Tumor Mutation Burden",
            title="TMB Comparison",
        )

        assert ax.get_xlabel() == "Cancer Subtype"
        assert ax.get_ylabel() == "Tumor Mutation Burden"
        assert ax.get_title() == "TMB Comparison"
        assert len(ax.patches) >= 1 or len(ax.lines) >= 1
        assert len(ax.texts) >= 1  # statannotations significance label
        plt.close("all")

    @pytest.mark.plot
    def test_pca_categorical_and_numeric_render_artists(self, sample_pivot_table):
        sample_pivot_table.sample_metadata["TMB"] = [2.5, 3.1, 1.8, 2.9]
        expected_df, expected_var, _ = sample_pivot_table.pca(to_binary=True)

        plot_cat = PivotStatsPlot(sample_pivot_table)
        pca_df, explained_var, pca_obj = plot_cat.pca(
            color_col="subtype",
            figsize=(8, 4),
            to_binary=True,
        )
        pd.testing.assert_frame_equal(pca_df[["PC1", "PC2"]], expected_df)
        np.testing.assert_allclose(explained_var, expected_var)
        assert pca_obj is not None
        assert any(ax.collections for ax in plot_cat.fig.axes)
        plt.close("all")

        plot_num = PivotStatsPlot(sample_pivot_table)
        num_df, _, _ = plot_num.pca(
            color_col="TMB",
            is_numeric=True,
            palette="viridis",
            to_binary=True,
        )
        pd.testing.assert_frame_equal(num_df[["PC1", "PC2"]], expected_df)
        # Numeric color encoding attaches a colorbar (extra axes beyond main + legend)
        assert len(plot_num.fig.axes) >= 2
        assert any(
            c.get_array() is not None
            for ax in plot_num.fig.axes
            for c in ax.collections
        )
        plt.close("all")


class TestModelPlot:
    """Behavioral tests for ModelPlot metric comparisons."""

    @pytest.mark.plot
    def test_metric_comparison_draws_axes_rotation_and_saves(
        self, sample_model_metrics, tmp_path
    ):
        plot = ModelPlot()
        save_path = tmp_path / "metrics.png"

        plot.plot_metric_comparison_with_annotation(
            data=sample_model_metrics,
            metrics=["acc", "f1"],
            group_col="model",
            test="Mann-Whitney",
            rotation=30,
            fontsize=10,
            save_path=str(save_path),
            dpi=120,
        )

        assert len(plot.fig.axes) == 2
        tick_rotations = {
            tick.get_rotation() for tick in plot.fig.axes[0].get_xticklabels()
        }
        assert tick_rotations == {30}
        assert save_path.is_file() and save_path.stat().st_size > 0
        plt.close("all")


class TestPlotSaving:
    """Tests for BasePlot.save across formats, Path objects, and DPI."""

    @pytest.mark.plot
    @pytest.mark.parametrize("ext", ["png", "pdf", "svg"])
    def test_save_supports_formats_and_pathlike(self, tmp_path, ext):
        plot = BasePlot()
        plot.fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        save_path = tmp_path / f"plot.{ext}"

        plot.save(save_path)

        assert save_path.is_file()
        assert save_path.stat().st_size > 0
        plot.close()

    @pytest.mark.plot
    def test_higher_dpi_produces_larger_png(self, tmp_path):
        plot = BasePlot()
        plot.fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(np.arange(20), np.arange(20))

        low_dpi = tmp_path / "low.png"
        high_dpi = tmp_path / "high.png"
        plot.save(low_dpi, dpi=72)
        plot.save(high_dpi, dpi=300)
        plot.close()

        assert high_dpi.stat().st_size > low_dpi.stat().st_size

    @pytest.mark.plot
    def test_save_extensionless_path_inside_dotted_directory(self, tmp_path):
        plot = BasePlot()
        plot.fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [1, 0])
        dotted_dir = tmp_path / "dir.v1"
        dotted_dir.mkdir()
        save_path = dotted_dir / "figure"

        plot.save(save_path)

        assert (dotted_dir / "figure.png").is_file()
        plot.close()


class TestPlotValidation:
    """Error handling for invalid plotting inputs."""

    def test_invalid_column_names(self, sample_pivot_table):
        plot = PivotStatsPlot(sample_pivot_table)

        with pytest.raises(ValueError, match="not found in data"):
            plot.boxplot(test_col="nonexistent_column", group_col="subtype")

        with pytest.raises(ValueError, match="not found in sample_metadata"):
            plot.pca(color_col="nonexistent_column")


@pytest.mark.plot
def test_metric_comparison_default_order_uses_data_groups():
    # The default order used to be the ASC project's omics names.
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "model": ["RNA"] * 8 + ["Methylation"] * 8,
            "acc": rng.normal(0.7, 0.05, 16),
        }
    )
    plot = ModelPlot()
    plot.plot_metric_comparison_with_annotation(data=data, metrics=["acc"])

    labels = [tick.get_text() for tick in plot.fig.axes[0].get_xticklabels()]
    plt.close("all")
    assert labels == ["Methylation", "RNA"]
