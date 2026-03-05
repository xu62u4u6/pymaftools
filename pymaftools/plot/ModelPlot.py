"""
ModelPlot class for machine learning model visualization
Inherits from BasePlot to use common plotting functionality
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from typing import Any
from statannotations.Annotator import Annotator
from ..plot.BasePlot import BasePlot
from ..core.PivotTable import PivotTable


class ModelPlot(BasePlot):
    """
    Plotting class for machine learning models.

    Inherits from BasePlot to use save method and legend management.
    """

    def __init__(self) -> None:
        """Initialize ModelPlot by calling the parent BasePlot constructor."""
        super().__init__()

    def plot_metric_comparison_with_annotation(
        self,
        data: pd.DataFrame,
        metrics: list[str] | None = None,
        group_col: str = "model",
        order: list[str] | None = None,
        palette: str = "Set2",
        test: str = "Mann-Whitney",
        alpha: float = 0.8,
        fontsize: int = 14,
        title_fontsize: int | None = None,
        label_fontsize: int | None = None,
        tick_fontsize: int | None = None,
        figsize: tuple[int, int] | None = None,
        title_prefix: str | None = None,
        rotation: int = 45,
        save_path: str | None = None,
        **save_kwargs: Any,
    ) -> ModelPlot:
        """
        Plot metric comparison with statistical annotation.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with model metrics.
        metrics : list[str] or None, optional
            List of metrics to plot. Defaults to ``['acc', 'f1', 'auc']``.
        group_col : str, optional
            Column name for grouping. Default is ``'model'``.
        order : list[str] or None, optional
            Order of groups. Defaults to
            ``["SNV", "CNV-gene", "CNV-cluster", "STACK"]``.
        palette : str, optional
            Color palette. Default is ``"Set2"``.
        test : str, optional
            Statistical test method. Default is ``"Mann-Whitney"``.
        alpha : float, optional
            Transparency level. Default is ``0.8``.
        fontsize : int, optional
            Base font size (used as default for other sizes if not specified).
            Default is ``14``.
        title_fontsize : int or None, optional
            Font size for titles. Defaults to ``fontsize + 2``.
        label_fontsize : int or None, optional
            Font size for axis labels. Defaults to ``fontsize``.
        tick_fontsize : int or None, optional
            Font size for tick labels. Defaults to ``fontsize - 2``.
        figsize : tuple[int, int] or None, optional
            Figure size. Defaults to ``(6 * len(metrics), 6)``.
        title_prefix : str or None, optional
            Title prefix. Set to ``None`` to disable titles.
        rotation : int, optional
            Rotation angle for x-axis tick labels in degrees. Default is ``45``.
        save_path : str or None, optional
            Path to save figure.
        **save_kwargs : Any
            Additional arguments for the save method.

        Returns
        -------
        ModelPlot
            Self, for method chaining.
        """
        if metrics is None:
            metrics = ["acc", "f1", "auc"]
        if order is None:
            order = ["SNV", "CNV-gene", "CNV-cluster", "STACK"]

        # 設定字體大小層次
        if title_fontsize is None:
            title_fontsize = fontsize + 2
        if label_fontsize is None:
            label_fontsize = fontsize
        if tick_fontsize is None:
            tick_fontsize = fontsize - 2
        if figsize is None:
            figsize = (6 * len(metrics), 6)

        self.fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

        if len(metrics) == 1:
            axes = [axes]  # Make it iterable for single metric

        if order is None:
            order = sorted(data[group_col].unique())

        for i, metric in enumerate(metrics):
            ax = axes[i]
            test_col = metric
            group_pairs = list(combinations(order, 2))

            sns.boxplot(
                data=data,
                x=group_col,
                y=test_col,
                ax=ax,
                hue=group_col,
                palette=palette,
                order=order,
            )

            # 設定 alpha 半透明
            for patch in ax.patches:
                r, g, b, _ = patch.get_facecolor()
                patch.set_facecolor((r, g, b, alpha))

            # 顯著性標註
            annotator = Annotator(
                ax=ax,
                pairs=group_pairs,
                data=data,
                x=group_col,
                y=test_col,
                order=order,
            )

            annotator.configure(
                test=test,
                text_format="star",
                loc="inside",
                verbose=0,
            )
            annotator.apply_test()
            annotator.annotate(line_offset_to_group=0.1)

            ax.set_xlabel("", fontsize=fontsize)
            ax.set_ylabel(metric, fontsize=fontsize)
            ax.set_yticks(ax.get_yticks())  # 確保 ticks 不變
            ax.set_yticklabels(
                [f"{tick:.2f}" if tick <= 1 else "" for tick in ax.get_yticks()],
                fontsize=fontsize,
            )
            ax.tick_params(axis="x", labelsize=fontsize, rotation=rotation)
            # Set title if title_prefix is provided
            if title_prefix:
                ax.set_title(f"{title_prefix} {metric.upper()}", fontsize=fontsize)

        plt.tight_layout()

        # Use inherited save method if save_path is provided
        if save_path:
            self.save(save_path, **save_kwargs)

        plt.show()
        return self

    def plot_top_feature_importance_heatmap(
        self,
        importance_df: pd.DataFrame,
        omic: str,
        title: str | None = None,
        top_n: int = 10,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
        xticklabel: bool = False,
        **save_kwargs: Any,
    ) -> ModelPlot:
        """
        Plot top feature importance heatmap.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature importance data.
        omic : str
            Omics type name.
        title : str or None, optional
            Plot title.
        top_n : int, optional
            Number of top features to show. Default is ``10``.
        cmap : str, optional
            Color map. Default is ``"viridis"``.
        figsize : tuple[int, int], optional
            Figure size. Default is ``(10, 6)``.
        save_path : str or None, optional
            Path to save figure.
        xticklabel : bool, optional
            Whether to show x-axis tick labels. Default is ``False``.
        **save_kwargs : Any
            Additional arguments for the save method.

        Returns
        -------
        ModelPlot
            Self, for method chaining.
        """
        table = self._to_importance_table(importance_df, omic)

        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(
            table.head(top_n),
            cmap=cmap,
            ax=ax,
            xticklabels=xticklabel,
        )
        if title:
            ax.set_title(title)

        ax.set_xlabel("Seed")
        ax.set_ylabel("Feature")

        plt.tight_layout()

        # Use inherited save method if save_path is provided
        if save_path:
            self.save(save_path, **save_kwargs)

        plt.show()
        return self

    def plot_rfecv_curve(
        self,
        selector: Any,
        scoring: str = "accuracy",
        palette: str = "tab10",
        title: str | None = None,
        figsize: tuple[int, int] = (15, 5),
        save_path: str | None = None,
        **save_kwargs: Any,
    ) -> ModelPlot:
        """
        Plot recursive feature elimination with cross-validation curve.

        Parameters
        ----------
        selector : Any
            Fitted RFECV selector object with ``cv_results_`` attribute.
        scoring : str, optional
            Scoring metric name. Default is ``"accuracy"``.
        palette : str, optional
            Color palette. Default is ``"tab10"``.
        title : str or None, optional
            Plot title.
        figsize : tuple[int, int], optional
            Figure size. Default is ``(15, 5)``.
        save_path : str or None, optional
            Path to save figure.
        **save_kwargs : Any
            Additional arguments for the save method.

        Returns
        -------
        ModelPlot
            Self, for method chaining.
        """
        df = pd.DataFrame(selector.cv_results_)

        # 找出最佳分數點
        best_idx = df["mean_test_score"].idxmax()
        best_n = df.loc[best_idx, "n_features"]
        # 繪圖
        self.fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(
            x="n_features",
            y="mean_test_score",
            data=df,
            label="CV Score",
            color=sns.color_palette(palette)[0],
            ax=ax,
        )
        ax.fill_between(
            df["n_features"],
            df["mean_test_score"] - df["std_test_score"],
            df["mean_test_score"] + df["std_test_score"],
            alpha=0.2,
            color=sns.color_palette(palette)[0],
            label="±1 Std",
        )

        # 最佳線
        ax.axvline(
            best_n, color="red", linestyle="--", label=f"Best ({best_n} features)"
        )

        # 標籤與圖例
        ax.set_xlabel("Number of features selected")
        ax.set_ylabel("CV score (accuracy)")
        ax.grid(alpha=0.3)
        ax.legend()

        if title:
            ax.set_title(title)

        if save_path:
            self.save(save_path, **save_kwargs)

        plt.tight_layout()
        plt.show()
        return self

    def plot_feature_importance_distribution(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        top_n: int = 20,
        figsize: tuple[int, int] = (10, 8),
        save_path: str | None = None,
        **save_kwargs: Any,
    ) -> ModelPlot:
        """
        Plot feature importance distribution across CV folds.

        Parameters
        ----------
        importance_df : pd.DataFrame
            Long format DataFrame with feature importance.
        model_name : str
            Name of the model to plot.
        top_n : int, optional
            Number of top features to show. Default is ``20``.
        figsize : tuple[int, int], optional
            Figure size. Default is ``(10, 8)``.
        save_path : str or None, optional
            Path to save figure.
        **save_kwargs : Any
            Additional arguments for the save method.

        Returns
        -------
        ModelPlot
            Self, for method chaining.
        """
        # Filter data for specific model
        model_data = importance_df[importance_df["model"] == model_name]

        # Get top features by mean importance
        top_features = (
            model_data.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        # Filter for top features
        plot_data = model_data[model_data["feature"].isin(top_features)]

        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)

        # Plot boxplot
        sns.boxplot(
            data=plot_data, x="importance", y="feature", order=top_features, ax=ax
        )
        ax.set_title(f"Feature Importance Distribution - {model_name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        plt.tight_layout()
        # Use inherited save method if save_path is provided
        if save_path:
            self.save(save_path, **save_kwargs)

        plt.show()
        return self

    def _to_importance_table(
        self,
        all_importance_df: pd.DataFrame,
        omic: str,
    ) -> PivotTable:
        """
        Convert importance data to PivotTable format.

        Parameters
        ----------
        all_importance_df : pd.DataFrame
            DataFrame with feature importance data.
        omic : str
            Name of the omic type.

        Returns
        -------
        PivotTable
            Sorted table with mean importance.
        """
        # 選定 omic 資料並 pivot 成 feature × seed matrix
        pivot_df = (
            all_importance_df.query(f"model == '{omic}'")
            .groupby(["seed", "feature"])["importance"]
            .mean()
            .unstack("seed")
        )

        # 建立 PivotTable 並計算平均
        table = PivotTable(pivot_df)
        table.feature_metadata["mean"] = table.mean(axis=1)

        # 按平均重要性排序
        sorted_table = table.sort_features(by="mean", ascending=False)
        return sorted_table
