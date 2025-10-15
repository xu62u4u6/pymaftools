"""
ModelPlot class for machine learning model visualization
Inherits from BasePlot to use common plotting functionality
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from statannotations.Annotator import Annotator
from ..plot.BasePlot import BasePlot
from ..core.PivotTable import PivotTable


class ModelPlot(BasePlot):
    """
    Plotting class for machine learning models
    Inherits from BasePlot to use save method and legend management
    """

    def __init__(self):
        super().__init__()

    def plot_metric_comparison_with_annotation(self, data,
                                               metrics=['acc', 'f1', 'auc'],
                                               group_col='model',
                                               order=["SNV", "CNV-gene",
                                                      "CNV-cluster", "STACK"],
                                               palette="Set2",
                                               test="Mann-Whitney",
                                               alpha=0.8,
                                               fontsize=14,
                                               title_fontsize=None,
                                               label_fontsize=None,
                                               tick_fontsize=None,
                                               figsize=None,
                                               title_prefix=None,
                                               rotation=45,
                                               save_path=None,
                                               **save_kwargs):
        """
        Plot metric comparison with statistical annotation

        Args:
            data: DataFrame with model metrics
            metrics: List of metrics to plot
            group_col: Column name for grouping
            order: Order of groups
            palette: Color palette
            test: Statistical test method
            alpha: Transparency level
            fontsize: Base font size (used as default for other sizes if not specified)
            title_fontsize: Font size for titles (default: fontsize + 2)
            label_fontsize: Font size for axis labels (default: fontsize)
            tick_fontsize: Font size for tick labels (default: fontsize - 2)
            figsize: Figure size
            title_prefix: Title prefix (optional, set to None to disable titles)
            rotation: Rotation angle for x-axis tick labels in degrees (default: 45)
            save_path: Path to save figure (optional)
            **save_kwargs: Additional arguments for save method
        """
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
            gb = data.groupby(group_col)
            group_pairs = list(combinations(order, 2))

            sns.boxplot(data=data, x=group_col, y=test_col,
                        ax=ax, hue=group_col, palette=palette, order=order)

            # 設定 alpha 半透明
            for patch in ax.patches:
                r, g, b, _ = patch.get_facecolor()
                patch.set_facecolor((r, g, b, alpha))

            # 顯著性標註
            annotator = Annotator(ax=ax, pairs=group_pairs,
                                  data=data, x=group_col, y=test_col, order=order)
            
            # annotator.configure(test=test, text_format='star',
            #                     loc="outside", verbose=0,     )
                                #line_offset=0.5,
                                #line_offset_to_group=0.1,
                                #use_fixed_offset=True)
            annotator.configure(
                test=test,
                text_format='star',
                loc="inside",
                verbose=0,
            )
            annotator.apply_test()
            annotator.annotate(line_offset_to_group=0.1)
            #annotator.apply_and_annotate()

            ax.set_xlabel('', fontsize=fontsize)
            ax.set_ylabel(metric, fontsize=fontsize)
            ax.set_yticks(ax.get_yticks())  # 確保 ticks 不變
            ax.set_yticklabels([
                f"{tick:.2f}" if tick <= 1 else "" 
                for tick in ax.get_yticks()
            ], fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize, rotation=rotation)
            # Set title if title_prefix is provided
            if title_prefix:
                ax.set_title(f"{title_prefix} {metric.upper()}", fontsize=fontsize)

        plt.tight_layout()

        # Use inherited save method if save_path is provided
        if save_path:
            self.save(save_path, **save_kwargs)

        plt.show()
        return self

    def plot_top_feature_importance_heatmap(self,
                                            importance_df,
                                            omic,
                                            title=None,
                                            top_n=10,
                                            cmap="viridis",
                                            figsize=(10, 6),
                                            save_path=None,
                                            xticklabel=False,
                                            **save_kwargs):
        """
        Plot top feature importance heatmap

        Args:
            importance_df: DataFrame with feature importance
            omic: Omics type name
            top_n: Number of top features to show
            cmap: Color map
            figsize: Figure size
            save_path: Path to save figure (optional)
            **save_kwargs: Additional arguments for save method
        """
        table = self._to_importance_table(importance_df, omic)

        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(table.head(top_n), cmap=cmap, ax=ax, xticklabels=xticklabel,)
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

    def plot_rfecv_curve(self,
                         selector,
                         scoring="accuracy",
                         palette="tab10",
                         title=None,
                         figsize=(15, 5),
                         save_path=None,
                         **save_kwargs):

        df = pd.DataFrame(selector.cv_results_)

        # 找出最佳分數點
        best_idx = df["mean_test_score"].idxmax()
        best_n = df.loc[best_idx, "n_features"]
        best_score = df.loc[best_idx, "mean_test_score"]

        # 繪圖
        self.fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(x="n_features",
                     y="mean_test_score",
                     data=df,
                     label="CV Score",
                     color=sns.color_palette(palette)[0],
                     ax=ax)
        ax.fill_between(df["n_features"],
                        df["mean_test_score"] - df["std_test_score"],
                        df["mean_test_score"] + df["std_test_score"],
                        alpha=0.2,
                        color=sns.color_palette(palette)[0],
                        label="±1 Std")

        # 最佳線
        ax.axvline(best_n, color="red", linestyle="--",
                   label=f"Best ({best_n} features)")

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

    def plot_feature_importance_distribution(self, importance_df, model_name,
                                             top_n=20, figsize=(10, 8),
                                             save_path=None, 
                                             **save_kwargs):
        """
        Plot feature importance distribution across CV folds

        Args:
            importance_df: Long format DataFrame with feature importance
            model_name: Name of the model to plot
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure (optional)
            **save_kwargs: Additional arguments for save method
        """
        # Filter data for specific model
        model_data = importance_df[importance_df['model'] == model_name]

        # Get top features by mean importance
        top_features = (model_data.groupby('feature')['importance']
                        .mean()
                        .sort_values(ascending=False)
                        .head(top_n)
                        .index.tolist())

        # Filter for top features
        plot_data = model_data[model_data['feature'].isin(top_features)]

        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)

        # Plot boxplot
        sns.boxplot(data=plot_data, x='importance', y='feature',
                    order=top_features, ax=ax)
        ax.set_title(f"Feature Importance Distribution - {model_name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        plt.tight_layout()
        # Use inherited save method if save_path is provided
        if save_path:
            self.save(save_path, **save_kwargs)

        plt.show()
        return self

    def _to_importance_table(self, all_importance_df, omic):
        """
        Convert importance data to PivotTable format

        Args:
            all_importance_df: DataFrame with feature importance data
            omic: Name of the omic type

        Returns:
            PivotTable: Sorted table with mean importance
        """
        # 選定 omic 資料並 pivot 成 feature × seed matrix
        pivot_df = (
            all_importance_df
            .query(f"model == '{omic}'")
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
