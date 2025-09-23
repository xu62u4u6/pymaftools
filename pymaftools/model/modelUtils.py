import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from statannotations.Annotator import Annotator
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from ..core.PivotTable import PivotTable

def get_importance(model):
    """擷取模型的 feature importance，支援 sklearn 和 OmicsStackingModel 類型"""
    if hasattr(model, "feature_importances_"):
        try:
            return pd.Series(model.feature_importances_, index=model.feature_names_in_)
        except AttributeError:
            return pd.Series(model.feature_importances_)
        
    elif hasattr(model, "get_omics_feature_importance") and hasattr(model, "omics_dict"):
        importances = []
        for omics_name in model.omics_dict.keys():
            imp = model.get_omics_feature_importance(omics_name)
            importances.append(imp)
        return pd.concat(importances)
    
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def evaluate_model(model, X_test, y_test):
    """評估單個模型並返回指標字典"""
    y_pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    
    # 計算 AUC
    if prob.shape[1] == 2:
        auc = roc_auc_score(y_test, prob[:, 1])
    else:
        auc = roc_auc_score(y_test, prob, multi_class="ovr")
    
    return {
        "acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "auc": auc
    }

def cross_validate_importance(
    X, y, model_func, model_name,
    n_seeds=5, n_splits=5, random_state_base=0, verbose=True,
    evaluate_func=None
):
    """
    對指定模型與資料進行多次交叉驗證，回傳：
    - 特徵重要性的 long format DataFrame（importance_df）
    - 每個 fold 的分類表現（metric_df）
    """
    
    importance_records = []
    metrics_records = []

    for seed in tqdm(range(n_seeds), desc="CV seeds", disable=not verbose):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + random_state_base)


        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = model_func(seed)
            model.fit(X_train, y_train)

            # 評估模型效能
            if evaluate_func is not None:
                metrics = {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_id
                }
                metrics.update(evaluate_func(model, X_test, y_test))
                metrics_records.append(metrics)

            # 收集特徵重要性（Long Format）
            imp = get_importance(model)
            for feature, importance in imp.items():
                importance_records.append({
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_id,
                    "feature": feature,
                    "importance": importance
                })

    # 轉換為 DataFrame
    importance_df = pd.DataFrame(importance_records)
    metric_df = pd.DataFrame(metrics_records) if metrics_records else None
    
    return importance_df, metric_df

def plot_metric_comparison_with_annotation(data, 
                                           metrics=['acc', 'f1', 'auc'], 
                                           group_col='model',
                                           order=["SNV", "CNV-gene", "CNV-cluster", "STACK"], 
                                           palette="Set2", 
                                           test="Mann-Whitney", 
                                           alpha=0.8,
                                           fontsize=14, 
                                           figsize=None,
                                           title_prefix=None,
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
        fontsize: Font size
        figsize: Figure size
        title_prefix: Title prefix (optional, set to None to disable titles)
        save_path: Path to save figure (optional)
        **save_kwargs: Additional arguments for save method
    
    Returns:
        ModelPlot instance
    """
    from ..plot.ModelPlot import ModelPlot
    
    plotter = ModelPlot()
    return plotter.plot_metric_comparison_with_annotation(
        data=data,
        metrics=metrics,
        group_col=group_col,
        order=order,
        palette=palette,
        test=test,
        alpha=alpha,
        fontsize=fontsize,
        figsize=figsize,
        title_prefix=title_prefix,
        save_path=save_path,
        **save_kwargs
    )


def to_importance_table(all_importance_df, omic):

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

def plot_top_feature_importance_heatmap(mean_importance_df, omic, top_n=20, 
                                        cmap="viridis", figsize=(10, 6),
                                        title=None,
                                        save_path=None, **save_kwargs):
    """
    Plot heatmap of top feature importance
    
    Args:
        mean_importance_df: DataFrame with feature importance data
        omic: Name of the omic type
        top_n: Number of top features to show
        cmap: Colormap for heatmap
        figsize: Figure size
        title: Title for the plot (optional, set to None to disable title)
        save_path: Path to save figure (optional)
        **save_kwargs: Additional arguments for save method
    
    Returns:
        ModelPlot instance
    """
    from ..plot.ModelPlot import ModelPlot
    
    plotter = ModelPlot()
    return plotter.plot_top_feature_importance_heatmap(
        importance_df=mean_importance_df,
        omic=omic,
        top_n=top_n,
        cmap=cmap,
        figsize=figsize,
        title=title,
        save_path=save_path,
        **save_kwargs
    )

def run_rfecv_feature_selection(pivot: PivotTable,
                                label_col: str = "subtype",
                                estimator=None,
                                step: int = 10,
                                scoring: str = "accuracy",
                                min_features_to_select: int = 10,
                                plot: bool = True,
                                random_state: int = 42,
                                title=None,
                                save_path=None,
                                **save_kwargs):
    """
    執行 RFECV 特徵選擇，適用於 PivotTable (SNV or CNV)

    Parameters
    ----------
    pivot : PivotTable
        具有 feature × sample 結構的表格
    label_col : str
        在 sample_metadata 中的標籤欄位名稱
    estimator : sklearn estimator
        要使用的模型（預設為 RandomForest）
    step : int
        每輪移除特徵的數量
    scoring : str
        評估指標（如 'accuracy', 'f1_macro'）
    min_features_to_select : int
        最少保留幾個特徵
    plot : bool
        是否繪製 performance 曲線
    random_state : int
        隨機種子
    title : str, optional
        圖表標題（設定為 None 則不顯示標題）
    save_path : str, optional
        圖片儲存路徑
    **save_kwargs : dict
        儲存圖片的額外參數
    
    Returns
    -------
    selected_features : list
        被選中的特徵名稱（pivot.index）
    selector : RFECV
        訓練好的 RFECV 物件
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)

    X = pivot.T.values
    y = np.array(pivot.sample_metadata[label_col].values)  # 轉換為 numpy array 以解決類型問題

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=StratifiedKFold(5),
        scoring=scoring,
        verbose=0,
        min_features_to_select=min_features_to_select
    )

    selector.fit(X, y)

    print(f"Optimal number of features: {selector.n_features_}")
    selected_features = pivot.index[selector.support_].tolist()

    if plot:
        from ..plot.ModelPlot import ModelPlot
        plotter = ModelPlot()
        plotter.plot_rfecv_curve(
            selector=selector,
            title=title,
            scoring=scoring,
            save_path=save_path,
            **save_kwargs
        )

    return selected_features, selector


def run_model_evaluation(
    model_configs,
    y,
    n_seeds=100,
    n_splits=5,
    evaluate_func=None,
    verbose=True
):
    """
    執行多模型交叉驗證與重要性分析，回傳 result dict 與合併後的資料表。
    
    Returns
    -------
    result_dict : dict
        每個模型名稱對應的 {'importance': ..., 'metrics': ...}
    all_importance_df : pd.DataFrame
        合併後的長格式特徵重要性資料
    all_metrics_df : pd.DataFrame
        合併後的長格式分類效能資料
    """
    result = {}

    for model_config in model_configs:
        model_name = model_config["name"]
        if verbose:
            print(f"Processing model: {model_name}")
        model_func = model_config["model_func"]
        X = model_config["X"]

        importance_df, metric_df = cross_validate_importance(
            X=X,
            y=y,
            model_name=model_name,
            model_func=model_func,
            n_seeds=n_seeds,
            n_splits=n_splits,
            evaluate_func=evaluate_func,
            verbose=verbose
        )

        result[model_name] = {
            "importance": importance_df,
            "metrics": metric_df
        }

    # 合併所有模型的結果
    all_importance_df = pd.concat(
        [res["importance"] for res in result.values()],
        ignore_index=True
    )

    all_metrics_df = pd.concat(
        [res["metrics"] for res in result.values()],
        ignore_index=True
    )

    return result, all_importance_df, all_metrics_df


# example

# model_configs = [
#     {
#         "name": "SNV",
#         "model_func": lambda random_state: RandomForestClassifier(n_estimators=100, random_state=random_state),
#         "X": snv.T
#     },
#     {
#         "name": "CNV-cluster", 
#         "model_func": lambda random_state: RandomForestClassifier(n_estimators=100, random_state=random_state),
#         "X": cluster_table.T
#     },
#     {
#         "name": "CNV-arm", 
#         "model_func": lambda random_state: RandomForestClassifier(n_estimators=100, random_state=random_state),
#         "X": cnv_arm_table.T
#     },
#     {
#         "name": "CNV-gene", 
#         "model_func": lambda random_state: RandomForestClassifier(n_estimators=100, random_state=random_state),
#         "X": cnv_gene_table.T
#     },
#     {
#         "name": "CNV-cytoband", 
#         "model_func": lambda random_state: RandomForestClassifier(n_estimators=100, random_state=random_state),
#         "X": cnv_cytoband_table.T
#     },
#     {
#         "name": "STACK",
#         "model_func": lambda random_state: ASCStackingModel(
#                         omics_dict={"snv": snv, "cnv": cluster_table},
#                         class_order=["LUAD", "LUSC"],
#                         random_state=random_state
#                     ),
#         "X": pd.concat([snv, cluster_table], axis=0).T
#     }
# ]

# result, all_importance_df, all_metrics_df = run_model_evaluation(
#     model_configs=model_configs,
#     y=cohort_no_ASC.sample_metadata.subtype,
#     n_seeds=100,
#     n_splits=5,
#     evaluate_func=evaluate_model
# )

# # 儲存成 CSV
# all_importance_df.to_csv("data/model/all_model_importance_100seed_5fold_add_cytoband.csv", index=False)
# all_metrics_df.to_csv("data/model/all_model_metrics_100seed_5fold_add_cytoband.csv", index=False)
	