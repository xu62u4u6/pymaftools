from __future__ import annotations

import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_rand_score, classification_report
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from typing import Literal

from .PivotTable import PivotTable


def table_to_distance(table):
    """將表格轉換為距離矩陣"""
    similarity = table.T.compute_similarity(method="cosine")
    distance = 1 - similarity.values
    return distance

def k_fold_clustering_evaluation(table: PivotTable,
                                 min_clusters: int = 2,
                                 max_clusters: int = 50,
                                 metric: Literal['cosine', 'hamming', 'jaccard'] = "cosine",
                                 random_state: int = 42, 
                                 group_col="subtype"):
    """
    使用K-fold交叉驗證評估基因聚類的最佳聚類數

    Parameters
    ----------
    table : PivotTable
        基因表達或CNV數據表
    max_clusters : int
        最大聚類數
    metric : str
        相似性度量方法
    random_state : int
        隨機種子

    Returns
    -------
    pd.DataFrame
        包含每個fold和聚類數的silhouette分數
    """
    group = table.sample_metadata[group_col].values
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    if not isinstance(table, PivotTable):
        raise ValueError("Input table must be a PivotTable instance.")
    if min_clusters < 2 or max_clusters < min_clusters:
        raise ValueError(
            "min_clusters must be at least 2 and max_clusters must be greater than or equal to min_clusters.")
    all_results = []
    cluster_label_dict = {}
    # only use (k-1)/k train part for clustering
    for fold, (train_idx, _) in enumerate(kf.split(table.T.values, np.array(group))):
        print(f"Processing fold {fold + 1}/5")

        # 獲取訓練樣本
        sample_train = table.sample_metadata.iloc[train_idx].index
        table_train = table.subset(samples=sample_train)
        similarity_matrix = table_train.T.compute_similarity(method=metric)
        distance_matrix_train = 1 - similarity_matrix # dist = 1 - similarity
        
        # test different number of clusters
        for k in tqdm(range(min_clusters, max_clusters + 1), desc=f"Fold {fold + 1}"):
            model = AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed',
                linkage='average'
            )
            labels = model.fit_predict(distance_matrix_train)
            if k not in cluster_label_dict:
                cluster_label_dict[k] = {}
            cluster_label_dict[k][fold+1] = labels

            # 計算silhouette分數
            score = silhouette_score(table_train, labels, metric=metric)

            all_results.append({
                'fold': fold + 1,
                'n_clusters': k,
                'silhouette_score': score
            })

    return pd.DataFrame(all_results), cluster_label_dict

def align_clusters(ref_labels, target_labels, n_clusters):
    cm = confusion_matrix(ref_labels, target_labels, labels=range(n_clusters))
    cost_matrix = -cm  # Hungarian algorithm finds *minimum* cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = dict(zip(col_ind, row_ind))
    aligned = np.vectorize(lambda x: mapping.get(x, x))(target_labels)
    return aligned


def align_cluster_label_dict(cluster_label_dict):
    """
    將 cluster_label_dict 中的聚類結果進行對齊（以 fold 1 為參考）。

    Parameters
    ----------
    cluster_label_dict : dict
        每個聚類數 k 對應的 fold → label dict。
        結構為 {k: {fold: labels}}

    Returns
    -------
    dict
        每個 k 對應的已對齊的 DataFrame（樣本 × fold）
    """
    aligned_fold_df_dict = {}

    for k, fold_dict in cluster_label_dict.items():
        fold_df = pd.DataFrame(fold_dict)  # 樣本為 index，欄位為 fold
        ref_labels = fold_df.iloc[:, 0].values  # fold_1 作為 reference
        aligned_df = pd.DataFrame(index=fold_df.index)

        for col in fold_df.columns:
            if col == fold_df.columns[0]:
                aligned_df[col] = fold_df[col]
            else:
                aligned_df[col] = align_clusters(
                    ref_labels, fold_df[col].values, n_clusters=int(k))

        aligned_fold_df_dict[k] = aligned_df

    return aligned_fold_df_dict


def convert_ndarray_to_list(obj):
    """將 dict 結構中所有 numpy.ndarray 遞迴轉換為 list"""
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj


def calculate_ari_matrix(aligned_cluster_label_dict, k):
    df = aligned_cluster_label_dict[k]  # 使用指定的 fold
    folds = df.columns
    n = len(folds)
    ari_matrix = pd.DataFrame(index=folds, columns=folds, dtype=float)

    for i in range(n):
        for j in range(n):
            ari_matrix.iloc[i, j] = adjusted_rand_score(
                df.iloc[:, i], df.iloc[:, j])

    return ari_matrix


def plot_ari_matrix(aligned_cluster_label_dict, k):
    ari_matrix = calculate_ari_matrix(aligned_cluster_label_dict, k)
    n = len(ari_matrix)

    def mean_off_diagonal_ari(ari_matrix: pd.DataFrame) -> float:
        """計算 ARI 矩陣中非對角線元素的平均值"""
        n = len(ari_matrix)
        mask = ~np.eye(n, dtype=bool)
        return ari_matrix.values[mask].mean()

    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(ari_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(
        f"Cluster {k} ARI between {n} folds, mean={mean_off_diagonal_ari(ari_matrix):.2f}")
    plt.show()


def run_random_forest_cv(X, y, feature_names, n_splits=5, random_state=42, n_estimators=100):
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_scores = []
    all_importances = []

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = RandomForestClassifier(random_state=random_state, 
                                       
                                       n_estimators=n_estimators)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_true = y[test_idx]

        acc = model.score(X[test_idx], y[test_idx])
        cv_scores.append(acc)
        all_importances.append(model.feature_importances_)

        print(f"\n[Fold {i+1}] Overall Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, digits=4))
        
    importance_df = pd.DataFrame(all_importances, 
                                 columns=feature_names,
                                 index=[f"importance_{i+1}" for i in range(n_splits)]).T
    importance_df["mean_importance"] = importance_df.mean(axis=1)
    return model, cv_scores, importance_df

def run_random_forest_multiple_seeds(X, y, feature_names, seeds=range(5), n_estimators=100):
    """
    Run RandomForestClassifier multiple times with different seeds using the entire dataset each time.
    
    Args:
        X (ndarray): Feature matrix (samples x features)
        y (array-like): Target labels
        feature_names (list): List of feature names
        seeds (iterable): List or range of random seeds
        n_estimators (int): Number of trees in the forest
    
    Returns:
        models (list): Trained RandomForestClassifier models
        importance_df (DataFrame): Feature importances per run and their mean
    """
    models = []
    all_importances = []

    for seed in seeds:
        model = RandomForestClassifier(random_state=seed, n_estimators=n_estimators)
        model.fit(X, y)
        models.append(model)
        all_importances.append(model.feature_importances_)
        print(f"[Seed {seed}] Training done.")

    importance_df = pd.DataFrame(all_importances, 
                                 columns=feature_names,
                                 index=[f"importance_{s}" for s in seeds]).T
    importance_df["mean_importance"] = importance_df.mean(axis=1)
    
    return models, importance_df

    
def plot_cluster_feature_importance_boxplot(table: pd.DataFrame, importance_cols: "list[str]", top_n: int = 20):
    """
    Draw a bar+box plot of top N cluster feature importances.

    Parameters:
        table (pd.DataFrame): DataFrame containing cluster info with importance scores.
        importance_cols (list[str]): Columns for fold-wise importance scores.
        top_n (int): Number of top clusters to show.
    """
    # 取 top_n
    top_data = table.sort_values(by="mean_importance", ascending=False).head(top_n).copy()
    positions = np.arange(top_n)

    # 整理資料
    box_values = [top_data.iloc[i][importance_cols].values.tolist() for i in range(top_n)]
    bar_values = top_data["mean_importance"].values.tolist()

    # 繪圖
    plt.figure(figsize=(14, 7))

    # Boxplot（透明）
    plt.boxplot(
        box_values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor='none', color='gray'),
        medianprops=dict(color='red'),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray'),
        flierprops=dict(marker='o', color='gray', alpha=0.3)
    )

    # Barplot（青色底）
    plt.bar(positions, bar_values, width=0.6, color='skyblue', label='Mean Importance')

    # X軸標籤：cluster index + chr_arm + gene_count
    xlabels = [
        f"\n\nC{i}\n arm: {top_data.loc[i, 'unique_chr_arm']}\n n={top_data.loc[i, 'gene_count']}"
        for i in top_data.index
    ]
    plt.xticks(positions, xlabels, rotation=45, ha='right')

    plt.ylabel("Feature Importance")
    plt.title("Top Cluster Feature Importances\n(Bar = Mean, Box = 5-Fold Distribution)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cluster_feature_importance(table: pd.DataFrame, importance_cols: "list[str]", top_n: int = 20):
    """
    Plot top N cluster feature importances as bar (mean) + scatter (per-fold) using seaborn.
    """
    # 取前 top_n 並加 cluster label
    top_data = table.nlargest(top_n, "mean_importance").copy()
    top_data["Cluster"] = [f"C{i}" for i in top_data.index]

    # 長格式轉換
    long_df = top_data[["Cluster", "unique_chr_arm", "gene_count"] + importance_cols].melt(
        id_vars=["Cluster", "unique_chr_arm", "gene_count"],
        value_vars=importance_cols,
        var_name="Fold",
        value_name="Importance"
    )

    # 繪圖
    plt.figure(figsize=(14, 7))

    sns.barplot(data=top_data, x="Cluster", y="mean_importance", color="skyblue", zorder=0)
    sns.stripplot(data=long_df, x="Cluster", y="Importance", color="gray", alpha=0.6, jitter=0.1, zorder=1)

    # 改 x 軸標籤：cluster + arm + n
    new_labels = [
        f"\n\n{row['Cluster']}\narm: {row['unique_chr_arm']}\nn={row['gene_count']}"
        for _, row in top_data.iterrows()
    ]
    plt.xticks(ticks=np.arange(top_n), labels=new_labels, rotation=45, ha='right')

    plt.ylabel("Feature Importance")
    plt.title("Top Cluster Feature Importances\n(Dots = 5-Fold, Bar = Mean)")
    plt.tight_layout()
    plt.show()

def run_feature_clustering(table, result_path, max_clusters=200):
    metric = 'cosine'
    similarity = table.T.compute_similarity(method=metric)
    distance = 1 - similarity
    distance_matrix = distance.values
    X = table.values 

    results = []
    for k in tqdm(range(2, max_clusters+1)):
        print(f"Clustering with {k} clusters")
        model = AgglomerativeClustering(n_clusters=k, 
                                        metric='precomputed', 
                                        linkage='average')
        labels = model.fit_predict(distance_matrix)
        score = silhouette_score(X, labels, metric=metric)  # 原始矩陣
        results.append({'n_clusters': k, 'silhouette': score})
        print(f"score: {score}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(result_path, index=False)
    return results_df

def plot_clustering_metrics_and_find_best_k(
        metric_df, 
        filename, 
        title=None,
        target_col="mean_silhouette", 
        dpi=300, 
        bbox_inches='tight', 
        transparent=True, 
        format=None, 
        **kwargs):

    fig, ax = plt.subplots(figsize=(12, 6))

    fold_columns = [f"fold{fold}_silhouette" for fold in range(1, 6)]
    fold_df = metric_df.loc[:, fold_columns]
    fold_means = fold_df.mean(axis=1)
    metric_df["mean_silhouette"] = fold_means
    fold_stds = fold_df.std(axis=1)
    
    ax.plot(metric_df.index, fold_means, label="Mean silhouette (5-fold)", color="#1f77b4", linestyle='-')
    ax.errorbar(metric_df.index, fold_means, yerr=fold_stds, fmt='none', capsize=2, capthick=1, color='#bbbbbb', alpha=0.6, label="5-fold mean ± std")

    ax.plot(metric_df.index, metric_df["mean_ari_5_fold"], label="Mean ARI (5-fold)", color="#2ca02c", linestyle='-.')

    best_k = metric_df[target_col].idxmax()
    ax.axvline(best_k, color='red', linestyle=':', alpha=0.6, label=f'Best k={best_k}')

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Score")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # save figure
    format = filename.split('.')[-1].lower()
    pil_kwargs = {"compression": "tiff_lzw"} if format == "tiff" else {}
    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        format=format,
        pil_kwargs=pil_kwargs,
        **kwargs
    )
    plt.close(fig)  # optional: 釋放記憶體

    return best_k


def gpt_known_genes_summary(client, genes: "list[str]", arm: str, cancer_type="肺癌"):
    prompt = "\n".join([
        f"以下是一些位於{arm}的人類CNV基因列表：",
        ", ".join(genes),
        "",
        f"請你以癌症文獻中常出現、功能已知或具生物醫學研究價值為標準，找出在{cancer_type}中，這些基因中「較知名、常被研究」的基因，並簡單說明理由。",
        "請使用以下格式：",
        "```",
        "Gene: 基因名稱, 原因: 原因簡述",
        "```",
        "不要加上數字或破折號，不要換行或多段解釋，一行輸出一個基因。"
    ])

    result = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content
    except Exception as e:
        print("Error:", e)

    return result, prompt
