import json
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from .PivotTable import PivotTable

def table_to_distance(table):
    """將表格轉換為距離矩陣"""
    similarity = table.T.compute_similarity(method="cosine")
    distance = 1 - similarity.values
    return distance

def k_fold_clustering_evaluation(table: PivotTable, 
                                min_clusters: int = 2,
                                max_clusters: int = 50, 
                                metric: str = "cosine", 
                                random_state: int = 42):
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
    subtype = table.sample_metadata.subtype.values
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    if not isinstance(table, PivotTable):
        raise ValueError("Input table must be a PivotTable instance.")
    if min_clusters < 2 or max_clusters < min_clusters:
        raise ValueError("min_clusters must be at least 2 and max_clusters must be greater than or equal to min_clusters.")
    all_results = []
    cluster_label_dict = {}  

    for fold, (train_idx, test_idx) in enumerate(kf.split(table.T.values, subtype)):
        print(f"Processing fold {fold + 1}/5")
        
        # 獲取訓練樣本
        sample_train = table.sample_metadata.iloc[train_idx].index
        table_train = table.subset(samples=sample_train)
        distance_matrix_train = table_to_distance(table_train)
        X_train = table_train.values
        
        # 測試不同的聚類數
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
            score = silhouette_score(X_train, labels, metric=metric)
            
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
                aligned_df[col] = align_clusters(ref_labels, fold_df[col].values, n_clusters=int(k))

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
            ari_matrix.iloc[i, j] = adjusted_rand_score(df.iloc[:, i], df.iloc[:, j])

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
    plt.title(f"Cluster {k} ARI between {n} folds, mean={mean_off_diagonal_ari(ari_matrix):.2f}")
    plt.show()

def plot_clustering_metrics_and_find_best_k(metric_df, filename, dpi=300, bbox_inches='tight', transparent=True, format=None, **kwargs):

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metric_df.index, metric_df["penalized_silhouette"], label="Penalized Silhouette", color="orange", linestyle='--')

    fold_columns = [f"fold{fold}_silhouette" for fold in range(1, 6)]
    fold_df = metric_df.loc[:, fold_columns]
    fold_means = fold_df.mean(axis=1)
    fold_stds = fold_df.std(axis=1)
    
    ax.plot(metric_df.index, fold_means, label="Mean silhouette (5-fold)", color="blue", linestyle='-')
    ax.errorbar(metric_df.index, fold_means, yerr=fold_stds, fmt='none', capsize=2, capthick=1, color='red', alpha=0.5, label="5-fold mean ± std")

    ax.plot(metric_df.index, metric_df["mean_ari_5_fold"], label="Mean ARI (5-fold)", color="green", linestyle='-.')

    best_k = metric_df["penalized_silhouette"].idxmax()
    ax.axvline(best_k, color='red', linestyle=':', alpha=0.7, label=f'Best k={best_k}')

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Score")
    ax.set_title("Clustering Quality Metrics vs. Cluster Number")
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

