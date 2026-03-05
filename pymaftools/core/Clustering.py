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


def table_to_distance(table: PivotTable) -> np.ndarray:
    """
    Convert a PivotTable to a distance matrix.

    Parameters
    ----------
    table : PivotTable
        Input data table with samples and features.

    Returns
    -------
    numpy.ndarray
        Distance matrix computed as 1 minus cosine similarity.
    """
    similarity = table.T.compute_similarity(method="cosine")
    distance = 1 - similarity.values
    return distance

def k_fold_clustering_evaluation(table: PivotTable,
                                 min_clusters: int = 2,
                                 max_clusters: int = 50,
                                 metric: Literal['cosine', 'hamming', 'jaccard'] = "cosine",
                                 random_state: int = 42,
                                 group_col: str = "subtype") -> tuple[pd.DataFrame, dict[int, dict[int, np.ndarray]]]:
    """
    Evaluate the optimal number of clusters using K-fold cross-validation.

    Parameters
    ----------
    table : PivotTable
        Gene expression or CNV data table.
    min_clusters : int, optional
        Minimum number of clusters to evaluate, by default 2.
    max_clusters : int, optional
        Maximum number of clusters to evaluate, by default 50.
    metric : {'cosine', 'hamming', 'jaccard'}, optional
        Similarity metric to use, by default 'cosine'.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    group_col : str, optional
        Column name in sample metadata used for grouping, by default 'subtype'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing silhouette scores for each fold and cluster count.
    dict[int, dict[int, numpy.ndarray]]
        Mapping of cluster count k to fold-wise cluster labels.
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

        sample_train = table.sample_metadata.iloc[train_idx].index
        table_train = table.subset(samples=sample_train)
        similarity_matrix = table_train.T.compute_similarity(method=metric)
        distance_matrix_train = 1 - similarity_matrix # dist = 1 - similarity
        # fill diagonal to 0
        for i in range(len(distance_matrix_train)):
            distance_matrix_train.iloc[i, i] = 0.0

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

            score = silhouette_score(table_train, labels, metric=metric)

            all_results.append({
                'fold': fold + 1,
                'n_clusters': k,
                'silhouette_score': score
            })

    return pd.DataFrame(all_results), cluster_label_dict

def align_clusters(ref_labels: np.ndarray, target_labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Align target cluster labels to reference labels using the Hungarian algorithm.

    Parameters
    ----------
    ref_labels : numpy.ndarray
        Reference cluster labels to align against.
    target_labels : numpy.ndarray
        Target cluster labels to be remapped.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray
        Remapped target labels aligned to the reference labeling.
    """
    cm = confusion_matrix(ref_labels, target_labels, labels=range(n_clusters))
    cost_matrix = -cm  # Hungarian algorithm finds *minimum* cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = dict(zip(col_ind, row_ind))
    aligned = np.vectorize(lambda x: mapping.get(x, x))(target_labels)
    return aligned


def align_cluster_label_dict(cluster_label_dict: dict[int, dict[int, np.ndarray]]) -> dict[int, pd.DataFrame]:
    """
    Align cluster labels across folds using fold 1 as the reference.

    Parameters
    ----------
    cluster_label_dict : dict[int, dict[int, numpy.ndarray]]
        Mapping of cluster count k to a dict of fold number to label array.
        Structure: ``{k: {fold: labels}}``.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping of each k to an aligned DataFrame (samples x folds).
    """
    aligned_fold_df_dict = {}

    for k, fold_dict in cluster_label_dict.items():
        fold_df = pd.DataFrame(fold_dict)
        ref_labels = fold_df.iloc[:, 0].values
        aligned_df = pd.DataFrame(index=fold_df.index)

        for col in fold_df.columns:
            if col == fold_df.columns[0]:
                aligned_df[col] = fold_df[col]
            else:
                aligned_df[col] = align_clusters(
                    ref_labels, fold_df[col].values, n_clusters=int(k))

        aligned_fold_df_dict[k] = aligned_df

    return aligned_fold_df_dict


def convert_ndarray_to_list(obj: object) -> object:
    """
    Recursively convert all numpy.ndarray values in a nested structure to lists.

    Parameters
    ----------
    obj : object
        Input object, typically a dict, list, or numpy.ndarray.

    Returns
    -------
    object
        The same structure with all numpy.ndarray instances replaced by lists.
    """
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj


def calculate_ari_matrix(aligned_cluster_label_dict: dict[int, pd.DataFrame], k: int) -> pd.DataFrame:
    """
    Compute the pairwise Adjusted Rand Index (ARI) matrix across folds.

    Parameters
    ----------
    aligned_cluster_label_dict : dict[int, pd.DataFrame]
        Aligned cluster labels, as returned by ``align_cluster_label_dict``.
    k : int
        Number of clusters to evaluate.

    Returns
    -------
    pd.DataFrame
        Square DataFrame of pairwise ARI scores between folds.
    """
    df = aligned_cluster_label_dict[k]
    folds = df.columns
    n = len(folds)
    ari_matrix = pd.DataFrame(index=folds, columns=folds, dtype=float)

    for i in range(n):
        for j in range(n):
            ari_matrix.iloc[i, j] = adjusted_rand_score(
                df.iloc[:, i], df.iloc[:, j])

    return ari_matrix


def plot_ari_matrix(aligned_cluster_label_dict: dict[int, pd.DataFrame], k: int) -> None:
    """
    Plot the upper-triangle ARI heatmap for a given cluster count k.

    Parameters
    ----------
    aligned_cluster_label_dict : dict[int, pd.DataFrame]
        Aligned cluster labels, as returned by ``align_cluster_label_dict``.
    k : int
        Number of clusters to visualize.
    """
    ari_matrix = calculate_ari_matrix(aligned_cluster_label_dict, k)
    n = len(ari_matrix)

    def mean_off_diagonal_ari(ari_matrix: pd.DataFrame) -> float:
        """Compute the mean of off-diagonal elements in the ARI matrix."""
        n = len(ari_matrix)
        mask = ~np.eye(n, dtype=bool)
        return ari_matrix.values[mask].mean()

    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(ari_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(
        f"Cluster {k} ARI between {n} folds, mean={mean_off_diagonal_ari(ari_matrix):.2f}")
    plt.show()


def run_random_forest_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 100,
) -> tuple[RandomForestClassifier, list[float], pd.DataFrame]:
    """
    Run stratified K-fold cross-validated Random Forest classification.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target labels of shape (n_samples,).
    feature_names : list[str]
        Names corresponding to each feature column in X.
    n_splits : int, optional
        Number of CV folds, by default 5.
    random_state : int, optional
        Random seed, by default 42.
    n_estimators : int, optional
        Number of trees in the forest, by default 100.

    Returns
    -------
    RandomForestClassifier
        The last trained model.
    list[float]
        Accuracy scores for each fold.
    pd.DataFrame
        Feature importances per fold and their mean.
    """
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

def run_random_forest_multiple_seeds(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seeds: range | list[int] = range(5),
    n_estimators: int = 100,
) -> tuple[list[RandomForestClassifier], pd.DataFrame]:
    """
    Train Random Forest classifiers with multiple random seeds on the full dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : array-like
        Target labels.
    feature_names : list[str]
        Names corresponding to each feature column in X.
    seeds : range or list[int], optional
        Random seeds to iterate over, by default ``range(5)``.
    n_estimators : int, optional
        Number of trees in each forest, by default 100.

    Returns
    -------
    list[RandomForestClassifier]
        Trained models, one per seed.
    pd.DataFrame
        Feature importances per seed and their mean.
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


def plot_cluster_feature_importance_boxplot(
    table: pd.DataFrame,
    importance_cols: list[str],
    top_n: int = 20,
) -> None:
    """
    Draw a bar and box plot of the top N cluster feature importances.

    Parameters
    ----------
    table : pd.DataFrame
        DataFrame containing cluster information with importance scores.
        Must include a ``mean_importance`` column, plus ``unique_chr_arm``
        and ``gene_count`` for axis labels.
    importance_cols : list[str]
        Column names for per-fold importance scores.
    top_n : int, optional
        Number of top clusters to display, by default 20.
    """
    top_data = table.sort_values(by="mean_importance", ascending=False).head(top_n).copy()
    positions = np.arange(top_n)

    box_values = [top_data.iloc[i][importance_cols].values.tolist() for i in range(top_n)]
    bar_values = top_data["mean_importance"].values.tolist()

    plt.figure(figsize=(14, 7))

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

    plt.bar(positions, bar_values, width=0.6, color='skyblue', label='Mean Importance')

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

def plot_cluster_feature_importance(
    table: pd.DataFrame,
    importance_cols: list[str],
    top_n: int = 20,
) -> None:
    """
    Plot top N cluster feature importances as bar (mean) and scatter (per-fold).

    Parameters
    ----------
    table : pd.DataFrame
        DataFrame containing cluster information with importance scores.
        Must include ``mean_importance``, ``unique_chr_arm``, and
        ``gene_count`` columns.
    importance_cols : list[str]
        Column names for per-fold importance scores.
    top_n : int, optional
        Number of top clusters to display, by default 20.
    """
    top_data = table.nlargest(top_n, "mean_importance").copy()
    top_data["Cluster"] = [f"C{i}" for i in top_data.index]

    long_df = top_data[["Cluster", "unique_chr_arm", "gene_count"] + importance_cols].melt(
        id_vars=["Cluster", "unique_chr_arm", "gene_count"],
        value_vars=importance_cols,
        var_name="Fold",
        value_name="Importance"
    )

    plt.figure(figsize=(14, 7))

    sns.barplot(data=top_data, x="Cluster", y="mean_importance", color="skyblue", zorder=0)
    sns.stripplot(data=long_df, x="Cluster", y="Importance", color="gray", alpha=0.6, jitter=0.1, zorder=1)

    new_labels = [
        f"\n\n{row['Cluster']}\narm: {row['unique_chr_arm']}\nn={row['gene_count']}"
        for _, row in top_data.iterrows()
    ]
    plt.xticks(ticks=np.arange(top_n), labels=new_labels, rotation=45, ha='right')

    plt.ylabel("Feature Importance")
    plt.title("Top Cluster Feature Importances\n(Dots = 5-Fold, Bar = Mean)")
    plt.tight_layout()
    plt.show()

def run_feature_clustering(
    table: PivotTable,
    result_path: str,
    max_clusters: int = 200,
) -> pd.DataFrame:
    """
    Run agglomerative clustering on features for a range of cluster counts.

    Parameters
    ----------
    table : PivotTable
        Input data table with features as rows and samples as columns.
    result_path : str
        File path to save the resulting CSV of silhouette scores.
    max_clusters : int, optional
        Maximum number of clusters to evaluate, by default 200.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``n_clusters`` and ``silhouette`` for each k.
    """
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
        score = silhouette_score(X, labels, metric=metric)
        results.append({'n_clusters': k, 'silhouette': score})
        print(f"score: {score}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(result_path, index=False)
    return results_df

def plot_clustering_metrics_and_find_best_k(
        metric_df: pd.DataFrame,
        filename: str,
        title: str | None = None,
        target_col: str = "mean_silhouette",
        dpi: int = 300,
        bbox_inches: str = 'tight',
        transparent: bool = True,
        format: str | None = None,
        **kwargs) -> int:
    """
    Plot silhouette and ARI metrics across cluster counts and find the best k.

    Parameters
    ----------
    metric_df : pd.DataFrame
        DataFrame indexed by cluster count with per-fold silhouette columns
        (``fold1_silhouette`` ... ``fold5_silhouette``) and ``mean_ari_5_fold``.
    filename : str
        Output file path for the saved figure.
    title : str or None, optional
        Plot title. If None, no title is displayed.
    target_col : str, optional
        Column name to maximize for selecting the best k, by default
        ``'mean_silhouette'``.
    dpi : int, optional
        Resolution in dots per inch, by default 300.
    bbox_inches : str, optional
        Bounding box setting for saving, by default ``'tight'``.
    transparent : bool, optional
        Whether the background is transparent, by default True.
    format : str or None, optional
        Output format. Inferred from ``filename`` extension if None.
    **kwargs
        Additional keyword arguments passed to ``fig.savefig``.

    Returns
    -------
    int
        The cluster count k that maximizes ``target_col``.
    """
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
    plt.close(fig)

    return best_k


def gpt_known_genes_summary(
    client: object,
    genes: list[str],
    arm: str,
    cancer_type: str = "lung cancer",
) -> tuple[str, str]:
    """
    Query GPT-4 for well-known genes in a given chromosomal arm and cancer type.

    Parameters
    ----------
    client : object
        OpenAI client instance with a ``chat.completions.create`` method.
    genes : list[str]
        List of gene names to evaluate.
    arm : str
        Chromosomal arm where the genes are located (e.g., ``'3p'``).
    cancer_type : str, optional
        Cancer type context for the query, by default ``'lung cancer'``.

    Returns
    -------
    str
        GPT-4 response text listing notable genes and reasons.
    str
        The prompt that was sent to the model.
    """
    prompt = "\n".join([
        f"The following is a list of human CNV genes located on {arm}:",
        ", ".join(genes),
        "",
        f"Based on cancer literature, known functions, and biomedical research value, identify the well-known and frequently studied genes among these in {cancer_type}, and briefly explain why.",
        "Use the following format:",
        "```",
        "Gene: gene_name, Reason: brief explanation",
        "```",
        "Do not add numbers or dashes. Do not use multiple lines or paragraphs for explanation. Output one gene per line."
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
