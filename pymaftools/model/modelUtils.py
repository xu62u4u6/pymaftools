from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from ..core.PivotTable import PivotTable


def get_importance(model: object) -> pd.Series:
    """
    Extract feature importance from a fitted model.

    Supports sklearn estimators with ``feature_importances_`` and
    ``OmicsStackingModel`` instances. For ``OmicsStackingModel``, each feature
    is prefixed with its omics layer name (``"{omics_name}::{feature}"``) so
    identical gene names across layers never collide.

    Parameters
    ----------
    model : object
        A fitted model.

    Returns
    -------
    pd.Series
        Feature importances indexed by feature names.

    Raises
    ------
    ValueError
        If the model type is not supported.
    """
    if hasattr(model, "feature_importances_"):
        try:
            return pd.Series(model.feature_importances_, index=model.feature_names_in_)
        except AttributeError:
            return pd.Series(model.feature_importances_)

    elif hasattr(model, "get_omics_feature_importance") and hasattr(
        model, "omics_dict"
    ):
        importances = []
        for omics_name in model.omics_dict.keys():
            imp = model.get_omics_feature_importance(omics_name)
            prefix = f"{omics_name}::"
            prefixed_index = [
                feat if str(feat).startswith(prefix) else f"{prefix}{feat}"
                for feat in imp.index
            ]
            importances.append(pd.Series(imp.to_numpy(), index=prefixed_index))
        return pd.concat(importances)

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Evaluate a single model and return metric dictionary.

    Parameters
    ----------
    model : object
        A fitted model with ``predict`` and ``predict_proba`` methods.
    X_test : pd.DataFrame
        Test features.
    y_test : array-like
        True labels.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``acc``, ``f1``, and ``auc``.
    """
    y_pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    # predict_proba column i is model.classes_[i] (sklearn convention, also
    # implemented by OmicsStackingModel). roc_auc_score would otherwise assume
    # sorted labels and score the wrong columns.
    fitted_labels = np.asarray(model.classes_)
    if len(fitted_labels) != prob.shape[1]:
        raise ValueError(
            f"model.classes_ has {len(fitted_labels)} labels but predict_proba "
            f"returned {prob.shape[1]} columns."
        )
    unseen = sorted(set(np.asarray(y_test)) - set(fitted_labels))
    if unseen:
        warnings.warn(
            f"y_test contains classes not seen in training {unseen}; the model "
            "has no probability for them, so AUC is undefined and set to NaN. "
            "Use stratified splits so every class appears in each training fold.",
            UserWarning,
            stacklevel=2,
        )
        auc = np.nan
    elif prob.shape[1] == 2:
        auc = roc_auc_score(np.asarray(y_test) == fitted_labels[1], prob[:, 1])
    else:
        # sklearn requires ``labels`` sorted, so sort the columns to match.
        sorted_columns = np.argsort(fitted_labels)
        auc = roc_auc_score(
            y_test,
            prob[:, sorted_columns],
            multi_class="ovr",
            labels=fitted_labels[sorted_columns],
        )

    return {
        "acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "auc": auc,
    }


def cross_validate_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model_func: callable,
    model_name: str,
    n_seeds: int = 5,
    n_splits: int = 5,
    random_state_base: int = 0,
    verbose: bool = True,
    evaluate_func: callable | None = None,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Run repeated stratified cross-validation, collecting feature importances and metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples as rows).
    y : pd.Series
        Target labels.
    model_func : callable
        Factory ``model_func(seed) -> model`` returning a fresh model instance.
    model_name : str
        Name identifier for this model.
    n_seeds : int, default 5
        Number of random seeds (repetitions).
    n_splits : int, default 5
        Number of CV folds per seed.
    random_state_base : int, default 0
        Base value added to each seed for reproducibility.
    verbose : bool, default True
        Whether to display a progress bar.
    evaluate_func : callable, optional
        Function ``(model, X_test, y_test) -> dict`` returning per-fold metrics.
    groups : array-like, optional
        Group identifiers, such as patient IDs. When supplied, use
        ``StratifiedGroupKFold`` so one group cannot appear in both train and
        test folds.

    Returns
    -------
    importance_df : pd.DataFrame
        Long-format feature importance table.
    metric_df : pd.DataFrame or None
        Long-format metrics table (``None`` if ``evaluate_func`` is not provided).
    """
    importance_records: list[dict] = []
    metrics_records: list[dict] = []

    if groups is not None:
        if isinstance(groups, pd.Series):
            if not groups.index.equals(X.index):
                raise ValueError("groups index must match X index exactly.")
            groups = groups.copy()
        else:
            groups = pd.Series(groups, index=X.index)
        if len(groups) != len(X):
            raise ValueError("groups must contain one value per row in X.")
        if groups.isna().any():
            raise ValueError("groups cannot contain missing values.")

    warned_nested_cv = False
    for seed in tqdm(range(n_seeds), desc="CV seeds", disable=not verbose):
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed + random_state_base,
            )
            split_iterator = splitter.split(X, y)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed + random_state_base,
            )
            split_iterator = splitter.split(X, y, groups)

        for fold_id, (train_idx, test_idx) in enumerate(split_iterator):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = model_func(seed)
            if groups is not None:
                train_groups = groups.iloc[train_idx]
                fit_params = inspect.signature(model.fit).parameters
                if "groups" in fit_params:
                    model.fit(X_train, y_train, groups=train_groups)
                else:
                    if hasattr(model, "cv") and not warned_nested_cv:
                        warnings.warn(
                            f"{type(model).__name__}.fit() does not accept 'groups'; "
                            "outer CV respects groups, but inner CV splits may leak "
                            "samples from the same group across inner folds.",
                            UserWarning,
                            stacklevel=2,
                        )
                        warned_nested_cv = True
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            if evaluate_func is not None:
                metrics: dict = {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_id,
                }
                metrics.update(evaluate_func(model, X_test, y_test))
                metrics_records.append(metrics)

            imp = get_importance(model)
            for feature, importance in imp.items():
                importance_records.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold_id,
                        "feature": feature,
                        "importance": importance,
                    }
                )

    importance_df = pd.DataFrame(importance_records)
    metric_df = pd.DataFrame(metrics_records) if metrics_records else None

    return importance_df, metric_df


def plot_metric_comparison_with_annotation(
    data: pd.DataFrame,
    metrics: list[str] | None = None,
    group_col: str = "model",
    order: list[str] | None = None,
    palette: str = "Set2",
    test: str = "Mann-Whitney",
    alpha: float = 0.8,
    fontsize: int = 14,
    figsize: tuple[int, int] | None = None,
    title_prefix: str | None = None,
    save_path: str | None = None,
    **save_kwargs,
):
    """
    Plot metric comparison boxplots with statistical annotations.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing model metrics.
    metrics : list[str], optional
        Metric column names to plot. Default ``["acc", "f1", "auc"]``.
    group_col : str, default ``"model"``
        Column used for grouping.
    order : list[str], optional
        Display order of groups.
    palette : str, default ``"Set2"``
        Seaborn color palette.
    test : str, default ``"Mann-Whitney"``
        Statistical test for annotations.
    alpha : float, default 0.8
        Box transparency.
    fontsize : int, default 14
        Font size.
    figsize : tuple, optional
        Figure size.
    title_prefix : str, optional
        Title prefix (``None`` disables titles).
    save_path : str, optional
        Path to save the figure.
    **save_kwargs
        Additional arguments passed to save method.

    Returns
    -------
    ModelPlot
        The plotter instance.
    """
    if metrics is None:
        metrics = ["acc", "f1", "auc"]
    if order is None:
        order = ["SNV", "CNV-gene", "CNV-cluster", "STACK"]

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
        **save_kwargs,
    )


def to_importance_table(all_importance_df: pd.DataFrame, omic: str) -> PivotTable:
    """
    Convert long-format importance data to a sorted PivotTable.

    Features not selected in a given ``(seed, fold)`` run are treated as having
    ``0.0`` importance rather than being skipped, avoiding upward selection bias
    for unstable features. ``feature_metadata`` records both ``mean`` (across
    all seeds, with unselected runs counted as zero) and ``selection_freq``
    (fraction of ``(seed, fold)`` runs in which the feature was selected).

    Parameters
    ----------
    all_importance_df : pd.DataFrame
        Long-format importance DataFrame with columns ``model``, ``seed``,
        ``fold`` (optional), ``feature``, and ``importance``.
    omic : str
        Omics/model name to filter by.

    Returns
    -------
    PivotTable
        Feature x seed matrix sorted by mean importance (descending).
    """
    omic_df = all_importance_df.loc[all_importance_df["model"] == omic]
    if omic_df.empty:
        available = sorted(all_importance_df["model"].dropna().unique().tolist())
        raise ValueError(
            f"Model {omic!r} not found in all_importance_df['model']. "
            f"Available models: {available}."
        )

    if "fold" in omic_df.columns:
        fold_table = omic_df.pivot_table(
            index="feature",
            columns=["seed", "fold"],
            values="importance",
            aggfunc="mean",
            fill_value=0.0,
        )
        n_total_runs = fold_table.shape[1]
        selected_counts = (
            omic_df.drop_duplicates(["seed", "fold", "feature"])
            .groupby("feature")
            .size()
        )
        pivot_df = fold_table.T.groupby(level="seed").mean().T
    else:
        pivot_df = omic_df.pivot_table(
            index="feature",
            columns="seed",
            values="importance",
            aggfunc="mean",
            fill_value=0.0,
        )
        n_total_runs = pivot_df.shape[1]
        selected_counts = (
            omic_df.drop_duplicates(["seed", "feature"]).groupby("feature").size()
        )

    table = PivotTable(pivot_df)
    table.feature_metadata["mean"] = table.mean(axis=1)
    table.feature_metadata["selection_freq"] = (
        selected_counts.reindex(table.index).fillna(0).astype(float) / n_total_runs
    )

    sorted_table = table.sort_features(by="mean", ascending=False)
    return sorted_table


def plot_top_feature_importance_heatmap(
    mean_importance_df: pd.DataFrame,
    omic: str,
    top_n: int = 20,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    save_path: str | None = None,
    **save_kwargs,
):
    """
    Plot heatmap of top-N most important features.

    Parameters
    ----------
    mean_importance_df : pd.DataFrame
        Feature importance data.
    omic : str
        Omics name identifier.
    top_n : int, default 20
        Number of top features to display.
    cmap : str, default ``"viridis"``
        Colormap for the heatmap.
    figsize : tuple, default ``(10, 6)``
        Figure size.
    title : str, optional
        Plot title (``None`` disables title).
    save_path : str, optional
        Path to save the figure.
    **save_kwargs
        Additional arguments passed to save method.

    Returns
    -------
    ModelPlot
        The plotter instance.
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
        **save_kwargs,
    )


def run_rfecv_feature_selection(
    pivot: PivotTable,
    label_col: str = "subtype",
    estimator: object | None = None,
    step: int = 10,
    scoring: str = "accuracy",
    min_features_to_select: int = 10,
    plot: bool = True,
    random_state: int = 42,
    title: str | None = None,
    save_path: str | None = None,
    groups: pd.Series | np.ndarray | None = None,
    **save_kwargs,
) -> tuple[list[str], RFECV]:
    """
    Run RFECV feature selection on a PivotTable.

    Parameters
    ----------
    pivot : PivotTable
        Feature x sample table.
    label_col : str, default ``"subtype"``
        Column in ``sample_metadata`` containing target labels.
    estimator : sklearn estimator, optional
        Model to use (default: ``RandomForestClassifier``).
    step : int, default 10
        Number of features removed per iteration.
    scoring : str, default ``"accuracy"``
        Scoring metric (e.g. ``"accuracy"``, ``"f1_macro"``).
    min_features_to_select : int, default 10
        Minimum number of features to keep.
    plot : bool, default True
        Whether to plot the performance curve.
    random_state : int, default 42
        Random seed.
    title : str, optional
        Plot title (``None`` disables title).
    save_path : str, optional
        Path to save the figure.
    groups : array-like, optional
        Patient or other group identifiers. When supplied, use
        ``StratifiedGroupKFold`` so a group cannot appear in both RFECV folds.
    **save_kwargs
        Additional arguments passed to save method.

    Returns
    -------
    selected_features : list[str]
        Selected feature names.
    selector : RFECV
        Fitted RFECV object.
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)

    X = pivot.T.values
    y = np.array(pivot.sample_metadata[label_col].values)
    fit_groups = None
    if groups is not None:
        if isinstance(groups, pd.Series):
            if not groups.index.equals(pivot.columns):
                raise ValueError("groups index must match PivotTable samples exactly.")
            fit_groups = groups.to_numpy()
        else:
            fit_groups = np.asarray(groups)
            if len(fit_groups) != len(pivot.columns):
                raise ValueError("groups must contain one value per PivotTable sample.")
        if pd.isna(fit_groups).any():
            raise ValueError("groups cannot contain missing values.")

    cv = (
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
        if fit_groups is not None
        else StratifiedKFold(5, shuffle=True, random_state=random_state)
    )

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        verbose=0,
        min_features_to_select=min_features_to_select,
    )

    if fit_groups is None:
        selector.fit(X, y)
    else:
        selector.fit(X, y, groups=fit_groups)

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
            **save_kwargs,
        )

    return selected_features, selector


def run_model_evaluation(
    model_configs: list[dict],
    y: pd.Series,
    n_seeds: int = 100,
    n_splits: int = 5,
    evaluate_func: callable | None = None,
    verbose: bool = True,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None]:
    """
    Run cross-validation and importance analysis for multiple models.

    Parameters
    ----------
    model_configs : list[dict]
        Each dict must have keys ``"name"`` (str), ``"model_func"``
        (callable), and ``"X"`` (pd.DataFrame).
    y : pd.Series
        Target labels.
    n_seeds : int, default 100
        Number of random seeds.
    n_splits : int, default 5
        Number of CV folds.
    evaluate_func : callable, optional
        Evaluation function ``(model, X_test, y_test) -> dict``. When ``None``,
        ``all_metrics_df`` is returned as ``None``.
    verbose : bool, default True
        Whether to print progress.
    groups : array-like, optional
        Group identifiers forwarded to :func:`cross_validate_importance`.

    Returns
    -------
    result_dict : dict
        Per-model results with ``"importance"`` and ``"metrics"`` keys.
    all_importance_df : pd.DataFrame
        Combined long-format feature importance data.
    all_metrics_df : pd.DataFrame or None
        Combined long-format classification metrics, or ``None`` if
        ``evaluate_func`` was not provided.
    """
    if not model_configs:
        raise ValueError("model_configs must contain at least one model configuration.")

    result: dict = {}

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
            verbose=verbose,
            groups=groups,
        )

        result[model_name] = {
            "importance": importance_df,
            "metrics": metric_df,
        }

    all_importance_df = pd.concat(
        [res["importance"] for res in result.values()],
        ignore_index=True,
    )

    metric_frames = [
        res["metrics"] for res in result.values() if res["metrics"] is not None
    ]
    all_metrics_df = (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else None
    )

    return result, all_importance_df, all_metrics_df
