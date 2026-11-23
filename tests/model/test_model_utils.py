import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from pymaftools.core.PivotTable import PivotTable
from pymaftools.model.modelUtils import (
    cross_validate_importance,
    evaluate_model,
    get_importance,
    to_importance_table,
)


def test_evaluate_model_returns_metric_dict():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(30, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series([0] * 15 + [1] * 15)

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert {"acc", "f1", "auc"}.issubset(metrics.keys())
    assert all(0.0 <= metrics[k] <= 1.0 for k in ["acc", "f1", "auc"])


def test_get_importance_returns_series_with_feature_names():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(20, 4)), columns=["A", "B", "C", "D"])
    y = pd.Series([0] * 10 + [1] * 10)

    model = RandomForestClassifier(n_estimators=10, random_state=123)
    model.fit(X, y)

    importance = get_importance(model)

    assert isinstance(importance, pd.Series)
    assert list(importance.index) == ["A", "B", "C", "D"]


def test_to_importance_table_returns_sorted_pivot_table():
    all_importance_df = pd.DataFrame(
        [
            {"model": "SNV", "seed": 0, "fold": 0, "feature": "TP53", "importance": 0.9},
            {"model": "SNV", "seed": 1, "fold": 0, "feature": "TP53", "importance": 0.8},
            {"model": "SNV", "seed": 0, "fold": 0, "feature": "KRAS", "importance": 0.2},
            {"model": "SNV", "seed": 1, "fold": 0, "feature": "KRAS", "importance": 0.3},
            {"model": "CNV", "seed": 0, "fold": 0, "feature": "MYC", "importance": 0.6},
        ]
    )

    table = to_importance_table(all_importance_df, "SNV")

    assert isinstance(table, PivotTable)
    assert list(table.index)[0] == "TP53"
    assert "mean" in table.feature_metadata.columns
    assert table.shape[1] == 2


def test_cross_validate_importance_accepts_array_groups():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(18, 4)), columns=list("ABCD"))
    y = pd.Series([0, 1] * 9, index=X.index)
    groups = np.repeat(np.arange(9), 2)

    importance, metrics = cross_validate_importance(
        X,
        y,
        model_func=lambda seed: RandomForestClassifier(
            n_estimators=5, random_state=seed
        ),
        model_name="grouped",
        n_seeds=1,
        n_splits=3,
        verbose=False,
        groups=groups,
        evaluate_func=evaluate_model,
    )

    assert not importance.empty
    assert metrics is not None
    assert len(metrics) == 3


class _PerfectClassifier:
    """predict_proba columns follow ``classes_`` in a non-sorted order."""

    def __init__(self, classes, y):
        self.classes_ = np.asarray(classes)
        self.y = np.asarray(y)

    def predict(self, X):
        return self.y

    def predict_proba(self, X):
        return np.array(
            [[float(label == truth) for label in self.classes_] for truth in self.y]
        )


def test_evaluate_model_auc_follows_model_class_order_multiclass():
    # Non-alphabetical order: sklearn's default (sorted) mapping would score
    # P(LUAD) as ASC and report 0.5 for a perfect model (audit ISS-001).
    y = np.array(["LUAD", "ASC", "LUSC"] * 10)
    model = _PerfectClassifier(["LUAD", "ASC", "LUSC"], y)

    assert evaluate_model(model, None, y)["auc"] == 1.0


def test_evaluate_model_auc_follows_model_class_order_binary():
    # Reversed order used to invert the binary AUC to 1 - AUC.
    y = np.array(["LUAD", "LUSC"] * 10)
    model = _PerfectClassifier(["LUSC", "LUAD"], y)

    assert evaluate_model(model, None, y)["auc"] == 1.0


def test_evaluate_model_imperfect_multiclass_matches_sorted_reference():
    # Asymmetric scores, so a wrong column permutation cannot give the same AUC.
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    y = np.array(["LUAD", "ASC", "LUSC"] * 20)
    prob = rng.dirichlet([1, 1, 1], size=len(y))
    prob[y == "LUAD", 0] += 0.6
    prob[y == "ASC", 1] += 0.2
    prob = prob / prob.sum(axis=1, keepdims=True)

    class _Model:
        classes_ = np.array(["LUAD", "ASC", "LUSC"])

        def predict(self, X):
            return y

        def predict_proba(self, X):
            return prob

    sorted_reference = roc_auc_score(
        y, prob[:, [1, 0, 2]], multi_class="ovr", labels=["ASC", "LUAD", "LUSC"]
    )
    assert evaluate_model(_Model(), None, y)["auc"] == pytest.approx(sorted_reference)


def test_evaluate_model_unseen_test_class_gives_nan_auc_with_warning():
    # A stacking model trained without ASC returns two probability columns;
    # scoring "is ASC" against P(LUSC) used to produce a silent bogus AUC.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_stacking_model import _build_model_inputs

    from pymaftools.model.StackingModel import OmicsStackingModel

    omics_dict, X, _ = _build_model_inputs(n_samples=30)
    y_train = np.array(["LUAD"] * 15 + ["LUSC"] * 15)
    model = OmicsStackingModel(omics_dict=omics_dict, class_order=["LUAD", "ASC", "LUSC"])
    model.fit(X, y_train)
    assert list(model.classes_) == ["LUAD", "LUSC"]

    y_test = y_train.copy()
    y_test[:3] = "ASC"
    with pytest.warns(UserWarning, match="not seen in training"):
        metrics = evaluate_model(model, X, y_test)
    assert np.isnan(metrics["auc"])
