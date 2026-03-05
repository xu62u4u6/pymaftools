import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pymaftools.core.PivotTable import PivotTable
from pymaftools.model.modelUtils import evaluate_model, get_importance, to_importance_table


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
