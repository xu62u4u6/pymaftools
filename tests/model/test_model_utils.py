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
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    class _FixedBinaryModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            probs = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.9])
            return np.column_stack([1.0 - probs, probs])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    X_test = pd.DataFrame({"f0": np.arange(6)})
    # Imbalanced labels so macro F1 (0.829) and weighted F1 (0.838) differ:
    # with balanced labels both averages coincide and the oracle cannot tell
    # which one evaluate_model reports.
    y_test = pd.Series([0, 0, 0, 0, 1, 1])
    model = _FixedBinaryModel()

    metrics = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    assert macro_f1 != pytest.approx(f1_score(y_test, y_pred, average="weighted"))
    assert metrics["acc"] == pytest.approx(accuracy_score(y_test, y_pred))
    assert metrics["f1"] == pytest.approx(macro_f1)
    assert metrics["auc"] == pytest.approx(roc_auc_score(y_test, y_prob))


def test_get_importance_returns_series_with_feature_names():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(20, 4)), columns=["A", "B", "C", "D"])
    y = pd.Series([0] * 10 + [1] * 10)

    model = RandomForestClassifier(n_estimators=10, random_state=123)
    model.fit(X, y)

    importance = get_importance(model)

    assert isinstance(importance, pd.Series)
    assert list(importance.index) == ["A", "B", "C", "D"]
    assert importance.sum() == pytest.approx(1.0)


def test_to_importance_table_returns_sorted_pivot_table():
    all_importance_df = pd.DataFrame(
        [
            {
                "model": "SNV",
                "seed": 0,
                "fold": 0,
                "feature": "TP53",
                "importance": 0.9,
            },
            {
                "model": "SNV",
                "seed": 1,
                "fold": 0,
                "feature": "TP53",
                "importance": 0.8,
            },
            {
                "model": "SNV",
                "seed": 0,
                "fold": 0,
                "feature": "KRAS",
                "importance": 0.2,
            },
            {
                "model": "SNV",
                "seed": 1,
                "fold": 0,
                "feature": "KRAS",
                "importance": 0.3,
            },
            {"model": "CNV", "seed": 0, "fold": 0, "feature": "MYC", "importance": 0.6},
        ]
    )

    table = to_importance_table(all_importance_df, "SNV")

    assert isinstance(table, PivotTable)
    assert list(table.index) == ["TP53", "KRAS"]
    assert table.feature_metadata.loc["TP53", "mean"] == pytest.approx(0.85)
    assert table.feature_metadata.loc["KRAS", "mean"] == pytest.approx(0.25)
    assert table.shape[1] == 2


def test_cross_validate_importance_accepts_array_groups():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(18, 4)), columns=list("ABCD"))
    y = pd.Series([0, 1] * 9, index=X.index)
    groups = np.repeat(np.arange(9), 2)
    group_series = pd.Series(groups, index=X.index)

    seen_test_groups: list[set[int]] = []

    def recording_evaluate(model, X_test, y_test):
        test_groups = set(group_series.loc[X_test.index].tolist())
        # Each group has 2 samples; every test fold must contain complete groups.
        for g in test_groups:
            group_members = set(group_series[group_series == g].index)
            assert group_members.issubset(set(X_test.index))
        seen_test_groups.append(test_groups)
        return evaluate_model(model, X_test, y_test)

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
        evaluate_func=recording_evaluate,
    )

    assert len(seen_test_groups) == 3
    # Disjoint test folds covering all 9 groups -> zero group leakage across splits.
    assert set().union(*seen_test_groups) == set(range(9))
    assert sum(len(g) for g in seen_test_groups) == 9
    assert set(importance["feature"]) == set("ABCD")
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
    model = OmicsStackingModel(
        omics_dict=omics_dict, class_order=["LUAD", "ASC", "LUSC"]
    )
    model.fit(X, y_train)
    assert list(model.classes_) == ["LUAD", "LUSC"]

    y_test = y_train.copy()
    y_test[:3] = "ASC"
    with pytest.warns(UserWarning, match="not seen in training"):
        metrics = evaluate_model(model, X, y_test)
    assert np.isnan(metrics["auc"])


def test_cross_validate_importance_forwards_groups_to_stacking_inner_cv():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_stacking_model import _build_model_inputs

    from pymaftools.model.StackingModel import OmicsStackingModel

    # 36 samples from 18 patients (2 samples per patient, same class per patient)
    omics_dict, X, _ = _build_model_inputs(n_samples=36, seed=21)
    y = pd.Series(
        ["LUAD"] * 18 + ["LUSC"] * 18,
        index=X.index,
    )
    groups = pd.Series(
        [f"P{i // 2}" for i in range(36)],
        index=X.index,
    )

    observed_inner_splits = []

    def make_model(seed):
        model = OmicsStackingModel(
            omics_dict=omics_dict,
            class_order=["LUAD", "LUSC"],
            cv=3,
            random_state=seed,
        )
        orig_fit = model.fit

        def tracking_fit(X_tr, y_tr, groups=None):
            assert groups is not None
            orig_fit(X_tr, y_tr, groups=groups)
            inner_cv = model.model.cv
            assert isinstance(inner_cv, list) and len(inner_cv) == 3
            g_arr = np.asarray(groups)
            for inner_tr, inner_val in inner_cv:
                assert set(g_arr[inner_tr]).isdisjoint(set(g_arr[inner_val]))
            observed_inner_splits.append(len(inner_cv))

        model.fit = tracking_fit
        return model

    imp_df, metrics_df = cross_validate_importance(
        X=X,
        y=y,
        model_func=make_model,
        model_name="STACK",
        n_seeds=1,
        n_splits=3,
        verbose=False,
        evaluate_func=evaluate_model,
        groups=groups,
    )

    assert len(observed_inner_splits) == 3
    assert len(metrics_df) == 3
    assert set(imp_df["feature"]).issuperset({"SNV::snv_f0", "CNV::cnv_f0"})


def test_cross_validate_importance_warns_when_nested_cv_estimator_ignores_groups():
    class _NestedCVStub:
        cv = 3

        def fit(self, X, y):
            self.feature_names_in_ = np.asarray(X.columns)
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
            return self

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(18, 3)), columns=["a", "b", "c"])
    y = pd.Series([0, 1] * 9, index=X.index)
    groups = pd.Series(np.repeat(np.arange(9), 2), index=X.index)

    with pytest.warns(UserWarning, match="inner CV splits may leak"):
        cross_validate_importance(
            X,
            y,
            model_func=lambda seed: _NestedCVStub(),
            model_name="stub",
            n_seeds=1,
            n_splits=3,
            verbose=False,
            groups=groups,
        )


def test_get_importance_prefixes_overlapping_genes_across_omics_layers():
    from pymaftools.model.StackingModel import OmicsStackingModel

    rng = np.random.default_rng(5)
    samples = [f"s{i}" for i in range(20)]
    snv = PivotTable(
        pd.DataFrame(rng.normal(size=(2, 20)), index=["TP53", "KRAS"], columns=samples)
    )
    cnv = PivotTable(
        pd.DataFrame(rng.normal(size=(2, 20)), index=["TP53", "KRAS"], columns=samples)
    )
    model = OmicsStackingModel(
        {"SNV": snv, "CNV": cnv}, class_order=["A", "B"], cv=3, random_state=0
    )
    X = model.prepare_features()
    y = np.array(["A"] * 10 + ["B"] * 10)
    model.fit(X, y)

    imp = get_importance(model)
    assert imp.index.is_unique
    assert list(imp.index) == ["SNV::TP53", "SNV::KRAS", "CNV::TP53", "CNV::KRAS"]
    assert imp.loc["SNV::TP53"] == pytest.approx(
        model.get_omics_feature_importance("SNV").loc["TP53"]
    )
    assert imp.loc["CNV::TP53"] == pytest.approx(
        model.get_omics_feature_importance("CNV").loc["TP53"]
    )


def test_to_importance_table_counts_unselected_runs_as_zero_and_records_selection_freq():
    from pymaftools.plot.ModelPlot import ModelPlot

    df = pd.DataFrame(
        [
            {
                "model": "SNV",
                "seed": 0,
                "fold": 0,
                "feature": "TP53",
                "importance": 0.5,
            },
            {
                "model": "SNV",
                "seed": 0,
                "fold": 0,
                "feature": "NOISE",
                "importance": 0.9,
            },
            {
                "model": "SNV",
                "seed": 1,
                "fold": 0,
                "feature": "TP53",
                "importance": 0.5,
            },
            {
                "model": "SNV",
                "seed": 2,
                "fold": 0,
                "feature": "TP53",
                "importance": 0.5,
            },
        ]
    )

    table = to_importance_table(df, "SNV")
    assert list(table.index) == ["TP53", "NOISE"]
    assert table.feature_metadata.loc["TP53", "mean"] == pytest.approx(0.5)
    assert table.feature_metadata.loc["NOISE", "mean"] == pytest.approx(0.3)
    assert table.feature_metadata.loc["TP53", "selection_freq"] == pytest.approx(1.0)
    assert table.feature_metadata.loc["NOISE", "selection_freq"] == pytest.approx(1 / 3)

    # ModelPlot._to_importance_table must agree with modelUtils.to_importance_table
    plot_table = ModelPlot()._to_importance_table(df, "SNV")
    pd.testing.assert_frame_equal(plot_table, table)


def test_run_model_evaluation_with_evaluate_func_none_returns_none_metrics():
    from pymaftools.model.modelUtils import run_model_evaluation

    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(16, 3)), columns=["g1", "g2", "g3"])
    y = pd.Series([0] * 8 + [1] * 8, index=X.index)

    result, all_imp, all_metrics = run_model_evaluation(
        model_configs=[
            {
                "name": "RF",
                "model_func": lambda seed: RandomForestClassifier(
                    n_estimators=5, random_state=seed
                ),
                "X": X,
            }
        ],
        y=y,
        n_seeds=1,
        n_splits=2,
        evaluate_func=None,
        verbose=False,
    )

    assert all_metrics is None
    assert result["RF"]["metrics"] is None
    assert not all_imp.empty
    assert set(all_imp["feature"]) == {"g1", "g2", "g3"}
