import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import pytest

from pymaftools.core.PivotTable import PivotTable
from pymaftools.model.StackingModel import OmicsStackingModel


def _build_model_inputs(n_samples: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    sample_ids = [f"s{i}" for i in range(n_samples)]

    snv_features = [f"snv_f{i}" for i in range(5)]
    cnv_features = [f"cnv_f{i}" for i in range(5)]

    snv_table = PivotTable(
        pd.DataFrame(
            rng.normal(size=(len(snv_features), n_samples)),
            index=snv_features,
            columns=sample_ids,
        )
    )
    cnv_table = PivotTable(
        pd.DataFrame(
            rng.normal(size=(len(cnv_features), n_samples)),
            index=cnv_features,
            columns=sample_ids,
        )
    )

    X = pd.DataFrame(
        {
            **{f: snv_table.loc[f].values for f in snv_features},
            **{f: cnv_table.loc[f].values for f in cnv_features},
        },
        index=sample_ids,
    )
    y = np.array(["LUAD"] * (n_samples // 2) + ["LUSC"] * (n_samples // 2))
    return {"SNV": snv_table, "CNV": cnv_table}, X, y


def test_fit_predict_predict_proba_and_evaluate():
    omics_dict, X, y = _build_model_inputs()
    model = OmicsStackingModel(omics_dict=omics_dict, class_order=["LUAD", "LUSC"])

    model.fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    metrics = model.evaluate(X, y, show=False)

    assert pred.shape == (len(y),)
    assert set(pred).issubset({"LUAD", "LUSC"})
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)), atol=1e-6)
    assert {"accuracy", "f1", "precision", "recall", "roc_auc"}.issubset(metrics.keys())
    assert metrics["roc_auc"] is not None

    weights = model.get_omics_weights()
    assert list(weights.index) == ["SNV", "CNV"]
    assert "LUSC_vs_LUAD" in weights.columns
    assert weights["abs_ratio"].sum() == pytest.approx(1.0)


def test_class_order_controls_encoding_instead_of_lexical_order():
    omics_dict, _, _ = _build_model_inputs()
    model = OmicsStackingModel(omics_dict=omics_dict, class_order=["LUSC", "LUAD"])

    encoded = model.encode_y(np.array(["LUSC", "LUAD", "LUSC"]))

    np.testing.assert_array_equal(encoded, [0, 1, 0])
    np.testing.assert_array_equal(model.decode_y(encoded), ["LUSC", "LUAD", "LUSC"])


def test_multiclass_omics_weights_aggregate_probability_blocks():
    omics_dict, X, _ = _build_model_inputs(n_samples=30)
    y = np.array(["LUAD", "LUSC", "ASC"] * 10)
    model = OmicsStackingModel(
        omics_dict=omics_dict,
        class_order=["LUAD", "LUSC", "ASC"],
    )
    model.fit(X, y)

    weights = model.get_omics_weights()
    metrics = model.evaluate(X, y, show=False)

    assert weights.shape == (2, 5)
    assert list(weights.columns[:3]) == ["LUAD", "LUSC", "ASC"]
    assert (weights.iloc[:, :3] >= 0).all().all()
    assert weights["abs_ratio"].sum() == pytest.approx(1.0)
    assert metrics["roc_auc"] is not None


def test_get_omics_feature_importance_returns_series_per_omics():
    omics_dict, X, y = _build_model_inputs()
    model = OmicsStackingModel(omics_dict=omics_dict, class_order=["LUAD", "LUSC"])
    model.fit(X, y)

    snv_imp = model.get_omics_feature_importance("SNV")
    cnv_imp = model.get_omics_feature_importance("CNV")

    assert isinstance(snv_imp, pd.Series)
    assert isinstance(cnv_imp, pd.Series)
    assert len(snv_imp) == len(omics_dict["SNV"].index)
    assert len(cnv_imp) == len(omics_dict["CNV"].index)


def test_prepare_features_namespaces_overlapping_genes():
    sample_ids = [f"s{i}" for i in range(20)]
    mutation = PivotTable(
        pd.DataFrame([range(20), range(20)], index=["TP53", "KRAS"], columns=sample_ids)
    )
    expression = PivotTable(
        pd.DataFrame([range(20), range(20)], index=["TP53", "EGFR"], columns=sample_ids)
    )
    model = OmicsStackingModel(
        {"mutation": mutation, "expression": expression},
        class_order=["A", "B"],
    )

    X = model.prepare_features()

    assert list(X.columns) == ["mutation::TP53", "KRAS", "expression::TP53", "EGFR"]
    model.fit(X, np.array(["A"] * 10 + ["B"] * 10))
    assert len(model.predict(X)) == 20


def test_feature_selection_is_fitted_inside_stacking_and_reported():
    omics_dict, X, y = _build_model_inputs(n_samples=30)
    model = OmicsStackingModel(
        omics_dict,
        class_order=["LUAD", "LUSC"],
        max_features=2,
        cv=3,
    )

    model.fit(X, y)

    selected = model.get_selected_features()
    assert set(selected) == {"SNV", "CNV"}
    assert all(len(features) == 2 for features in selected.values())
    assert len(model.get_omics_feature_importance("SNV")) == 2


def test_prepare_features_requires_exact_sample_universe_by_default():
    sample_ids = [f"s{i}" for i in range(4)]
    first = PivotTable(
        pd.DataFrame([[1, 2, 3, 4]], index=["f"], columns=sample_ids)
    )
    second = PivotTable(
        pd.DataFrame([[4, 5, 6]], index=["g"], columns=sample_ids[1:])
    )

    strict = OmicsStackingModel(
        {"first": first, "second": second}, class_order=["A", "B"]
    )
    with pytest.raises(ValueError, match="do not exactly match"):
        strict.prepare_features()

    legacy = OmicsStackingModel(
        {"first": first, "second": second},
        class_order=["A", "B"],
        sample_policy="intersection",
    )
    assert legacy.prepare_features().index.tolist() == sample_ids[1:]


def test_custom_estimators_do_not_require_random_forest_constructor_kwargs():
    class MinimalClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self._model = LogisticRegression()

        def fit(self, X, y):
            self._model.fit(X, y)
            self.classes_ = self._model.classes_
            return self

        def predict(self, X):
            return self._model.predict(X)

        def predict_proba(self, X):
            return self._model.predict_proba(X)

    omics_dict, X, y = _build_model_inputs(n_samples=30)
    model = OmicsStackingModel(
        omics_dict,
        class_order=["LUAD", "LUSC"],
        base_model=MinimalClassifier,
        cv=3,
    )

    model.fit(X, y)
    assert len(model.predict(X)) == len(y)
