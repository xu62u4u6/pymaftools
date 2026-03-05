import numpy as np
import pandas as pd

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
