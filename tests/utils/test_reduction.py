import numpy as np
import pandas as pd
import pytest

from pymaftools.utils.reduction import PCA_CCA


def _build_tables(n_features: int = 6, n_samples: int = 12, seed: int = 11):
    rng = np.random.default_rng(seed)
    samples = [f"s{i}" for i in range(n_samples)]
    snv = pd.DataFrame(
        rng.normal(size=(n_features, n_samples)),
        index=[f"snv_g{i}" for i in range(n_features)],
        columns=samples,
    )
    cnv = pd.DataFrame(
        rng.normal(size=(n_features, n_samples)),
        index=[f"cnv_g{i}" for i in range(n_features)],
        columns=samples,
    )
    return snv, cnv


def test_fit_transform_transform_and_get_weights():
    snv, cnv = _build_tables()
    reducer = PCA_CCA(n_pca_components=3, n_cca_components=2, random_state=42)

    cca_snv, cca_cnv = reducer.fit_transform(snv, cnv)
    cca_snv_new, cca_cnv_new = reducer.transform(snv, cnv)
    w_snv, w_cnv = reducer.get_weights()

    assert cca_snv.shape == (snv.shape[1], 2)
    assert cca_cnv.shape == (cnv.shape[1], 2)
    assert np.allclose(cca_snv, cca_snv_new)
    assert np.allclose(cca_cnv, cca_cnv_new)
    assert "CCA_SNV_comp1" in w_snv.columns
    assert "CCA_CNV_comp1" in w_cnv.columns
    assert "abs_weight_mean" in w_snv.columns
    assert "abs_weight_mean" in w_cnv.columns
    assert w_snv["abs_weight_mean"].is_monotonic_decreasing
    assert w_cnv["abs_weight_mean"].is_monotonic_decreasing


def test_get_weights_raises_before_fit():
    reducer = PCA_CCA(n_pca_components=2, n_cca_components=1)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        reducer.get_weights()


def test_get_weights_reproduce_canonical_scores_from_centered_features():
    rng = np.random.default_rng(42)
    samples = [f"S{i}" for i in range(30)]
    snv_values = rng.normal(size=(10, 30))
    snv_values[0] *= 20.0  # unequal PCA score variances make 1/std matter
    snv_values[1] *= 5.0
    cnv_values = 0.7 * snv_values + 0.3 * rng.normal(size=(10, 30))
    snv = pd.DataFrame(snv_values, index=[f"G{i}" for i in range(10)], columns=samples)
    cnv = pd.DataFrame(cnv_values, index=[f"C{i}" for i in range(10)], columns=samples)

    model = PCA_CCA(n_pca_components=5, n_cca_components=3)
    snv_scores, cnv_scores = model.fit_transform(snv, cnv)
    snv_weights, cnv_weights = model.get_weights()

    for table, scores, weights in [(snv, snv_scores, snv_weights), (cnv, cnv_scores, cnv_weights)]:
        components = [c for c in weights.columns if c != "abs_weight_mean"]
        centered = table.T.values - table.T.values.mean(axis=0)
        projected = centered @ weights.loc[table.index, components].values
        np.testing.assert_allclose(projected, np.asarray(scores), atol=1e-8)
