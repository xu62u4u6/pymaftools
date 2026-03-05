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
    assert cca_snv_new.shape == (snv.shape[1], 2)
    assert cca_cnv_new.shape == (cnv.shape[1], 2)
    assert "CCA_SNV_comp1" in w_snv.columns
    assert "CCA_CNV_comp1" in w_cnv.columns
    assert "abs_weight_mean" in w_snv.columns
    assert "abs_weight_mean" in w_cnv.columns


def test_get_weights_raises_before_fit():
    reducer = PCA_CCA(n_pca_components=2, n_cca_components=1)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        reducer.get_weights()
