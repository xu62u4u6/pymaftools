from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import pandas as pd

class PCA_CCA:
    """
    Perform PCA on SNV and CNV tables separately, followed by Canonical Correlation Analysis (CCA).

    This class provides utilities to project two omics data tables (SNV and CNV)
    into a shared latent space using PCA for dimensionality reduction and CCA for
    capturing cross-omics correlations. It also allows mapping the canonical
    weights back to the original feature space for interpretation.

    Parameters
    ----------
    n_pca_components : int, default=20
        Number of PCA components to retain for each omics table.
    n_cca_components : int, default=1
        Number of canonical components for CCA.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(self, n_pca_components=20, n_cca_components=1, random_state=42):
        self.pca_model_snv = PCA(n_components=n_pca_components, random_state=random_state)
        self.pca_model_cnv = PCA(n_components=n_pca_components, random_state=random_state)
        self.cca_model = CCA(n_components=n_cca_components)

        self.features_snv = None
        self.features_cnv = None

    def fit_transform(self, snv_table: pd.DataFrame, cnv_table: pd.DataFrame):
        """
        Fit PCA on SNV and CNV tables, then fit CCA on the reduced embeddings.

        Parameters
        ----------
        snv_table : pd.DataFrame
            SNV data table with features as rows and samples as columns.
        cnv_table : pd.DataFrame
            CNV data table with features as rows and samples as columns.

        Returns
        -------
        cca_snv : ndarray of shape (n_samples, n_cca_components)
            Canonical variates for SNV data.
        cca_cnv : ndarray of shape (n_samples, n_cca_components)
            Canonical variates for CNV data.
        """
        self.features_snv = list(snv_table.index)
        self.features_cnv = list(cnv_table.index)

        snv = pd.DataFrame(snv_table).T
        cnv = pd.DataFrame(cnv_table).T

        pca_snv = self.pca_model_snv.fit_transform(snv)
        pca_cnv = self.pca_model_cnv.fit_transform(cnv)

        cca_snv, cca_cnv = self.cca_model.fit_transform(pca_snv, pca_cnv)
        return cca_snv, cca_cnv

    def transform(self, snv_table: pd.DataFrame, cnv_table: pd.DataFrame):
        """
        Project new SNV and CNV data into the canonical space
        using fitted PCA and CCA models.

        Parameters
        ----------
        snv_table : pd.DataFrame
            SNV data table with features as rows and samples as columns.
        cnv_table : pd.DataFrame
            CNV data table with features as rows and samples as columns.

        Returns
        -------
        cca_snv : ndarray of shape (n_samples, n_cca_components)
            Canonical variates for SNV data.
        cca_cnv : ndarray of shape (n_samples, n_cca_components)
            Canonical variates for CNV data.
        """
        snv = pd.DataFrame(snv_table).T
        cnv = pd.DataFrame(cnv_table).T

        pca_snv = self.pca_model_snv.transform(snv)
        pca_cnv = self.pca_model_cnv.transform(cnv)

        cca_snv, cca_cnv = self.cca_model.transform(pca_snv, pca_cnv)
        return cca_snv, cca_cnv

    def get_weights(self):
        """
        Retrieve feature weights in the canonical variates.

        This method back-projects the CCA weights from the PCA-reduced space
        into the original feature space, enabling interpretation of which features
        contribute most to the canonical correlation.

        Returns
        -------
        df_snv : pd.DataFrame
            DataFrame of SNV feature weights in canonical components.
        df_cnv : pd.DataFrame
            DataFrame of CNV feature weights in canonical components.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.features_snv is None or self.features_cnv is None:
            raise ValueError("Model has not been fitted yet.")

        loadings_snv = self.pca_model_snv.components_.T   # (n_features, n_pca)
        loadings_cnv = self.pca_model_cnv.components_.T

        cca_w_snv = self.cca_model.x_weights_             # (n_pca, n_cca)
        cca_w_cnv = self.cca_model.y_weights_

        feature_w_snv = loadings_snv @ cca_w_snv
        feature_w_cnv = loadings_cnv @ cca_w_cnv

        df_snv = pd.DataFrame(
            feature_w_snv,
            index=self.features_snv,
            columns=[f"CCA_SNV_comp{i+1}" for i in range(feature_w_snv.shape[1])]
        )

        df_cnv = pd.DataFrame(
            feature_w_cnv,
            index=self.features_cnv,
            columns=[f"CCA_CNV_comp{i+1}" for i in range(feature_w_cnv.shape[1])]
        )
        
        def _process_weights(df: pd.DataFrame) -> pd.DataFrame:
            """Helper to add abs/mean weights and return sorted DataFrame."""
            df = df.copy()
            df["abs_weight_mean"] = df.abs().mean(axis=1)
            return df.sort_values(by="abs_weight_mean", ascending=False)

        return _process_weights(df_snv), _process_weights(df_cnv)