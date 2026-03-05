from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..core.PivotTable import PivotTable
from ..plot.OncoPlot import OncoPlot


class OmicsStackingModel:
    """
    Multi-omics stacking classifier.

    Builds a ``StackingClassifier`` where each base estimator operates on
    a single omics layer, and a final meta-learner combines their
    predictions.

    Parameters
    ----------
    omics_dict : dict[str, PivotTable]
        Mapping of omics names to PivotTable objects (features as index).
    class_order : list[str]
        Ordered class labels used for encoding/decoding.
    base_model : type, default ``RandomForestClassifier``
        Class of the base estimator (instantiated per omics layer).
    final_model : type, default ``LogisticRegression``
        Class of the final meta-learner.
    random_state : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        omics_dict: dict[str, PivotTable],
        class_order: list[str],
        base_model: type = RandomForestClassifier,
        final_model: type = LogisticRegression,
        random_state: int = 42,
    ) -> None:
        self.omics_dict = omics_dict
        self.base_model_class = base_model
        self.final_model_class = final_model
        self.model: StackingClassifier | None = None
        self.class_order = class_order
        self.random_state = random_state

        self.le = LabelEncoder()
        self.le.classes_ = np.array(class_order)

        self.build_model()

    def build_model(self) -> None:
        """Build the stacking classifier from ``omics_dict``."""
        estimators: list[tuple[str, Pipeline]] = []
        for name, table in self.omics_dict.items():
            model = self.base_model_class(
                n_estimators=100, random_state=self.random_state
            )
            selector = ColumnTransformer([(name, "passthrough", table.index)])
            pipe = Pipeline([("select", selector), ("model", model)])
            estimators.append((name, pipe))

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_model_class(
                max_iter=1000, random_state=self.random_state
            ),
            cv=5,
            stack_method="predict_proba",
        )

    def encode_y(self, y: np.ndarray | pd.Series) -> np.ndarray:
        """Encode labels to integer indices using ``class_order``."""
        return self.le.transform(y)

    def decode_y(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer indices back to original labels."""
        return self.le.inverse_transform(y_encoded)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> None:
        """
        Fit the stacking model.

        Parameters
        ----------
        X : pd.DataFrame
            Training data (samples as rows, all omics features as columns).
        y : array-like
            Target labels.
        """
        y_encoded = self.encode_y(y)
        self.model.fit(X, y_encoded)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Decoded class labels.
        """
        y_pred = self.model.predict(X)
        return self.decode_y(y_pred)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        return self.model.predict_proba(X)

    def get_omics_feature_importance(self, omics_key: str) -> pd.Series:
        """
        Get feature importances for a specific omics layer.

        Parameters
        ----------
        omics_key : str
            Key in ``omics_dict`` identifying the omics layer.

        Returns
        -------
        pd.Series
            Feature importances indexed by feature names.
        """
        base = self.model.named_estimators_[omics_key]
        rf = base.named_steps["model"]
        return pd.Series(rf.feature_importances_, index=self.omics_dict[omics_key].index)

    def get_omics_weights(self) -> pd.DataFrame:
        """
        Return the weights of each omics layer in the final meta-learner.

        Returns
        -------
        pd.DataFrame
            Weights with omics as rows. Includes ``abs_mean`` and
            ``abs_ratio`` columns for interpretability.

        Raises
        ------
        ValueError
            If the model has not been fitted or the final estimator
            does not expose ``coef_``.
        """
        if not hasattr(self.model, "final_estimator_"):
            raise ValueError("Model must be fitted before getting omics weights.")

        final_estimator = self.model.final_estimator_

        if not hasattr(final_estimator, "coef_"):
            raise ValueError(
                "Final estimator does not have coefficients (not a linear model)."
            )

        coefficients = final_estimator.coef_
        omics_names = list(self.omics_dict.keys())
        class_names = self.le.classes_

        if coefficients.shape[0] == 1:
            # Binary classification
            weights_df = pd.DataFrame(
                coefficients.T,
                index=omics_names,
                columns=[f"{class_names[1]}_vs_{class_names[0]}"],
            )
        else:
            # Multiclass classification
            weights_df = pd.DataFrame(
                coefficients.T,
                index=omics_names,
                columns=class_names,
            )

        weights_df["abs_mean"] = weights_df.abs().mean(axis=1)
        weights_df["abs_ratio"] = weights_df["abs_mean"] / weights_df["abs_mean"].sum()
        return weights_df

    def plot_final_coefficients(self) -> None:
        """Plot the final meta-learner coefficients as a heatmap."""
        df = self.get_omics_weights()

        plot_df = df.drop(columns=["abs_mean"])

        table = PivotTable(plot_df.T)
        table.sample_metadata = pd.DataFrame({"omic": df.index})
        table.sample_metadata.set_index(table.sample_metadata.index, inplace=True)
        table.sample_metadata["abs_mean"] = df["abs_mean"]

        (
            OncoPlot(table)
            .set_config(numeric_columns=["abs_mean"], figsize=(10, 8))
            .numeric_heatmap(annot=True, symmetric=True, cmap="coolwarm")
            .plot_numeric_metadata(annotate=True)
            .add_xticklabel()
        )

    def confusion_matrix(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        title: str | None = None,
    ) -> None:
        """
        Plot a confusion matrix heatmap.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        title : str, optional
            Plot title.
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.class_order)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self.class_order,
            yticklabels=self.class_order,
            cmap="Blues",
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if title:
            plt.title(title)
        plt.show()

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        average: str = "macro",
        show: bool = True,
    ) -> dict[str, float | None]:
        """
        Evaluate classification performance.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y_true : array-like
            True labels.
        average : str, default ``"macro"``
            Averaging strategy for multi-class metrics.
        show : bool, default True
            Whether to print the metrics.

        Returns
        -------
        dict[str, float | None]
            Dictionary with keys ``accuracy``, ``f1``, ``precision``,
            ``recall``, and ``roc_auc``.
        """
        y_true_encoded = self.encode_y(y_true)
        y_pred_encoded = self.model.predict(X)

        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        f1 = f1_score(y_true_encoded, y_pred_encoded, average=average)
        prec = precision_score(y_true_encoded, y_pred_encoded, average=average)
        rec = recall_score(y_true_encoded, y_pred_encoded, average=average)

        try:
            proba = self.model.predict_proba(X)
            roc_auc = roc_auc_score(y_true_encoded, proba, multi_class="ovr")
        except (ValueError, TypeError):
            roc_auc = None

        if show:
            print(f"Accuracy     : {acc:.4f}")
            print(f"F1-score     : {f1:.4f}")
            print(f"Precision    : {prec:.4f}")
            print(f"Recall       : {rec:.4f}")
            if roc_auc is not None:
                print(f"ROC-AUC (ovr): {roc_auc:.4f}")

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": roc_auc,
        }


class ASCStackingModel(OmicsStackingModel):
    """
    Stacking model pre-configured for ASC (adenosquamous carcinoma) analysis.

    Parameters
    ----------
    omics_dict : dict[str, PivotTable]
        Mapping of omics names to PivotTable objects.
    class_order : list[str]
        Ordered class labels.
    random_state : int, default 42
        Random seed.
    """

    def __init__(
        self,
        omics_dict: dict[str, PivotTable],
        class_order: list[str],
        random_state: int = 42,
    ) -> None:
        super().__init__(omics_dict, class_order, random_state=random_state)

    def soft_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the LUSC probability score for each sample.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            LUSC class probability for each sample.
        """
        y_pred_proba = self.predict_proba(X)
        LUSC_prob = y_pred_proba[:, self.class_order.index("LUSC")]
        return LUSC_prob
