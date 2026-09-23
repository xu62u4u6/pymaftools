from __future__ import annotations

from collections import Counter
import inspect
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ..core.PivotTable import PivotTable
from ..plot.OncoPlot import OncoPlot


class _TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Select a deterministic top-variance feature set inside each CV split."""

    def __init__(self, max_features: int | None = None) -> None:
        self.max_features = max_features

    def fit(self, X: Any, y: Any = None) -> "_TopVarianceSelector":
        del y
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("Feature selector expects a two-dimensional matrix.")
        if values.shape[1] == 0:
            raise ValueError("Feature selector received no features.")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be positive or None.")

        variances = np.nanvar(values, axis=0)
        finite = np.isfinite(variances)
        order = np.lexsort((np.arange(values.shape[1]), -variances))
        order = order[finite[order]]
        limit = values.shape[1] if self.max_features is None else self.max_features
        selected = order[: min(limit, len(order))]
        if selected.size == 0 or not (variances[selected] > 0).any():
            raise ValueError("Feature selector found no finite variable features.")

        support = np.zeros(values.shape[1], dtype=bool)
        support[selected] = True
        self.support_ = support
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise ValueError("Feature selector must be fitted before transform.")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                "Feature selector received a matrix with an unexpected number "
                "of features."
            )
        if not np.isfinite(values).all():
            raise ValueError("Feature selector received missing or infinite values.")
        return values[:, self.support_]

    def get_support(self) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise ValueError("Feature selector must be fitted before get_support.")
        return self.support_.copy()


class _LayerVarianceSelector(BaseEstimator, TransformerMixin):
    """Select one omics layer without materializing unrelated columns."""

    def __init__(self, feature_indices: list[int], max_features: int | None) -> None:
        self.feature_indices = feature_indices
        self.max_features = max_features

    def _layer_values(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if max(self.feature_indices, default=-1) >= X.shape[1]:
                raise ValueError("Layer selector received an unexpected feature matrix.")
            return X.iloc[:, self.feature_indices].to_numpy(dtype=float, copy=True)
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or max(self.feature_indices, default=-1) >= values.shape[1]:
            raise ValueError("Layer selector received an unexpected feature matrix.")
        return values[:, self.feature_indices]

    def fit(self, X: Any, y: Any = None) -> "_LayerVarianceSelector":
        values = self._layer_values(X)
        self._selector = _TopVarianceSelector(self.max_features).fit(values, y)
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_selector"):
            raise ValueError("Layer selector must be fitted before transform.")
        if X.shape[1] != self.n_input_features_:
            raise ValueError("Layer selector received an unexpected feature matrix.")
        values = self._layer_values(X)
        return self._selector.transform(values)

    def get_support(self) -> np.ndarray:
        if not hasattr(self, "_selector"):
            raise ValueError("Layer selector must be fitted before get_support.")
        return self._selector.get_support()


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
    max_features : int, optional
        Maximum number of features retained per omics layer by a selector that
        is fitted independently inside each stacking CV split. ``None`` keeps
        every feature.
    cv : int or cross-validation splitter, default 5
        Cross-validation strategy used to generate the base-estimator
        out-of-fold predictions. A precomputed splitter such as
        ``PredefinedSplit`` can encode patient-grouped folds without relying
        on estimator metadata routing.
    sample_policy : {``"exact"``, ``"intersection"``}, default ``"exact"``
        How sample identifiers are reconciled across layers. ``"exact"``
        fails when any layer has a missing or extra sample. ``"intersection"``
        is an explicit opt-in for legacy inner-join behavior.
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
        max_features: int | None = None,
        cv: int | Any = 5,
        sample_policy: Literal["exact", "intersection"] = "exact",
    ) -> None:
        if not class_order or len(set(class_order)) != len(class_order):
            raise ValueError("class_order must contain at least two unique labels.")
        if len(class_order) < 2:
            raise ValueError("class_order must contain at least two unique labels.")
        if sample_policy not in {"exact", "intersection"}:
            raise ValueError("sample_policy must be 'exact' or 'intersection'.")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None.")

        self.omics_dict = omics_dict
        self.base_model_class = base_model
        self.final_model_class = final_model
        self.model: StackingClassifier | None = None
        self.class_order = class_order
        self.random_state = random_state
        self.max_features = max_features
        self.cv = cv
        self.sample_policy = sample_policy

        duplicate_features = {
            feature
            for feature, count in Counter(
                feature for table in omics_dict.values() for feature in table.index
            ).items()
            if count > 1
        }
        self.feature_columns: dict[str, list[str]] = {}
        for name, table in omics_dict.items():
            if not table.index.is_unique:
                raise ValueError(f"Omics table '{name}' has duplicate feature names.")
            self.feature_columns[name] = [
                f"{name}::{feature}" if feature in duplicate_features else feature
                for feature in table.index
            ]

        self.le = LabelEncoder()
        self.le.classes_ = np.array(class_order)

        self.build_model()

    def prepare_features(self) -> pd.DataFrame:
        """Build an aligned samples-by-features matrix from the omics tables.

        By default, all omics layers must expose the exact same sample
        identifiers and order is taken from the first layer. Set
        ``sample_policy="intersection"`` to explicitly request legacy
        inner-join behavior. Feature names shared by multiple layers are
        prefixed with ``<omics>::`` so sklearn selectors remain unambiguous.
        """
        if not self.omics_dict:
            raise ValueError("omics_dict must contain at least one table.")

        tables = list(self.omics_dict.values())
        for name, table in self.omics_dict.items():
            if not table.columns.is_unique:
                raise ValueError(f"Omics table '{name}' has duplicate sample IDs.")

        sample_index = tables[0].columns
        if self.sample_policy == "exact":
            for name, table in list(self.omics_dict.items())[1:]:
                if not sample_index.equals(table.columns):
                    missing = sample_index.difference(table.columns).tolist()
                    extra = table.columns.difference(sample_index).tolist()
                    raise ValueError(
                        f"Omics table '{name}' sample IDs do not exactly match "
                        f"the first table (missing={missing}, extra={extra})."
                    )
        else:
            for table in tables[1:]:
                sample_index = sample_index[sample_index.isin(table.columns)]
        if sample_index.empty:
            raise ValueError("Omics tables do not share any sample identifiers.")

        frames = []
        for name, table in self.omics_dict.items():
            frame = pd.DataFrame(table).T.reindex(sample_index)
            frame.columns = self.feature_columns[name]
            frames.append(frame)
        return pd.concat(frames, axis=1)

    def build_model(self) -> None:
        """Build the stacking classifier from ``omics_dict``."""
        estimators: list[tuple[str, Pipeline]] = []
        feature_position = {
            feature: position
            for position, feature in enumerate(self._expected_feature_columns())
        }
        for name, table in self.omics_dict.items():
            model = self._instantiate_estimator(
                self.base_model_class,
                n_estimators=100,
                random_state=self.random_state,
            )
            selector = _LayerVarianceSelector(
                [feature_position[feature] for feature in self.feature_columns[name]],
                self.max_features,
            )
            pipe = Pipeline(
                [
                    ("variance", selector),
                    ("model", model),
                ]
            )
            estimators.append((name, pipe))

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self._instantiate_estimator(
                self.final_model_class,
                max_iter=1000, random_state=self.random_state
            ),
            cv=self.cv,
            stack_method="predict_proba",
        )

    @staticmethod
    def _instantiate_estimator(
        estimator_class: type,
        **candidate_kwargs: Any,
    ) -> Any:
        """Instantiate custom sklearn classes without assuming RF/linear kwargs."""
        try:
            parameters = inspect.signature(estimator_class).parameters
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Estimator classes must expose an inspectable constructor."
            ) from exc
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs = {
            name: value
            for name, value in candidate_kwargs.items()
            if accepts_kwargs or name in parameters
        }
        return estimator_class(**kwargs)

    def _expected_feature_columns(self) -> list[str]:
        return [
            feature
            for name in self.omics_dict
            for feature in self.feature_columns[name]
        ]

    def _validate_input_features(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not X.index.is_unique:
            raise ValueError("X must have unique sample IDs as its index.")
        expected = self._expected_feature_columns()
        if list(X.columns) != expected:
            raise ValueError(
                "X columns must match model.prepare_features() exactly, including "
                "order and namespacing."
            )

    def _require_fitted(self) -> StackingClassifier:
        if self.model is None or not hasattr(self.model, "named_estimators_"):
            raise ValueError("Model must be fitted before this operation.")
        return self.model

    def encode_y(self, y: np.ndarray | pd.Series) -> np.ndarray:
        """Encode labels to integer indices using ``class_order``."""
        class_to_index = {label: index for index, label in enumerate(self.class_order)}
        labels = np.asarray(y)
        unknown = set(labels) - set(class_to_index)
        if unknown:
            raise ValueError(f"Unknown class labels: {sorted(unknown)}")
        return np.asarray([class_to_index[label] for label in labels], dtype=int)

    def decode_y(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer indices back to original labels."""
        encoded = np.asarray(y_encoded, dtype=int)
        if ((encoded < 0) | (encoded >= len(self.class_order))).any():
            raise ValueError("Encoded class index is outside class_order")
        return np.asarray(self.class_order)[encoded]

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
        self._validate_input_features(X)
        if len(y) != len(X):
            raise ValueError("X and y must contain the same number of samples.")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("X and y sample indices must match exactly.")
        y_encoded = self.encode_y(y)
        self._require_model().fit(X, y_encoded)

    def _require_model(self) -> StackingClassifier:
        if self.model is None:
            raise ValueError("Model has not been built.")
        return self.model

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
        self._validate_input_features(X)
        y_pred = self._require_fitted().predict(X)
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
        self._validate_input_features(X)
        return self._require_fitted().predict_proba(X)

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
        model = self._require_fitted()
        if omics_key not in self.omics_dict:
            raise KeyError(f"Unknown omics layer: {omics_key}")
        base = model.named_estimators_[omics_key]
        rf = base.named_steps["model"]
        selected = self.omics_dict[omics_key].index[
            base.named_steps["variance"].get_support()
        ]
        return pd.Series(rf.feature_importances_, index=selected)

    def get_selected_features(self) -> dict[str, pd.Index]:
        """Return features selected by each fitted layer on the full fit set."""
        model = self._require_fitted()
        selected: dict[str, pd.Index] = {}
        for name, table in self.omics_dict.items():
            layer_pipeline = model.named_estimators_[name]
            selected[name] = table.index[
                layer_pipeline.named_steps["variance"].get_support()
            ]
        return selected

    def get_omics_weights(self) -> pd.DataFrame:
        """
        Return the weights of each omics layer in the final meta-learner.

        Returns
        -------
        pd.DataFrame
            Weights with omics as rows. Binary weights retain their sign.
            Multiclass weights are the L2 norm of each omics probability
            coefficient block for each target class. Includes ``abs_mean``
            and ``abs_ratio`` columns for interpretability.

        Raises
        ------
        ValueError
            If the model has not been fitted or the final estimator
            does not expose ``coef_``.
        """
        model = self._require_fitted()
        if not hasattr(model, "final_estimator_"):
            raise ValueError("Model must be fitted before getting omics weights.")

        final_estimator = model.final_estimator_

        if not hasattr(final_estimator, "coef_"):
            raise ValueError(
                "Final estimator does not have coefficients (not a linear model)."
            )

        coefficients = final_estimator.coef_
        omics_names = list(self.omics_dict.keys())
        class_names = self.le.classes_

        if coefficients.shape[0] == 1:
            # Binary classification
            if coefficients.shape[1] != len(omics_names):
                raise ValueError(
                    "Unexpected binary meta-feature count: "
                    f"expected {len(omics_names)}, got {coefficients.shape[1]}"
                )
            weights_df = pd.DataFrame(
                coefficients.T,
                index=omics_names,
                columns=[f"{class_names[1]}_vs_{class_names[0]}"],
            )
        else:
            n_classes = len(class_names)
            expected_features = len(omics_names) * n_classes
            if coefficients.shape != (n_classes, expected_features):
                raise ValueError(
                    "Unexpected multiclass coefficient shape: "
                    f"expected {(n_classes, expected_features)}, "
                    f"got {coefficients.shape}"
                )
            coefficient_blocks = coefficients.reshape(
                n_classes, len(omics_names), n_classes
            )
            weights_df = pd.DataFrame(
                np.linalg.norm(coefficient_blocks, axis=2).T,
                index=omics_names,
                columns=class_names,
            )

        weights_df["abs_mean"] = weights_df.abs().mean(axis=1)
        abs_total = weights_df["abs_mean"].sum()
        weights_df["abs_ratio"] = (
            weights_df["abs_mean"] / abs_total if abs_total else 0.0
        )
        return weights_df

    def plot_final_coefficients(self) -> None:
        """Plot the final meta-learner coefficients as a heatmap."""
        df = self.get_omics_weights()

        plot_df = df.drop(columns=["abs_mean", "abs_ratio"])

        table = PivotTable(plot_df.T)
        table.sample_metadata = pd.DataFrame({"omic": df.index})
        table.sample_metadata.set_index(table.sample_metadata.index, inplace=True)
        table.sample_metadata["abs_mean"] = df["abs_mean"]

        (
            OncoPlot(table)
            .set_config(numeric_columns=["abs_mean"], figsize=(10, 8))
            .numeric_heatmap(annot=True, symmetric=True, cmap="coolwarm")
            .plot_numeric_metadata(annotate=True)
            .render()
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
        self._validate_input_features(X)
        y_true_encoded = self.encode_y(y_true)
        model = self._require_fitted()
        y_pred_encoded = model.predict(X)

        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        f1 = f1_score(y_true_encoded, y_pred_encoded, average=average)
        prec = precision_score(y_true_encoded, y_pred_encoded, average=average)
        rec = recall_score(y_true_encoded, y_pred_encoded, average=average)

        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_true_encoded, proba[:, 1])
            else:
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
