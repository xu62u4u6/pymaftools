import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns

from ..core.PivotTable import PivotTable

class OmicsStackingModel:
    def __init__(
        self, 
        omics_dict, 
        base_model=RandomForestClassifier, 
        final_model=LogisticRegression
    ):
        self.omics_dict = omics_dict
        self.base_model_class = base_model
        self.final_model_class = final_model
        self.model = None
        self.class_order = None

    def build_model(self):
        estimators = []
        for name, table in self.omics_dict.items():
            model = self.base_model_class(n_estimators=100, random_state=42)
            selector = ColumnTransformer([(name, 'passthrough', table.index)])
            pipe = Pipeline([('select', selector), ('model', model)])
            estimators.append((name, pipe))
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_model_class(max_iter=1000, random_state=42),
            cv=5,
            stack_method="predict_proba"
        )

    def fit(self, X, y):
        self.build_model()
        self.model.fit(X, y)
        self.class_order = self.model.classes_

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self, omics_key):
        base = self.model.named_estimators_[omics_key]
        rf = base.named_steps["model"]
        return pd.Series(rf.feature_importances_, index=self.omics_dict[omics_key].index)

    def get_final_coefficients(self):
        if not hasattr(self.model.final_estimator_, 'coef_'):
            raise ValueError("Final estimator has no coef_ (e.g., not linear)")
        coef = self.model.final_estimator_.coef_
        colnames = []
        for omics_name in self.omics_dict:
            for c in self.class_order:
                colnames.append(f"{omics_name.upper()}_{c}")
        return pd.DataFrame(coef, index=[f"predict_{c}" for c in self.class_order], columns=colnames)
    