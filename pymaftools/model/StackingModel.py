import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ..core.PivotTable import PivotTable
from ..plot.OncoPlot import OncoPlot

class OmicsStackingModel:
    def __init__(
        self, 
        omics_dict, 
        class_order,
        base_model=RandomForestClassifier, 
        final_model=LogisticRegression,
        random_state=42
    ):
        self.omics_dict = omics_dict
        self.base_model_class = base_model
        self.final_model_class = final_model
        self.model = None
        self.class_order = class_order  
        self.random_state = random_state

        # process labels - use the actual class_order instead of hardcoded classes
        self.le = LabelEncoder()
        self.le.classes_ = np.array(class_order)  # Use the actual class_order
        
        self.build_model()

    def build_model(self):
        estimators = []
        for name, table in self.omics_dict.items():
            model = self.base_model_class(n_estimators=100, random_state=self.random_state)
            selector = ColumnTransformer([(name, 'passthrough', table.index)])
            pipe = Pipeline([('select', selector), ('model', model)])
            estimators.append((name, pipe))

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_model_class(max_iter=1000, random_state=self.random_state),
            cv=5,
            stack_method="predict_proba"
        )

    def encode_y(self, y):
        y_encoded = self.le.transform(y)
        return y_encoded
    
    def decode_y(self, y_encoded):
        y_decoded = self.le.inverse_transform(y_encoded)
        return y_decoded

    def fit(self, X, y):
        y_encoded = self.encode_y(y)
        self.model.fit(X, y_encoded)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return self.decode_y(y_pred)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_omics_feature_importance(self, omics_key)-> pd.Series:
        base = self.model.named_estimators_[omics_key]
        rf = base.named_steps["model"]
        return pd.Series(rf.feature_importances_, index=self.omics_dict[omics_key].index)

    def get_omics_weights(self):
        """
        Return the weights/coefficients of each omics data type in the final stacking model.
        
        Returns:
            pd.DataFrame: Weights for each omics across different classes
        """
        if not hasattr(self.model, 'final_estimator_'):
            raise ValueError("Model must be fitted before getting omics weights")
        
        final_estimator = self.model.final_estimator_
        
        # Get coefficients from final estimator (LogisticRegression)
        if hasattr(final_estimator, 'coef_'):
            coefficients = final_estimator.coef_
            
            # Create DataFrame with omics names and class labels
            omics_names = list(self.omics_dict.keys())
            class_names = self.le.classes_
            
            # Handle binary vs multiclass classification
            if coefficients.shape[0] == 1:  # Binary classification
                # For binary classification, we only get one set of coefficients
                # The coefficients represent class_names[1] vs class_names[0]
                weights_df = pd.DataFrame(
                    coefficients.T,  # Transpose to have omics as rows
                    index=omics_names,
                    columns=[f"{class_names[1]}_vs_{class_names[0]}"]
                )
            else:  # Multiclass classification
                # For multiclass logistic regression, coef_ shape is (n_classes, n_features)
                weights_df = pd.DataFrame(
                    coefficients.T,  # Transpose to have omics as rows
                    index=omics_names,
                    columns=class_names
                )
            
            # Add absolute values for easier interpretation
            weights_df['abs_mean'] = weights_df.abs().mean(axis=1)
            weights_df['abs_ratio'] = weights_df['abs_mean'] / weights_df['abs_mean'].sum()
            return weights_df
        else:
            raise ValueError("Final estimator does not have coefficients (not a linear model)")

    def plot_final_coefficients(self):
        """
        Plot the final coefficients/weights of each omics in the stacking model.
        """
        df = self.get_omics_weights()
        
        # Remove abs_mean column for plotting
        plot_df = df.drop(columns=['abs_mean'])
        
        table = PivotTable(plot_df.T)  # Transpose so classes are features
        table.sample_metadata = pd.DataFrame({'omic': df.index})
        table.sample_metadata.set_index(table.sample_metadata.index, inplace=True)
        table.sample_metadata["abs_mean"] = df['abs_mean']

        (OncoPlot(table)
            .set_config(numeric_columns=["abs_mean"], figsize=(10, 8))
            .numeric_heatmap(annot=True, symmetric=True, cmap="coolwarm")
            .plot_numeric_metadata(annotate=True)
            .add_xticklabel())

    def confusion_matrix(self, y_true, y_pred, title=None):
        cm = confusion_matrix(y_true, y_pred, labels=self.class_order)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=self.class_order, 
                yticklabels=self.class_order,
                cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if title:
            plt.title(title)
        #plt.title('Confusion Matrix (with specified class order)')
        plt.show()

    def evaluate(self, X, y_true, average='macro', show=True):
        """
        輸出分類效能評估指標。
        """
        y_true_encoded = self.encode_y(y_true)
        y_pred_encoded = self.model.predict(X)
        y_pred = self.decode_y(y_pred_encoded)

        # 核心指標
        acc = accuracy_score(y_true_encoded, y_pred_encoded)
        f1 = f1_score(y_true_encoded, y_pred_encoded, average=average)
        prec = precision_score(y_true_encoded, y_pred_encoded, average=average)
        rec = recall_score(y_true_encoded, y_pred_encoded, average=average)

        try:
            proba = self.model.predict_proba(X)
            roc_auc = roc_auc_score(y_true_encoded, proba, multi_class="ovr")
        except:
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
            "roc_auc": roc_auc
        }


class ASCStackingModel(OmicsStackingModel):
    def __init__(self, omics_dict, class_order, random_state=42):
        #class_order = ["LUAD", "ASC", "LUSC"]
        super().__init__(omics_dict, class_order, random_state=random_state)

    def soft_score(self, X):
        y_pred_proba = self.predict_proba(X)
        LUSC_prob = y_pred_proba[:, self.class_order.index("LUSC")]
        return LUSC_prob
