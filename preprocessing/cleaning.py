import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropUselessColumns(BaseEstimator, TransformerMixin):
    def __init__(self, task=None):
        self.columns_to_drop_ = []
        self.task = task

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        self.columns_to_drop_ = []
        
        for col in X.columns:
            n_unique = X[col].nunique() 

            if n_unique <= 1:
                self.columns_to_drop_.append(col)
                continue
                

            is_highly_unique = (n_unique / len(X)) > 0.95
            is_unnamed = 'unnamed' in str(col).lower()
            
            if is_highly_unique or is_unnamed:
                self.columns_to_drop_.append(col)
                continue

        if y is not None:
            is_classification = True
            if self.task is not None:
                is_classification = str(self.task).lower() == 'classification'
            elif pd.api.types.is_numeric_dtype(y):
                is_classification = len(np.unique(y)) < 50
                
            if is_classification:
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.preprocessing import LabelEncoder
                y_target = LabelEncoder().fit_transform(y)
                model_cls = DecisionTreeClassifier
            else:
                from sklearn.tree import DecisionTreeRegressor
                y_target = y
                model_cls = DecisionTreeRegressor
            
            for col in X.columns:
                if col in self.columns_to_drop_:
                    continue
                    
                if pd.api.types.is_numeric_dtype(X[col]):
                    valid_idx = X[col].notna()
                    if valid_idx.sum() > 0:
                        dt = model_cls(max_depth=1, random_state=42)
                        dt.fit(X.loc[valid_idx, [col]], y_target[valid_idx])
                        score = dt.score(X.loc[valid_idx, [col]], y_target[valid_idx])
                        
                        if score > 0.99:
                            self.columns_to_drop_.append(col)

        return self

    def transform(self, X):
        cols_to_drop = [c for c in self.columns_to_drop_ if c in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array([f for f in input_features if f not in self.columns_to_drop_], dtype=object)

    def set_output(self, transform=None):
        return self