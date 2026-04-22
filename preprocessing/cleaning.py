import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DropUselessColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        self.columns_to_drop_ = []
        
        for col in X.columns:

            n_unique = X[col].nunique() 
            

            # 1. Drop constant columns (only 1 unique value)
            if n_unique <= 1:
                self.columns_to_drop_.append(col)
                continue
                
            # 2. Drop ID columns (every row is unique)
            # This is dangerous for numeric IDs which Random Forest can overfit.
            if n_unique == len(X):
                self.columns_to_drop_.append(col)
                continue
                
        # 3. Detect Target Leakage (if y is provided)
        # If a single feature can perfectly predict the target, it's likely a leak.
        if y is not None:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.preprocessing import LabelEncoder
            import numpy as np
            
            y_enc = LabelEncoder().fit_transform(y)
            
            for col in X.columns:
                if col in self.columns_to_drop_:
                    continue
                    
                # Only check numeric columns for simple leakage detection
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Check for nulls, drop them just for the check
                    valid_idx = X[col].notna()
                    if valid_idx.sum() > 0:
                        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                        dt.fit(X.loc[valid_idx, [col]], y_enc[valid_idx])
                        acc = dt.score(X.loc[valid_idx, [col]], y_enc[valid_idx])
                        
                        # If a single feature gives > 99% accuracy on its own, it's a leak
                        if acc > 0.99:
                            self.columns_to_drop_.append(col)

                
        return self

    def transform(self, X):

        cols_to_drop = [c for c in self.columns_to_drop_ if c in X.columns]
        
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def set_output(self, transform=None):
        return self