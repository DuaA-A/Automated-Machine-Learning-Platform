import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DropUselessColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        self.columns_to_drop_ = []
        
        for col in X.columns:

            n_unique = X[col].nunique() 
            

            if n_unique <= 1:
                self.columns_to_drop_.append(col)
                continue
                
            is_string = X[col].dtype in ['object', 'string', 'category']
            if is_string and n_unique == len(X):
                self.columns_to_drop_.append(col)
                
        return self

    def transform(self, X):

        cols_to_drop = [c for c in self.columns_to_drop_ if c in X.columns]
        
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X

    def set_output(self, transform=None):
        return self