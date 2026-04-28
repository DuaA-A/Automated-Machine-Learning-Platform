import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            counts = X_df[col].value_counts(normalize=True)
            self.mapping_[col] = counts.to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            mapping = self.mapping_.get(col, {})

            X_df[col] = X_df[col].map(mapping).fillna(0)
        return X_df
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return np.array(list(self.mapping_.keys()), dtype=object)

def get_encoder(X: pd.DataFrame, threshold: int = 15):

    cat_cols = X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

    if not cat_cols:
        return FunctionTransformer(lambda x: x, feature_names_out='one-to-one') 

    ohe_cols = [c for c in cat_cols if X[c].nunique() <= threshold]
    freq_cols = [c for c in cat_cols if X[c].nunique() > threshold]


    def to_string(df):
        df = df.copy()
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    transformers = []
    
    if ohe_cols:
        transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols))
    
    if freq_cols:
        transformers.append(('freq', FrequencyEncoder(), freq_cols))

    ct = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    ct.set_output(transform='pandas')
    

    encoder_pipeline = Pipeline([
        ('to_str', FunctionTransformer(to_string, feature_names_out='one-to-one')),
        ('encoder', ct)
    ])
    encoder_pipeline.set_output(transform='pandas')
    
    return encoder_pipeline