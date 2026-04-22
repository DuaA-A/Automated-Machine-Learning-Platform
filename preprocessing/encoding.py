import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def get_encoder(X: pd.DataFrame, threshold: int = 15):
    # Select categorical-like columns
    cat_cols = X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

    if not cat_cols:
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(lambda x: x) # No-op if no categorical columns

    ohe_cols = [c for c in cat_cols if X[c].nunique() <= threshold]
    ordinal_cols = [c for c in cat_cols if X[c].nunique() > threshold]

    # Pre-step: Ensure all categorical columns are strings to avoid mixed-type errors (e.g. bool vs str)
    from sklearn.preprocessing import FunctionTransformer
    def to_string(df):
        df = df.copy()
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    transformers = []
    
    if ohe_cols:
        transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols))
    
    if ordinal_cols:
        transformers.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols))

    ct = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # We return a pipeline that first converts to string, then encodes
    from sklearn.pipeline import Pipeline
    encoder_pipeline = Pipeline([
        ('to_str', FunctionTransformer(to_string)),
        ('encoder', ct)
    ])
    
    return encoder_pipeline