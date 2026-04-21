import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def get_encoder(X: pd.DataFrame, threshold: int = 15):
    cat_cols = X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

    ohe_cols = [c for c in cat_cols if X[c].nunique() <= threshold]
    ordinal_cols = [c for c in cat_cols if X[c].nunique() > threshold]

    transformers = []
    
    if ohe_cols:
        transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols))
    
    if ordinal_cols:
        transformers.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols))

    encoder = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    encoder.set_output(transform='pandas')
    return encoder