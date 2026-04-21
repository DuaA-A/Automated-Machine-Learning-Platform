import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def get_imputer(X: pd.DataFrame, categorical_fill_value='missing'):
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(('num_imputer', SimpleImputer(strategy='median'), numeric_features))
    if categorical_features:
        transformers.append(('cat_imputer', SimpleImputer(strategy='constant', fill_value=categorical_fill_value), categorical_features))


    preprocessor = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform='pandas')
    
    return preprocessor
