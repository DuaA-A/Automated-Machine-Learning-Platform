import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

def get_scaler(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    standard_cols = []
    minmax_cols = []
    
    for col in numeric_features:
        col_min = X[col].min()
        col_max = X[col].max()
        if col_min >= 0 and col_max <= 100:
            minmax_cols.append(col)
        else:
            standard_cols.append(col)
            
    transformers = []
    if standard_cols:
        transformers.append(('std_scaler', StandardScaler(), standard_cols))
    if minmax_cols:
        transformers.append(('mm_scaler', MinMaxScaler(), minmax_cols))

    scaler = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    scaler.set_output(transform='pandas')
    return scaler