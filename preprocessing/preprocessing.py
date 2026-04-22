import pandas as pd
from imblearn.pipeline import Pipeline
from .cleaning import DropUselessColumns  
from .imputation import get_imputer
from .Scaling import get_scaler
from .encoding import get_encoder
from .imbalance import get_smote_object

def clean_raw_data(df: pd.DataFrame):
    initial_rows = len(df)
    

    df_cleaned = df.dropna(how='all')
    

    df_cleaned = df_cleaned.drop_duplicates(keep='first')
    

    rows_dropped = initial_rows - len(df_cleaned)
    report = {"duplicate_and_empty_rows_dropped": rows_dropped}
    
    return df_cleaned, report
    
def build_preprocessing_pipeline(X_train: pd.DataFrame, task: str):

    steps = []
    

    cleaner = DropUselessColumns()
    steps.append(('cleaner', cleaner))
    

    X_tmp = cleaner.fit_transform(X_train)
    

    imputer = get_imputer(X_tmp) 
    steps.append(('imputer', imputer))
    X_tmp = imputer.fit_transform(X_tmp)
    

    scaler = get_scaler(X_tmp)
    steps.append(('scaler', scaler))
    X_tmp = scaler.fit_transform(X_tmp)
    

    encoder = get_encoder(X_tmp)
    steps.append(('encoder', encoder))
    X_tmp = encoder.fit_transform(X_tmp)

    if str(task).lower() == "classification":
        smote = get_smote_object(X_tmp)
        steps.append(('smote', smote))

    pipeline = Pipeline(steps=steps)
    
    return pipeline

def preprocess(X: pd.DataFrame, y: pd.Series, task: str):
    shape_before = list(X.shape)
    
    pipeline = build_preprocessing_pipeline(X, task)
    
    if str(task).lower() == "classification" and y is not None and not y.empty:
        X_res, y_res = pipeline.fit_resample(X, y)
    else:
        # Pipeline from imblearn supports fit_transform if no resampling or not provided
        X_res = pipeline.fit_transform(X, y)
        y_res = y

    shape_after = list(X_res.shape)
    if hasattr(X_res, 'columns'):
        columns_kept = list(X_res.columns)
    else:
        columns_kept = []

    # Try to extract info from pipeline
    columns_dropped = getattr(pipeline.named_steps.get('cleaner', None), 'columns_to_drop_', [])
    
    ohe_columns = []
    label_columns = []
    encoding_details = {}
    
    encoder = pipeline.named_steps.get('encoder', None)
    if encoder:
        for name, transformer, cols in encoder.transformers_:
            if name == 'ohe':
                ohe_columns = list(cols)
                for c in cols:
                    encoding_details[c] = {"technique": "OneHotEncoder", "new_cols": "Multiple"}
            elif name == 'ordinal':
                label_columns = list(cols)
                for c in cols:
                    encoding_details[c] = {"technique": "LabelEncoder", "new_cols": 1}
                    
    scale_standard_cols = []
    scale_minmax_cols = []
    scaling_details = {}
    
    scaler = pipeline.named_steps.get('scaler', None)
    if scaler:
        for name, transformer, cols in scaler.transformers_:
            if name == 'std_scaler':
                scale_standard_cols = list(cols)
                for c in cols:
                    scaling_details[c] = {"technique": "StandardScaler"}
            elif name == 'mm_scaler':
                scale_minmax_cols = list(cols)
                for c in cols:
                    scaling_details[c] = {"technique": "MinMaxScaler"}

    prep_info = {
        "columns_dropped": columns_dropped,
        "columns_kept": columns_kept,
        "shape_before_encoding": shape_before,
        "shape_after_encoding": shape_after,
        "ohe_columns": ohe_columns,
        "label_columns": label_columns,
        "encoding_details": encoding_details,
        "scale_standard_cols": scale_standard_cols,
        "scale_minmax_cols": scale_minmax_cols,
        "scaling_details": scaling_details
    }
    
    return X_res, y_res, prep_info

def get_serializable_info(prep_info: dict) -> dict:
    import builtins
    safe_info = {}
    for k, v in prep_info.items():
        if isinstance(v, list):
            safe_info[k] = [str(i) for i in v]
        elif isinstance(v, tuple):
            safe_info[k] = list(v)
        elif isinstance(v, dict):
            safe_info[k] = get_serializable_info(v)
        elif type(v).__module__ == 'numpy':
            safe_info[k] = v.item()
        else:
            safe_info[k] = v
    return safe_info