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
    

    if str(task).lower() == "classification":
        smote = get_smote_object(X_tmp)
        steps.append(('smote', smote))

    encoder = get_encoder(X_tmp)
    steps.append(('encoder', encoder))
    

    pipeline = Pipeline(steps=steps)
    
    return pipeline