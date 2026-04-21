import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN

def get_smote_object(X: pd.DataFrame, random_state=42):
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()
    
    num_categorical = len(categorical_features)
    num_total = len(X.columns)
    
    if num_categorical == 0:
        return SMOTE(random_state=random_state)
    elif num_categorical == num_total:
        return SMOTEN(random_state=random_state)
    else:

        categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
        return SMOTENC(categorical_features=categorical_indices, random_state=random_state)