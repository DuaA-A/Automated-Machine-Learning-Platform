import pandas as pd
import logging
from sklearn.pipeline import Pipeline as SklearnPipeline

logger = logging.getLogger(__name__)
from .cleaning import DropUselessColumns
from .imputation import get_imputer
from .Scaling import get_scaler
from .encoding import get_encoder
from .imbalance import get_smote_object


def clean_raw_data(df: pd.DataFrame):
    logger.info("Preprocessing: Starting clean_raw_data...")
    initial_rows = len(df)

    df_cleaned = df.dropna(how='all')
    logger.info(f"Preprocessing: Dropped completely empty rows. Remaining: {len(df_cleaned)}")

    df_cleaned = df_cleaned.drop_duplicates(keep='first')
    logger.info(f"Preprocessing: Dropped duplicate rows. Remaining: {len(df_cleaned)}")

    rows_dropped = initial_rows - len(df_cleaned)
    report = {"duplicate_and_empty_rows_dropped": rows_dropped}

    return df_cleaned, report


def _build_feature_pipeline(X_train: pd.DataFrame, y_train: pd.Series = None, task: str = None):

    logger.info("Preprocessing: Building feature pipeline...")
    steps = []

    cleaner = DropUselessColumns(task=task)
    steps.append(('cleaner', cleaner))
    logger.info("Preprocessing: Fitting DropUselessColumns...")
    X_tmp = cleaner.fit_transform(X_train, y_train)

    imputer = get_imputer(X_tmp)
    steps.append(('imputer', imputer))
    logger.info("Preprocessing: Fitting Imputer...")
    X_tmp = imputer.fit_transform(X_tmp)

    scaler = get_scaler(X_tmp)
    steps.append(('scaler', scaler))
    logger.info("Preprocessing: Fitting Scaler...")
    X_tmp = scaler.fit_transform(X_tmp)

    encoder = get_encoder(X_tmp)
    steps.append(('encoder', encoder))
    logger.info("Preprocessing: Adding Encoder to pipeline...")

    pipeline = SklearnPipeline(steps=steps)
    logger.info("Preprocessing: Feature pipeline built successfully.")
    return pipeline


def fit_preprocess(X_train: pd.DataFrame, y_train: pd.Series, task: str):

    logger.info(f"Preprocessing: Starting fit_preprocess for task '{task}'. Initial shape: {X_train.shape}")
    shape_before = list(X_train.shape)


    feature_pipeline = _build_feature_pipeline(X_train, y_train, task=task)
    logger.info("Preprocessing: Transforming training data with feature pipeline...")
    X_transformed = feature_pipeline.fit_transform(X_train, y_train)
    y_processed = y_train

    if str(task).lower() == "classification" and y_train is not None and not y_train.empty:
        logger.info("Preprocessing: Applying SMOTE to handle class imbalance...")
        smote = get_smote_object(X_transformed)
        X_processed, y_processed = smote.fit_resample(X_transformed, y_train)
        logger.info(f"Preprocessing: SMOTE applied. Shape after resampling: {X_processed.shape}")
    else:
        logger.info("Preprocessing: Skipping SMOTE (not a classification task or empty target).")
        X_processed = X_transformed

    shape_after = list(X_processed.shape)
    columns_kept = list(X_processed.columns) if hasattr(X_processed, 'columns') else []

    columns_dropped = getattr(
        feature_pipeline.named_steps.get('cleaner', None), 'columns_to_drop_', []
    )

    ohe_columns = []
    label_columns = []
    encoding_details = {}

    encoder = feature_pipeline.named_steps.get('encoder', None)
    
    from sklearn.pipeline import Pipeline
    actual_ct = None
    if isinstance(encoder, Pipeline):
        for _, step in encoder.steps:
            if hasattr(step, 'transformers_'):
                actual_ct = step
                break
    elif hasattr(encoder, 'transformers_'):
        actual_ct = encoder

    if actual_ct:
        for name, transformer, cols in actual_ct.transformers_:
            if name == 'ohe':
                ohe_columns = list(cols)
                for c in cols:
                    encoding_details[c] = {"technique": "OneHotEncoder", "unique_count": X_train[c].nunique()}
            elif name == 'ordinal':
                label_columns = list(cols)
                for c in cols:
                    encoding_details[c] = {"technique": "LabelEncoder", "unique_count": X_train[c].nunique()}

    scale_standard_cols = []
    scale_minmax_cols = []
    scaling_details = {}

    scaler = feature_pipeline.named_steps.get('scaler', None)
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

    return feature_pipeline, X_processed, y_processed, prep_info


def transform_test(feature_pipeline, X_test: pd.DataFrame):
    return feature_pipeline.transform(X_test)


def get_serializable_info(prep_info: dict) -> dict:
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