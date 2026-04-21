import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from preprocessing.preprocessing import build_preprocessing_pipeline, clean_raw_data 

def test_pipeline_on_dataset(file_name, target_col, task_type):
    print("\n" + "=" * 70)
    print(f" TESTING DATASET: {file_name}")
    print(f" TASK: {task_type.upper()}")
    print("=" * 70)

    try:

        df = pd.read_csv(file_name)
        

        df, cleaning_report = clean_raw_data(df)
        print(f"[0] Pre-cleaning: Dropped {cleaning_report['duplicate_and_empty_rows_dropped']} bad rows.")
        

        if task_type == 'Clustering':
            X = df.copy() 
            y = None
        else:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        
        print(f"[1] Original Shape: {X.shape}")
        print(f"    Missing Values: {X.isna().sum().sum()}")


        pipeline = build_preprocessing_pipeline(X_train=X, task=task_type)
        print("\n[2] Pipeline built successfully.")


        print("[3] Running pipeline execution... (this may take a few seconds)")
        
        if task_type == 'Classification':
            pipe_up_to_smote = pipeline[:-1] 
            X_resampled, y_processed = pipe_up_to_smote.fit_resample(X, y)
            encoder = pipeline.named_steps['encoder']
            X_processed = encoder.fit_transform(X_resampled)
            
        else:

            X_processed = pipeline.fit_transform(X)
            y_processed = y
        

        if not isinstance(X_processed, pd.DataFrame):
            X_processed = pd.DataFrame(X_processed)
            
        print("\n[4] RESULTS:")
        print(f"    New Shape: {X_processed.shape}")
        print(f"    Missing Values Remaining: {X_processed.isna().sum().sum()}")
        
        if task_type == 'Classification':
            new_counts = pd.Series(y_processed).value_counts()
            if new_counts.min() == new_counts.max():
                print("    -> SUCCESS: Classes are perfectly balanced!")
        elif task_type == 'Regression':
            print("    -> SUCCESS: Regression task properly skipped SMOTE.")
        elif task_type == 'Clustering':
            print("    -> SUCCESS: Clustering processed X smoothly without a 'y' target.")
            if 'CUST_ID' not in X_processed.columns:
                 print("    -> SUCCESS: DropUselessColumns automatically deleted 'CUST_ID'!")
            
        print("\n    Sample of Processed Data (First 3 Rows):")
        print(np.round(X_processed.head(3), 3))
        print("-" * 70 + "\n")

    except Exception as e:
        print(f"\n[!] ERROR PROCESSING {file_name}:")
        import traceback
        traceback.print_exc() 


if __name__ == "__main__":

    test_pipeline_on_dataset(
        file_name="mushrooms.csv", 
        target_col="class", 
        task_type="Classification"
    )


    test_pipeline_on_dataset(
        file_name="house_price_regression_dataset.csv", 
        target_col="House_Price", 
        task_type="Regression"
    )
    

    test_pipeline_on_dataset(
        file_name="CC GENERAL.csv", 
        target_col=None, 
        task_type="Clustering"
    )