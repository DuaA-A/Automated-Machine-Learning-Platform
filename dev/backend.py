from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import joblib
import io
import os
import uuid
import logging

from training import train_and_evaluate
from preprocessing.preprocessing import get_serializable_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

models_db = {}


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(None)
):
    try:
        
        logger.info("Step 1: Starting Data Loading...")
        contents = await file.read()
        if file.filename.endswith('.csv'):
            logger.info("Parsing CSV file format.")
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            logger.info("Parsing Excel file format.")
            df = pd.read_excel(io.BytesIO(contents))
        else:
            logger.error("Unsupported file format provided.")
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            logger.error("Dataset is empty after parsing.")
            raise HTTPException(status_code=400, detail="Dataset is empty")

        logger.info(f"Data successfully loaded. Shape: {df.shape}")


        logger.info("Step 2: Basic Cleaning (Duplicates & Empty Rows)...")
        from preprocessing.preprocessing import clean_raw_data
        df, clean_report = clean_raw_data(df)
        logger.info(f"Data shape after basic cleaning: {df.shape}")


        logger.info(f"Step 3: Separating features (X) and target (y). Task Type: {task_type}")
        X = df.copy()
        y = None

        if task_type in ["Classification", "Regression"]:
            if not target_column or target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found.")
                raise HTTPException(status_code=400, detail="Target column not found")
            y = X[target_column]
            X = X.drop(columns=[target_column])
            logger.info(f"Target column '{target_column}' extracted successfully.")

            if task_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
                logger.error(f"Regression task requires a numeric target, but got categorical for '{target_column}'.")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Target column '{target_column}' contains non-numeric data. Regression models require a numeric target variable. Did you mean to use Classification?"
                )

 
        logger.info("Step 4: Commencing Training Pipeline...")

        best_model, results, prep_info = train_and_evaluate(X, y, task_type)
        logger.info(f"Training Pipeline Completed. Best Model Algorithm: {results.get('algorithm', 'Unknown')}")

        safe_prep_info = get_serializable_info(prep_info)


        logger.info("Step 5: Saving the best model to disk...")
        model_id   = str(uuid.uuid4())
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump({"model": best_model, "prep_info": safe_prep_info,
                     "task_type": task_type}, model_path)
        models_db[model_id] = model_path
        logger.info(f"Model successfully saved with ID: {model_id}")


        logger.info("Returning processing results to the client.")
        return {
            "status":             "success",
            "model_id":           model_id,
            "metrics":            results,
            "task_type":          task_type,
            "preprocessing_info": safe_prep_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{model_id}")
async def download_model(model_id: str):
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    path = models_db[model_id]
    return FileResponse(path, filename=f"model_{model_id}.joblib",
                        media_type='application/octet-stream')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)