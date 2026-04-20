from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import joblib
import io
import os
import uuid

# ── Import YOUR preprocessing pipeline ──────────────────────────
# All preprocessing files are inside the preprocessing/ subfolder
from preprocessing.preprocessing import preprocess, get_serializable_info

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
        # ── 1. Load Data ─────────────────────────────────────────
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        # ── 2. Split X and y ─────────────────────────────────────
        X = df.copy()
        y = None

        if task_type in ["Classification", "Regression"]:
            if not target_column or target_column not in df.columns:
                raise HTTPException(status_code=400, detail="Target column not found")
            y = X[target_column]
            X = X.drop(columns=[target_column])

        # ── 3. Run YOUR Preprocessing Pipeline (Point C) ─────────
        X_processed, y_processed, prep_info = preprocess(
            X.copy(),
            y.copy() if y is not None else pd.Series(dtype='float64'),
            task=task_type
        )

        safe_prep_info = get_serializable_info(prep_info)

        # ── 4. Model Training (Point D) ──────────────────────────
        best_model = None
        best_score = -np.inf
        results    = {}

        if task_type == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            algos = {
                "Random Forest":       RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
            }
            for name, algo in algos.items():
                algo.fit(X_train, y_train)
                y_pred = algo.predict(X_test)
                score  = metrics.f1_score(y_test, y_pred, average='weighted')
                if score > best_score:
                    best_score = score
                    best_model = algo
                    results = {
                        "algorithm":        name,
                        "accuracy":         float(metrics.accuracy_score(y_test, y_pred)),
                        "precision":        float(metrics.precision_score(y_test, y_pred, average='weighted')),
                        "recall":           float(metrics.recall_score(y_test, y_pred, average='weighted')),
                        "f1_score":         float(score),
                        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred).tolist()
                    }

        elif task_type == "Regression":
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            algos = {
                "Random Forest":     RandomForestRegressor(random_state=42),
                "Linear Regression": LinearRegression()
            }
            best_score = np.inf
            for name, algo in algos.items():
                algo.fit(X_train, y_train)
                y_pred = algo.predict(X_test)
                score  = metrics.mean_squared_error(y_test, y_pred)
                if score < best_score:
                    best_score = score
                    best_model = algo
                    results = {
                        "algorithm": name,
                        "mae":       float(metrics.mean_absolute_error(y_test, y_pred)),
                        "mse":       float(score),
                        "r2_score":  float(metrics.r2_score(y_test, y_pred))
                    }

        elif task_type == "Clustering":
            algos = {
                "K-Means":       KMeans(n_clusters=3, random_state=42),
                "Agglomerative": AgglomerativeClustering(n_clusters=3)
            }
            for name, algo in algos.items():
                labels = algo.fit_predict(X_processed)
                score  = metrics.silhouette_score(X_processed, labels)
                if score > best_score:
                    best_score = score
                    best_model = algo
                    results = {
                        "algorithm":        name,
                        "silhouette_score": float(score)
                    }

        # ── 5. Save Model ─────────────────────────────────────────
        model_id   = str(uuid.uuid4())
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump({"model": best_model, "prep_info": safe_prep_info,
                     "task_type": task_type}, model_path)
        models_db[model_id] = model_path

        # ── 6. Return response including preprocessing_info ───────
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