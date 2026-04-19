from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import joblib
import io
import os
import uuid

app = FastAPI()

# In-memory storage for simplicity (in production, use a database and persistent storage)
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
        # 1. Load Data (Point A/G)
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        # 2. Automated Data Preprocessing Pipeline (Point C)
        X = df.copy()
        y = None

        if task_type in ["Classification", "Regression"]:
            if not target_column or target_column not in df.columns:
                raise HTTPException(status_code=400, detail="Target column not found")
            y = X[target_column]
            X = X.drop(columns=[target_column])

            # Simple Resampling for Classification (Point C.iv)
            if task_type == "Classification":
                class_counts = y.value_counts()
                if (class_counts.max() / class_counts.min()) > 1.5:  # Basic imbalance check
                    # Simple Random Oversampling using pandas
                    df_resampled = pd.concat([X, y], axis=1)
                    max_size = class_counts.max()
                    lst = [df_resampled]
                    for class_index, group in df_resampled.groupby(target_column):
                        lst.append(group.sample(max_size - len(group), replace=True))
                    df_resampled = pd.concat(lst)
                    X = df_resampled.drop(columns=[target_column])
                    y = df_resampled[target_column]

        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocessing steps (Point C.i, C.ii, C.iii)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # 3. Model Training (Point D)
        X_train, X_test, y_train, y_test = (None, None, None, None)
        best_model = None
        best_score = -np.inf
        results = {}

        if task_type == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Algorithms (Point D.ii)
            algos = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
            }

            for name, algo in algos.items():
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', algo)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = metrics.f1_score(y_test, y_pred, average='weighted')
                
                if score > best_score:
                    best_score = score
                    best_model = clf
                    # Point E.i
                    results = {
                        "algorithm": name,
                        "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
                        "precision": float(metrics.precision_score(y_test, y_pred, average='weighted')),
                        "recall": float(metrics.recall_score(y_test, y_pred, average='weighted')),
                        "f1_score": float(score),
                        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred).tolist()
                    }

        elif task_type == "Regression":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            algos = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Linear Regression": LinearRegression()
            }

            best_score = np.inf # For regression, lower MSE is better
            for name, algo in algos.items():
                reg = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', algo)])
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                score = metrics.mean_squared_error(y_test, y_pred)
                
                if score < best_score:
                    best_score = score
                    best_model = reg
                    # Point E.ii
                    results = {
                        "algorithm": name,
                        "mae": float(metrics.mean_absolute_error(y_test, y_pred)),
                        "mse": float(score),
                        "r2_score": float(metrics.r2_score(y_test, y_pred))
                    }

        elif task_type == "Clustering":
            # Preprocess all data since it's unsupervised
            X_proc = preprocessor.fit_transform(X)
            
            algos = {
                "K-Means": KMeans(n_clusters=3, random_state=42),
                "Agglomerative": AgglomerativeClustering(n_clusters=3)
            }

            for name, algo in algos.items():
                labels = algo.fit_predict(X_proc)
                score = metrics.silhouette_score(X_proc, labels)
                
                if score > best_score:
                    best_score = score
                    # For clustering, the pipeline is just the preprocessor + model
                    # But sklearn doesn't support model in pipeline for predict easily if it's fit_predict
                    # We'll save them separately or wrap them
                    best_model = (preprocessor, algo) 
                    # Point E.iii
                    results = {
                        "algorithm": name,
                        "silhouette_score": float(score)
                    }

        # 4. Save Model (Point F/G)
        model_id = str(uuid.uuid4())
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump(best_model, model_path)
        
        models_db[model_id] = model_path

        return {
            "status": "success",
            "model_id": model_id,
            "metrics": results,
            "task_type": task_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{model_id}")
async def download_model(model_id: str):
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    path = models_db[model_id]
    return FileResponse(path, filename=f"model_{model_id}.joblib", media_type='application/octet-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
