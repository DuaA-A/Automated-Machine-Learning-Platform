# Automated Machine Learning Platform

**IS424 — Selected Topics in Data Engineering, Spring 2026**  
Cairo University, Faculty of Computer and Artificial Intelligence

---

An end-to-end AutoML web application that allows non-technical users to upload a raw dataset, select a machine learning task, and receive a fully trained, evaluated, and downloadable model — without writing any code.

> **Full documentation:** Open [`README.html`](README.html) in a browser for the complete styled project report.

---

## Quick Start

**1. Install dependencies**

```bash
pip install fastapi uvicorn streamlit pandas numpy scikit-learn joblib openpyxl requests
```

**2. Start the backend**

```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

**3. Start the frontend** (in a separate terminal)

```bash
streamlit run frontend.py
```

The application will be available at `http://localhost:8501`.  
The FastAPI docs are available at `http://localhost:8000/docs`.

---

## Project Structure

```
DE-Project/
├── backend.py          # FastAPI service — preprocessing, training, evaluation, model export
├── frontend.py         # Streamlit UI — upload, task selection, results display
├── models/             # Serialised .joblib model artefacts (generated at runtime)
├── README.html         # Full styled project documentation
├── README.md           # This file
└── .gitignore
```

---

## Feature Summary

| Requirement | Status |
|---|---|
| CSV & Excel file upload with data preview | Complete |
| ML task selection: Classification, Regression, Clustering | Complete |
| Target column selection for supervised tasks | Complete |
| Missing value imputation (mean / most-frequent) | Complete |
| Categorical encoding (OneHotEncoder) | Complete |
| Feature scaling (StandardScaler) | Complete |
| Class imbalance detection & random oversampling | Complete |
| 80/20 train–test split with best-model selection | Complete |
| Classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix | Complete |
| Regression metrics: MAE, MSE, R² Score | Complete |
| Clustering metric: Silhouette Score | Complete |
| Results display with metric cards | Complete |
| Model download as `.joblib` file | Complete |
| FastAPI REST endpoints (`POST /train`, `GET /download/{id}`) | Complete |

---

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI + Uvicorn
- **ML:** Scikit-learn (Random Forest, Logistic Regression, Linear Regression, K-Means, Agglomerative Clustering)
- **Data:** Pandas, NumPy
- **Serialisation:** Joblib
