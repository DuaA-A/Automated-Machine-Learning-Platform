import streamlit as st
import pandas as pd
import io
import requests
import json
import base64

def main():
    st.set_page_config(page_title="Automated ML Platform", layout="centered")

    st.title("Automated Machine Learning Platform")
    st.write("Upload your dataset and configure your machine learning task seamlessly.")

    # A. Data Ingestion (Frontend)
    st.header("1. Data Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload a dataset", 
        type=["csv", "xlsx"],
        help="Supported formats: .csv, .xlsx"
    )

    df = None

    if uploaded_file is not None:
        try:
            # Check file extension and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            if df is not None:
                st.success(f"Successfully loaded {uploaded_file.name}")
                
                # Data Preview
                st.subheader("Data Preview")
                st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
                st.dataframe(df.head())

                # Dataset Quality Report (New Feature)
                st.subheader("📊 Dataset Quality Report")
                
                # Using columns for a cleaner layout
                q_col1, q_col2 = st.columns(2)
                
                with q_col1:
                    st.write("**Missing Values:**")
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning(f"Total missing cells: {missing_data.sum()}")
                        st.dataframe(missing_data[missing_data > 0])
                    else:
                        st.success("No missing values detected!")
                
                with q_col2:
                    st.write("**Duplicates:**")
                    duplicates = df.duplicated().sum()
                    if duplicates > 0:
                        st.warning(f"Found {duplicates} duplicate rows.")
                    else:
                        st.success("No duplicate rows detected!")
                
                st.write("**Column Data Types:**")
                st.dataframe(df.dtypes.astype(str).to_frame(name="Data Type"))
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # B. Task Selection (Frontend)
    st.header("2. Task Selection")

    ml_task = st.radio(
        "Select the Machine Learning Problem Type:",
        ("Classification", "Regression", "Clustering")
    )

    target_column = None

    # Target Variable Selection (For Supervised Learning)
    if ml_task in ["Classification", "Regression"]:
        if df is not None:
            columns = df.columns.tolist()
            target_column = st.selectbox(
                "Select the Target Variable (Label):", 
                options=columns,
                help="Choose the column you want the model to predict."
            )

            # Target Variable Preview (Enhancement for Point B)
            if target_column:
                st.subheader(f"Target Variable Preview: {target_column}")
                if ml_task == "Classification":
                    # Show value counts/distribution
                    counts = df[target_column].value_counts()
                    st.bar_chart(counts)
                    st.write("Class Distribution:")
                    st.dataframe(counts)
                else:
                    # For regression, show summary stats or histogram
                    st.write("Summary Statistics:")
                    st.write(df[target_column].describe())
                    st.line_chart(df[target_column].head(100)) # Simple line chart of first 100 values
        else:
            st.warning("Please upload a dataset first to select the target variable.")

    st.divider()

    # Submission button
    if st.button("Start Automated ML Pipeline", type="primary"):
        if df is None:
            st.error("Cannot start pipeline: No dataset uploaded.")
        elif ml_task in ["Classification", "Regression"] and target_column is None:
            st.error("Cannot start pipeline: Target variable must be selected for supervised learning.")
        else:
            with st.spinner("Training your model... This may take a moment."):
                try:
                    # Prepare the data for the API request
                    # Reset pointer for the uploaded file
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {
                        "task_type": ml_task,
                        "target_column": target_column if target_column else ""
                    }

                    # Call backend API (FastAPI)
                    response = requests.post("http://localhost:8000/train", files=files, data=data)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"Pipeline completed successfully! Best Algorithm: {res['metrics']['algorithm']}")
                        
                        # F. Results & Export (Frontend)
                        st.header("3. Model Evaluation Results")
                        
                        metrics_data = res['metrics']
                        cols = st.columns(3)
                        
                        if ml_task == "Classification":
                            cols[0].metric("Accuracy", f"{metrics_data['accuracy']:.4f}")
                            cols[1].metric("Weighted F1", f"{metrics_data['f1_score']:.4f}")
                            cols[2].metric("Precision", f"{metrics_data['precision']:.4f}")
                            
                            st.subheader("Confusion Matrix")
                            st.write(pd.DataFrame(metrics_data['confusion_matrix']))
                            
                        elif ml_task == "Regression":
                            cols[0].metric("MAE", f"{metrics_data['mae']:.4f}")
                            cols[1].metric("MSE", f"{metrics_data['mse']:.4f}")
                            cols[2].metric("R² Score", f"{metrics_data['r2_score']:.4f}")
                            
                        elif ml_task == "Clustering":
                            cols[0].metric("Silhouette Score", f"{metrics_data['silhouette_score']:.4f}")

                        # Model Export
                        st.divider()
                        st.subheader("4. Export Model")
                        model_id = res['model_id']
                        download_url = f"http://localhost:8000/download/{model_id}"
                        
                        if st.button("Download Trained Model (.joblib)"):
                            model_response = requests.get(download_url)
                            if model_response.status_code == 200:
                                b64 = base64.b64encode(model_response.content).decode()
                                href = f'<a href="data:application/octet-stream;base64,{b64}" download="model_{model_id}.joblib">Click here to download your model</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.error("Failed to fetch model for download.")
                    else:
                        st.error(f"Backend Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
                    st.info("Make sure the backend server is running at http://localhost:8000")

if __name__ == "__main__":
    main()
