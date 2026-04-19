import streamlit as st
import pandas as pd
import io

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
        else:
            st.warning("Please upload a dataset first to select the target variable.")

    st.divider()

    # Submission button (placeholder for further API integration)
    if st.button("Start Automated ML Pipeline", type="primary"):
        if df is None:
            st.error("Cannot start pipeline: No dataset uploaded.")
        elif ml_task in ["Classification", "Regression"] and target_column is None:
            st.error("Cannot start pipeline: Target variable must be selected for supervised learning.")
        else:
            st.success("Configuration is fully set! Ready to send to backend API.")
            
            # --- Future Integration: Send this to the backend API ---
            # payload = {
            #     "task_type": ml_task,
            #     "target_column": target_column,
            # }
            # # Also need to send the file/data
            # response = requests.post("http://localhost:8000/api/train", files={"file": uploaded_file.getvalue()}, data=payload)
            

if __name__ == "__main__":
    main()
