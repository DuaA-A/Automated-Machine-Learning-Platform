import streamlit as st
import pandas as pd
import requests
import base64

def main():
    st.set_page_config(page_title="Automated Machine Learning Platform", layout="wide")
    st.title("Automated Machine Learning Platform")
    st.write("An integrated system for dataset analysis and automated model training.")


    st.header("1. Data Ingestion")
    with st.container():
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
        df = None

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)

                if df is not None:
                    st.success(f"File loaded: {uploaded_file.name}")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("Metadata Summary")
                        st.write(f"Observations: {df.shape[0]}")
                        st.write(f"Features: {df.shape[1]}")
                        st.dataframe(df.dtypes.astype(str).to_frame(name="Type"), height=300)
                    
                    with col2:
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)

                    st.subheader("Data Quality Assessment")
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        st.write("Missing Values")
                        missing = df.isnull().sum()
                        if missing.sum() > 0:
                            st.warning(f"Count: {missing.sum()}")
                            st.dataframe(missing[missing > 0])
                        else:
                            st.info("No missing values.")
                    with q2:
                        st.write("Duplicate Entries")
                        dups = df.duplicated().sum()
                        if dups > 0:
                            st.warning(f"Count: {dups}")
                        else:
                            st.info("No duplicates.")
                    with q3:
                        st.write("Scale Variation")
                        num_cols = df.select_dtypes(include=['number']).columns
                        if not num_cols.empty:
                            ranges = df[num_cols].max() - df[num_cols].min()
                            if ranges.max() / (ranges.min() + 1e-9) > 10:
                                st.warning("High feature scale variation.")
                            else:
                                st.info("Consistent scales.")

            except Exception as e:
                st.error(f"Error during ingestion: {e}")


    st.header("2. Configuration")
    t1, t2 = st.columns(2)
    with t1:
        ml_task = st.radio("Task Category",
                            ("Classification", "Regression", "Clustering"), horizontal=True)
    
    target_column = None
    algorithm_choice = "AutoML (Find Best Model)"

    if ml_task in ["Classification", "Regression"]:
        if df is not None:
            with t2:
                if ml_task == "Regression":
                    valid_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if not valid_cols:
                        st.warning("No numerical columns found for Regression.")
                    target_column = st.selectbox("Target Column", options=valid_cols)
                else:
                    target_column = st.selectbox("Target Column",
                                                  options=df.columns.tolist())
        else:
            st.info("Dataset required for column selection.")

    st.subheader("Model Selection")
    if ml_task == "Classification":
        algorithm_choice = st.radio("Algorithm", 
            ("AutoML (Find Best Model)", "Random Forest", "Gradient Boosting"), horizontal=True)
    elif ml_task == "Regression":
        algorithm_choice = st.radio("Algorithm", 
            ("AutoML (Find Best Model)", "Random Forest", "Ridge Regression"), horizontal=True)
    elif ml_task == "Clustering":
        algorithm_choice = st.radio("Algorithm", 
            ("AutoML (Find Best Model)", "K-Means", "Agglomerative"), horizontal=True)

    st.divider()


    if st.button("Execute Pipeline", use_container_width=True):
        if df is None:
            st.error("Missing dataset.")
        elif ml_task in ["Classification", "Regression"] and target_column is None:
            st.error("Missing target column.")
        else:
            with st.spinner("Executing..."):
                try:
                    uploaded_file.seek(0)
                    response = requests.post(
                        "http://localhost:8000/train",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        data={"task_type": ml_task,
                              "target_column": target_column if target_column else "",
                              "algorithm_choice": algorithm_choice}
                    )

                    if response.status_code == 200:
                        res  = response.json()
                        prep = res.get("preprocessing_info", {})
                        metrics_data = res['metrics']

                        st.success(f"Processing Complete: {metrics_data['algorithm']}")


                        tab1, tab2, tab3 = st.tabs(["Evaluation Metrics", "Preprocessing Log", "Model Artifacts"])

                        with tab1:
                            st.subheader("Performance Indicators")
                            m_col1, m_col2, m_col3 = st.columns(3)

                            if ml_task == "Classification":
                                m_col1.metric("Accuracy",    f"{metrics_data['accuracy']:.4f}")
                                m_col2.metric("Weighted F1", f"{metrics_data['f1_score']:.4f}")
                                m_col3.metric("Precision",   f"{metrics_data['precision']:.4f}")
                                
                                st.write("Confusion Matrix")
                                cm_df = pd.DataFrame(
                                    metrics_data['confusion_matrix'],
                                    index=[f"Actual: {c}" for c in metrics_data.get('class_labels', [])],
                                    columns=[f"Predicted: {c}" for c in metrics_data.get('class_labels', [])]
                                )
                                st.dataframe(cm_df, use_container_width=True)

                            elif ml_task == "Regression":
                                m_col1.metric("MAE",      f"{metrics_data['mae']:.4f}")
                                m_col2.metric("MSE",      f"{metrics_data['mse']:.4f}")
                                m_col3.metric("R2 Score", f"{metrics_data['r2_score']:.4f}")

                            elif ml_task == "Clustering":
                                m_col1.metric("Silhouette Score", f"{metrics_data['silhouette_score']:.4f}")
                                if 'optimal_k' in metrics_data:
                                    m_col2.metric("Optimal Clusters", f"{metrics_data['optimal_k']}")
                                
                                dist_col, cent_col = st.columns([1, 2])
                                with dist_col:
                                    st.write("Cluster Frequency")
                                    sizes = metrics_data.get('cluster_sizes', {})
                                    if sizes:
                                        st.bar_chart(pd.Series(sizes, name="Count"))
                                
                                with cent_col:
                                    centroids = metrics_data.get('centroids')
                                    features = metrics_data.get('feature_names')
                                    if centroids and features:
                                        st.write("Cluster Centroids")
                                        centroid_df = pd.DataFrame(
                                            centroids, columns=features,
                                            index=[f"Cluster {i}" for i in range(len(centroids))]
                                        )
                                        st.dataframe(centroid_df, use_container_width=True)
                                

                                pca_data = metrics_data.get('pca_data')
                                if pca_data:
                                    st.write("2D Cluster Projection (PCA)")
                                    viz_df = pd.DataFrame({
                                        'Dimension 1': pca_data['x'],
                                        'Dimension 2': pca_data['y'],
                                        'Cluster': [f"Cluster {l}" for l in pca_data['labels']]
                                    })
                                    st.scatter_chart(viz_df, x='Dimension 1', y='Dimension 2', color='Cluster')

                        with tab2:
                            st.subheader("Automated Preprocessing Operations")
                            
                            dropped = prep.get("columns_dropped", [])
                            if dropped:
                                st.write(f"Removed Columns: {len(dropped)}")
                                st.caption(", ".join(dropped))
                            
                            p_col1, p_col2 = st.columns(2)
                            
                            with p_col1:
                                st.write("Categorical Encoding")
                                encoding_details = prep.get("encoding_details", {})
                                if encoding_details:
                                    enc_rows = [{"Column": col, "Method": d.get("technique"), "Cardinality": d.get("unique_count")} 
                                                for col, d in encoding_details.items()]
                                    st.dataframe(pd.DataFrame(enc_rows), use_container_width=True)
                                else:
                                    st.info("No categorical features detected.")

                            with p_col2:
                                st.write("Numerical Scaling")
                                std_cols = prep.get("scale_standard_cols", [])
                                mm_cols = prep.get("scale_minmax_cols", [])
                                if std_cols or mm_cols:
                                    st.write(f"Standardized: {len(std_cols)}")
                                    st.write(f"Normalized: {len(mm_cols)}")
                                else:
                                    st.info("No numeric features scaled.")

                            with st.expander("Detailed Logic Trace"):
                                reasons = prep.get("reasons", {})
                                if reasons:
                                    rows = [{"Feature": col, "Operation": reason} for col, reason in reasons.items()]
                                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                        with tab3:
                            st.subheader("Model Persistence")
                            model_id = res['model_id']
                            st.write(f"Artifact ID: {model_id}")
                            if st.button("Download Binary Model", use_container_width=True):
                                model_response = requests.get(f"http://localhost:8000/download/{model_id}")
                                if model_response.status_code == 200:
                                    b64 = base64.b64encode(model_response.content).decode()
                                    href = (f'<a href="data:application/octet-stream;base64,{b64}" '
                                            f'download="model_{model_id}.joblib">Save Joblib Artifact</a>')
                                    st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'System error')}")
                except Exception as e:
                    st.error(f"Operation failed: {e}")

if __name__ == "__main__":
    main()