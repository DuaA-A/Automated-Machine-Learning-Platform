import streamlit as st
import pandas as pd
import requests
import base64

def main():
    st.set_page_config(page_title="Automated ML Platform", layout="centered")
    st.title("Automated Machine Learning Platform")
    st.write("Upload your dataset and configure your machine learning task seamlessly.")

    # ── A. Data Ingestion ────────────────────────────────────────
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"],
                                      help="Supported formats: .csv, .xlsx")
    df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            if df is not None:
                st.success(f"Successfully loaded **{uploaded_file.name}**")
                st.subheader("Data Preview")
                st.write(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                st.dataframe(df.head())

                st.subheader("Dataset Quality Report")
                q1, q2 = st.columns(2)
                with q1:
                    st.write("**Missing Values:**")
                    missing = df.isnull().sum()
                    if missing.sum() > 0:
                        st.warning(f"Total missing cells: {missing.sum()}")
                        st.dataframe(missing[missing > 0])
                    else:
                        st.success("No missing values detected!")
                with q2:
                    st.write("**Duplicates:**")
                    dups = df.duplicated().sum()
                    if dups > 0:
                        st.warning(f"Found {dups} duplicate rows.")
                    else:
                        st.success("No duplicate rows detected!")

                st.write("**Scaling & Feature Analysis:**")
                num_cols = df.select_dtypes(include=['number']).columns
                if not num_cols.empty:
                    ranges = df[num_cols].max() - df[num_cols].min()
                    if ranges.max() / (ranges.min() + 1e-9) > 10:
                        st.info("Features have different scales — backend will handle automatically.")
                    else:
                        st.success("Numerical features are within similar scales.")

                st.write("**Column Data Types:**")
                st.dataframe(df.dtypes.astype(str).to_frame(name="Data Type"))

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # ── B. Task Selection ────────────────────────────────────────
    st.header("2. Task Selection")
    ml_task = st.radio("Select the Machine Learning Problem Type:",
                        ("Classification", "Regression", "Clustering"))
    target_column = None

    if ml_task in ["Classification", "Regression"]:
        if df is not None:
            target_column = st.selectbox("Select the Target Variable (Label):",
                                          options=df.columns.tolist())
            if target_column:
                st.subheader(f"Target Variable Preview: {target_column}")
                if ml_task == "Classification":
                    counts = df[target_column].value_counts()
                    st.bar_chart(counts)
                    ratio = counts.max() / counts.min()
                    if ratio > 1.5:
                        st.warning(f"Imbalance ratio: {ratio:.2f} — (resampling disabled)")
                    else:
                        st.success("Target classes are well-balanced.")
                    st.dataframe(counts)
                else:
                    st.write(df[target_column].describe())
                    st.line_chart(df[target_column].head(100))
        else:
            st.warning("Please upload a dataset first.")

    st.divider()

    # ── Start Pipeline ───────────────────────────────────────────
    if st.button("Start Automated ML Pipeline", type="primary"):
        if df is None:
            st.error("No dataset uploaded.")
        elif ml_task in ["Classification", "Regression"] and target_column is None:
            st.error("Please select a target variable.")
        else:
            with st.spinner("Running pipeline... This may take a moment."):
                try:
                    uploaded_file.seek(0)
                    response = requests.post(
                        "http://localhost:8000/train",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        data={"task_type": ml_task,
                              "target_column": target_column if target_column else ""}
                    )

                    if response.status_code == 200:
                        res  = response.json()
                        prep = res.get("preprocessing_info", {})

                        st.success(f"Pipeline completed! Best Algorithm: **{res['metrics']['algorithm']}**")

                        # ════════════════════════════════════════
                        #  3. PREPROCESSING REPORT
                        # ════════════════════════════════════════
                        st.header("3. Preprocessing Report")
                        st.caption("Applied only: ii. Encoding + iii. Scaling")

                        # Decision Log
                        reasons = prep.get("reasons", {})
                        if reasons:
                            with st.expander("📋 Full Decision Log", expanded=False):
                                rows = [
                                    {"Column": "TARGET" if col == "__target__" else col,
                                     "Decision": reason}
                                    for col, reason in reasons.items()
                                ]
                                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                        # ── Step 1: Drop Useless Columns ──────────────────────
                        with st.expander("Step 1 — Useless Column Removal", expanded=True):
                            dropped = prep.get("columns_dropped", [])
                            kept    = prep.get("columns_kept", [])
                            if dropped:
                                st.warning(f"**{len(dropped)} column(s) removed:** {', '.join(dropped)}")
                            else:
                                st.success("No columns were removed.")
                            if kept:
                                st.write(f"**Columns used for training ({len(kept)}):** {', '.join(kept)}")

                        # ── Step 3: ii. Categorical Encoding ──────────────────
                        with st.expander("Step 3 — ii. Categorical Encoding", expanded=True):
                            ohe_cols         = prep.get("ohe_columns", [])
                            label_cols       = prep.get("label_columns", [])
                            encoding_details = prep.get("encoding_details", {})
                            s_before         = prep.get("shape_before_encoding", [None, None])
                            s_after          = prep.get("shape_after_encoding",  [None, None])

                            if encoding_details:
                                e1, e2 = st.columns(2)
                                e1.success(
                                    "**ii. OneHotEncoder**\n\n"
                                    "Used when: ≤ 15 unique values\n"
                                    "Creates one binary column per category\n"
                                    f"Applied to: **{', '.join(ohe_cols) if ohe_cols else 'none'}**"
                                )
                                e2.info(
                                    "**ii. LabelEncoder**\n\n"
                                    "Used when: > 15 unique values\n"
                                    "Replaces each category with an integer\n"
                                    f"Applied to: **{', '.join(label_cols) if label_cols else 'none'}**"
                                )

                                st.write("**Encoding Breakdown:**")
                                enc_rows = []
                                for col, d in encoding_details.items():
                                    enc_rows.append({
                                        "Column": col,
                                        "Technique": d.get("technique", "—"),
                                        "Unique Values": d.get("unique_count", "—"),
                                        "New cols created": d.get("new_cols", 1)
                                    })
                                st.dataframe(pd.DataFrame(enc_rows), use_container_width=True)

                                if s_before[1] is not None and s_after[1] is not None:
                                    sc1, sc2, sc3 = st.columns(3)
                                    sc1.metric("Columns Before", s_before[1])
                                    sc2.metric("Columns After", s_after[1])
                                    sc3.metric("Added by OHE", 
                                               f"+{max(0, int(s_after[1]) - int(s_before[1]))}")
                            else:
                                st.success("No categorical columns found — encoding was not needed.")

                        # ── Step 4: iii. Numerical Scaling ───────────────────
                        with st.expander("Step 4 — iii. Numerical Scaling", expanded=True):
                            std_cols        = prep.get("scale_standard_cols", [])
                            mm_cols         = prep.get("scale_minmax_cols",   [])
                            scaling_details = prep.get("scaling_details", {})  # Note: currently empty in backend

                            if std_cols or mm_cols:
                                s1, s2 = st.columns(2)
                                s1.success(
                                    "**iii. StandardScaler**\n\n"
                                    "Formula: z = (x − mean) / std\n"
                                    "Result: mean=0, std=1\n"
                                    f"Applied to: **{', '.join(std_cols) if std_cols else 'none'}**"
                                )
                                s2.info(
                                    "**iii. MinMaxScaler**\n\n"
                                    "Formula: (x − min) / (max − min)\n"
                                    "Result: range [0, 1]\n"
                                    f"Applied to: **{', '.join(mm_cols) if mm_cols else 'none'}**"
                                )

                                if scaling_details:
                                    st.write("**Scaling Breakdown:**")
                                    scale_rows = []
                                    for col, d in scaling_details.items():
                                        scale_rows.append({
                                            "Column": col,
                                            "Technique": d.get("technique", "—"),
                                            "Before min": d.get("before_min", "—"),
                                            "Before max": d.get("before_max", "—"),
                                        })
                                    st.dataframe(pd.DataFrame(scale_rows), use_container_width=True)
                            else:
                                st.success("No numerical columns found — scaling was not needed.")

                        # ════════════════════════════════════════
                        #  4. MODEL RESULTS
                        # ════════════════════════════════════════
                        st.header("4. Model Evaluation Results")
                        metrics_data = res['metrics']
                        c = st.columns(3)

                        if ml_task == "Classification":
                            c[0].metric("Accuracy",    f"{metrics_data['accuracy']:.4f}")
                            c[1].metric("Weighted F1", f"{metrics_data['f1_score']:.4f}")
                            c[2].metric("Precision",   f"{metrics_data['precision']:.4f}")
                            st.subheader("Confusion Matrix")
                            st.dataframe(pd.DataFrame(metrics_data['confusion_matrix']))

                        elif ml_task == "Regression":
                            c[0].metric("MAE",      f"{metrics_data['mae']:.4f}")
                            c[1].metric("MSE",      f"{metrics_data['mse']:.4f}")
                            c[2].metric("R² Score", f"{metrics_data['r2_score']:.4f}")

                        elif ml_task == "Clustering":
                            c[0].metric("Silhouette Score",
                                        f"{metrics_data['silhouette_score']:.4f}")

                        # ════════════════════════════════════════
                        #  5. EXPORT
                        # ════════════════════════════════════════
                        st.divider()
                        st.subheader("5. Export Model")
                        model_id = res['model_id']

                        if st.button("Download Trained Model (.joblib)"):
                            model_response = requests.get(
                                f"http://localhost:8000/download/{model_id}")
                            if model_response.status_code == 200:
                                b64  = base64.b64encode(model_response.content).decode()
                                href = (f'<a href="data:application/octet-stream;base64,{b64}" '
                                        f'download="model_{model_id}.joblib">'
                                        f'Click here to download your model</a>')
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.error("Failed to fetch model.")
                    else:
                        st.error(f"Backend Error: {response.json().get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
                    st.info("Make sure the backend is running: python -m uvicorn backend:app --reload")

if __name__ == "__main__":
    main()