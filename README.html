<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AutoML Platform — IS424 Project Documentation</title>
  <style>
    /* ── Reset & Base ─────────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d1117;
      --surface:   #161b22;
      --surface2:  #1c2230;
      --border:    #30363d;
      --accent:    #2f81f7;
      --accent2:   #388bfd;
      --green:     #3fb950;
      --purple:    #a371f7;
      --orange:    #d29922;
      --red:       #f85149;
      --text:      #e6edf3;
      --muted:     #8b949e;
      --heading-font: 'Georgia', 'Times New Roman', serif;
      --body-font:    'Segoe UI', system-ui, -apple-system, sans-serif;
      --mono-font:    'Consolas', 'Courier New', monospace;
    }

    html { scroll-behavior: smooth; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--body-font);
      font-size: 15px;
      line-height: 1.75;
      min-height: 100vh;
    }

    /* ── Layout ───────────────────────────────────────────────────────── */
    .wrapper {
      max-width: 920px;
      margin: 0 auto;
      padding: 48px 24px 96px;
    }

    /* ── Cover ────────────────────────────────────────────────────────── */
    .cover {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--surface);
      padding: 52px 48px 44px;
      margin-bottom: 56px;
    }

    .cover .institution {
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .cover .course {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 28px;
    }

    .cover h1 {
      font-family: var(--heading-font);
      font-size: 2.15rem;
      font-weight: 700;
      line-height: 1.25;
      color: var(--text);
      margin-bottom: 12px;
    }

    .cover .subtitle {
      font-size: 1rem;
      color: var(--muted);
      max-width: 580px;
      margin-bottom: 32px;
    }

    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 20px 40px;
      border-top: 1px solid var(--border);
      padding-top: 22px;
    }

    .meta-item dt {
      font-size: 11px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 2px;
    }

    .meta-item dd {
      font-size: 13.5px;
      color: var(--text);
    }

    /* ── Table of Contents ────────────────────────────────────────────── */
    .toc {
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      background: var(--surface);
      padding: 22px 28px;
      margin-bottom: 56px;
    }

    .toc h2 {
      font-size: 0.7rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 14px;
      font-family: var(--body-font);
      font-weight: 600;
    }

    .toc ol {
      padding-left: 18px;
      display: grid;
      gap: 7px;
    }

    .toc a {
      color: var(--accent2);
      text-decoration: none;
      font-size: 14px;
    }

    .toc a:hover { text-decoration: underline; }

    /* ── Sections ─────────────────────────────────────────────────────── */
    section { margin-bottom: 56px; }

    h2.section-heading {
      font-family: var(--heading-font);
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text);
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 22px;
    }

    h3.sub-heading {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text);
      margin: 26px 0 10px;
    }

    p { margin-bottom: 14px; color: #cdd9e5; }

    /* ── Cards / Feature Grid ─────────────────────────────────────────── */
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 8px;
    }

    .card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px 22px;
    }

    .card .card-label {
      font-size: 10.5px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .card .card-title {
      font-size: 15px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 8px;
    }

    .card p {
      font-size: 13.5px;
      line-height: 1.6;
      margin: 0;
      color: var(--muted);
    }

    /* ── Requirement Table ────────────────────────────────────────────── */
    .req-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      margin-top: 8px;
    }

    .req-table thead tr {
      background: var(--surface2);
    }

    .req-table th {
      text-align: left;
      padding: 9px 14px;
      font-size: 11px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--muted);
      border: 1px solid var(--border);
    }

    .req-table td {
      padding: 10px 14px;
      border: 1px solid var(--border);
      vertical-align: top;
      color: #cdd9e5;
      line-height: 1.55;
    }

    .req-table tr:nth-child(even) td { background: var(--surface); }

    /* ── Status Badge ─────────────────────────────────────────────────── */
    .badge {
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.04em;
      padding: 2px 9px;
      border-radius: 20px;
      white-space: nowrap;
    }

    .badge-done    { background: rgba(63,185,80,.15);  color: var(--green);  border: 1px solid rgba(63,185,80,.35);  }
    .badge-partial { background: rgba(210,153,34,.12); color: var(--orange); border: 1px solid rgba(210,153,34,.35); }
    .badge-planned { background: rgba(47,129,247,.12); color: var(--accent2);border: 1px solid rgba(47,129,247,.35); }

    /* ── Code block ───────────────────────────────────────────────────── */
    pre {
      background: #010409;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px 22px;
      overflow-x: auto;
      font-family: var(--mono-font);
      font-size: 13px;
      line-height: 1.6;
      margin: 12px 0 20px;
      color: #e6edf3;
    }

    code {
      font-family: var(--mono-font);
      font-size: 13px;
      background: rgba(110,118,129,.2);
      border-radius: 4px;
      padding: 2px 6px;
      color: #e6edf3;
    }

    pre code { background: none; padding: 0; }

    /* ── Callout ──────────────────────────────────────────────────────── */
    .callout {
      border-radius: 8px;
      padding: 16px 20px;
      font-size: 13.5px;
      line-height: 1.6;
      margin: 16px 0;
    }

    .callout p { margin: 0; }

    .callout-note   { background: rgba(47,129,247,.08);  border-left: 3px solid var(--accent); }
    .callout-warn   { background: rgba(210,153,34,.08);  border-left: 3px solid var(--orange); }
    .callout-info   { background: rgba(163,113,247,.08); border-left: 3px solid var(--purple); }

    .callout-note p, .callout-note strong { color: #b0cdf7; }
    .callout-warn p, .callout-warn strong { color: #e3c06b; }
    .callout-info p, .callout-info strong { color: #c9b3f7; }

    /* ── Tech Stack Pills ─────────────────────────────────────────────── */
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 12px 0;
    }

    .pill {
      font-size: 12.5px;
      padding: 5px 14px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: var(--surface2);
      color: var(--text);
    }

    /* ── Architecture Diagram (text) ──────────────────────────────────── */
    .arch-box {
      border: 1px solid var(--border);
      background: #010409;
      border-radius: 8px;
      padding: 20px 24px;
      font-family: var(--mono-font);
      font-size: 12.5px;
      color: #8b949e;
      line-height: 2;
    }

    .arch-box .hi { color: var(--accent2); }
    .arch-box .lo { color: var(--muted); }

    /* ── List ─────────────────────────────────────────────────────────── */
    ul.styled, ol.styled {
      padding-left: 20px;
      display: grid;
      gap: 6px;
      margin: 8px 0;
    }

    ul.styled li, ol.styled li {
      color: #cdd9e5;
      font-size: 14px;
      line-height: 1.65;
    }

    /* ── Footer ───────────────────────────────────────────────────────── */
    footer {
      border-top: 1px solid var(--border);
      margin-top: 80px;
      padding-top: 24px;
      font-size: 12.5px;
      color: var(--muted);
      text-align: center;
    }
  </style>
</head>
<body>
<div class="wrapper">

  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- COVER                                                              -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <div class="cover">
    <p class="institution">Cairo University &mdash; Faculty of Computer and Artificial Intelligence</p>
    <p class="course">IS424 &mdash; Selected Topics in Data Engineering &mdash; Spring 2026</p>

    <h1>Automated Machine Learning Platform</h1>

    <p class="subtitle">
      An end-to-end AutoML application that enables non-technical users to upload
      raw datasets, select a machine learning task, and receive a fully trained,
      evaluated, and exportable model — without writing a single line of code.
    </p>

    <dl class="meta-row">
      <div class="meta-item">
        <dt>Submission Deadline</dt>
        <dd>2 May 2026</dd>
      </div>
      <div class="meta-item">
        <dt>Course Assignment</dt>
        <dd>Project &mdash; Assignment 2</dd>
      </div>
      <div class="meta-item">
        <dt>Backend Framework</dt>
        <dd>FastAPI (Python)</dd>
      </div>
      <div class="meta-item">
        <dt>Frontend Framework</dt>
        <dd>Streamlit (Python)</dd>
      </div>
      <div class="meta-item">
        <dt>Primary Language</dt>
        <dd>Python 3.10+</dd>
      </div>
    </dl>
  </div>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- TABLE OF CONTENTS                                                  -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <nav class="toc">
    <h2>Contents</h2>
    <ol class="styled">
      <li><a href="#overview">Project Overview</a></li>
      <li><a href="#architecture">System Architecture</a></li>
      <li><a href="#tech-stack">Technology Stack</a></li>
      <li><a href="#requirements">Requirement Coverage</a></li>
      <li><a href="#frontend">Frontend — Streamlit Application</a></li>
      <li><a href="#backend">Backend — FastAPI Service</a></li>
      <li><a href="#pipeline">Data &amp; Modeling Pipeline</a></li>
      <li><a href="#evaluation">Model Evaluation</a></li>
      <li><a href="#export">Model Export &amp; Download</a></li>
      <li><a href="#planned">Planned Enhancements</a></li>
      <li><a href="#setup">Setup &amp; Running the Application</a></li>
      <li><a href="#structure">Repository Structure</a></li>
    </ol>
  </nav>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 1. OVERVIEW                                                        -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="overview">
    <h2 class="section-heading">1. Project Overview</h2>

    <p>
      This project delivers a browser-based Automated Machine Learning (AutoML) platform
      built as the capstone for IS424. The platform is composed of two independent processes
      that communicate over HTTP: a <strong>Streamlit frontend</strong> that handles the user
      interface and a <strong>FastAPI backend</strong> that manages all data processing,
      model training, and serialisation logic.
    </p>

    <p>
      The design goal is zero friction for the end user. A person with no programming
      background can open the browser, upload a CSV or Excel file, choose the type of
      prediction they need — classification, regression, or clustering — and within seconds
      receive a comparative evaluation across multiple algorithms together with a downloadable
      trained model artefact.
    </p>

    <p>
      All heavy computation is intentionally isolated inside the backend service. The frontend
      remains a lightweight reporting and control surface, communicating with the backend
      through a well-defined REST API. This separation of concerns keeps the codebase readable,
      testable, and extensible.
    </p>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 2. ARCHITECTURE                                                    -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="architecture">
    <h2 class="section-heading">2. System Architecture</h2>

    <p>
      The system follows a client–server architecture where the Streamlit process acts as the
      client and the FastAPI process acts as the server. Both processes run locally on the same
      machine during development, communicating over <code>localhost</code> on their respective
      default ports.
    </p>

    <div class="arch-box">
<span class="lo">┌─────────────────────────────────────────────────────────────────────┐</span>
<span class="lo">│</span>                         <span class="hi">Browser (User)</span>                               <span class="lo">│</span>
<span class="lo">│                         http://localhost:8501                         │</span>
<span class="lo">└──────────────────────────────────┬──────────────────────────────────┘</span>
                                   <span class="lo">│  HTTP</span>
<span class="lo">┌──────────────────────────────────▼──────────────────────────────────┐</span>
<span class="lo">│</span>                     <span class="hi">Streamlit Frontend</span>  (frontend.py)               <span class="lo">│</span>
<span class="lo">│  • File upload UI         • Task &amp; target selection                  │</span>
<span class="lo">│  • Dataset quality report • Metric display &amp; model download          │</span>
<span class="lo">└──────────────────────────────────┬──────────────────────────────────┘</span>
                                   <span class="lo">│  HTTP POST /train</span>
                                   <span class="lo">│  HTTP GET  /download/{model_id}</span>
<span class="lo">┌──────────────────────────────────▼──────────────────────────────────┐</span>
<span class="lo">│</span>                     <span class="hi">FastAPI Backend</span>  (backend.py)                    <span class="lo">│</span>
<span class="lo">│  • Data ingestion &amp; validation    • Preprocessing pipeline           │</span>
<span class="lo">│  • Multi-algorithm training       • Evaluation &amp; best-model selection│</span>
<span class="lo">│  • Joblib serialisation           • Model file-system storage        │</span>
<span class="lo">└──────────────────────────────────┬──────────────────────────────────┘</span>
                                   <span class="lo">│</span>
<span class="lo">                    ┌──────────────▼──────────────┐</span>
<span class="lo">                    │</span>     <span class="hi">models/</span>  directory      <span class="lo">│</span>
<span class="lo">                    │  .joblib serialised artefacts │</span>
<span class="lo">                    └─────────────────────────────┘</span>
    </div>

    <div class="callout callout-note" style="margin-top:18px;">
      <p><strong>Communication Protocol:</strong> The frontend transmits the raw file as
      <code>multipart/form-data</code> alongside form fields for the task type and target column.
      The backend processes the request synchronously and returns a JSON payload containing
      the chosen algorithm name, all computed metrics, and a unique model identifier used
      for the subsequent download request.</p>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 3. TECH STACK                                                      -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="tech-stack">
    <h2 class="section-heading">3. Technology Stack</h2>

    <div class="card-grid">
      <div class="card">
        <div class="card-label">Frontend</div>
        <div class="card-title">Streamlit</div>
        <p>Rapid Python-native web application framework. Provides reactive UI components with no JavaScript required.</p>
      </div>
      <div class="card">
        <div class="card-label">Backend</div>
        <div class="card-title">FastAPI</div>
        <p>High-performance ASGI framework built on Starlette and Pydantic. Provides automatic OpenAPI documentation and async support.</p>
      </div>
      <div class="card">
        <div class="card-label">Data Processing</div>
        <div class="card-title">Pandas &amp; NumPy</div>
        <p>Industry-standard libraries for tabular data manipulation, statistical summaries, and numerical computation.</p>
      </div>
      <div class="card">
        <div class="card-label">Machine Learning</div>
        <div class="card-title">Scikit-learn</div>
        <p>Encompasses the preprocessing transformers, algorithm implementations, evaluation metrics, and Pipeline API used throughout the backend.</p>
      </div>
      <div class="card">
        <div class="card-label">Model Serialisation</div>
        <div class="card-title">Joblib</div>
        <p>Efficient binary serialisation of NumPy-heavy objects. Faster and more memory-efficient than pickle for large estimator artefacts.</p>
      </div>
      <div class="card">
        <div class="card-label">Server Runtime</div>
        <div class="card-title">Uvicorn</div>
        <p>ASGI server used to host the FastAPI application. Supports hot-reload during development and production deployment.</p>
      </div>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 4. REQUIREMENT COVERAGE                                            -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="requirements">
    <h2 class="section-heading">4. Requirement Coverage</h2>

    <p>
      The table below maps each item in the official IS424 project specification to its
      implementation status and the corresponding location in the codebase.
    </p>

    <table class="req-table">
      <thead>
        <tr>
          <th>Ref.</th>
          <th>Requirement</th>
          <th>Status</th>
          <th>Location</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>A</td>
          <td>File upload supporting .csv and .xlsx formats with data preview</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Data Ingestion section</td>
        </tr>
        <tr>
          <td>B.i – B.iii</td>
          <td>ML task selection via radio group (Classification, Regression, Clustering)</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Task Selection section</td>
        </tr>
        <tr>
          <td>B — supervised</td>
          <td>Target column selection populated from the uploaded dataset's column names</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Task Selection section</td>
        </tr>
        <tr>
          <td>C.i</td>
          <td>Automated handling of missing values (mean imputation for numeric; most-frequent for categorical)</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — <code>SimpleImputer</code> inside preprocessing pipeline</td>
        </tr>
        <tr>
          <td>C.ii</td>
          <td>Encoding of categorical variables using one-hot encoding</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — <code>OneHotEncoder</code> inside ColumnTransformer</td>
        </tr>
        <tr>
          <td>C.iii</td>
          <td>Normalisation / scaling of numerical features using StandardScaler</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — <code>StandardScaler</code> inside numeric pipeline</td>
        </tr>
        <tr>
          <td>C.iv</td>
          <td>Class imbalance detection and random oversampling for Classification tasks</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — Resampling block above preprocessor construction</td>
        </tr>
        <tr>
          <td>D</td>
          <td>80 / 20 train–test split; training of at least two algorithms per task; automatic best-model selection</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — Model Training block</td>
        </tr>
        <tr>
          <td>E.i</td>
          <td>Classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — Classification results dict</td>
        </tr>
        <tr>
          <td>E.ii</td>
          <td>Regression metrics: MAE, MSE, R² Score</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — Regression results dict</td>
        </tr>
        <tr>
          <td>E.iii</td>
          <td>Clustering metric: Silhouette Score</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code> — Clustering results dict</td>
        </tr>
        <tr>
          <td>F</td>
          <td>Readable metric report displayed in the frontend; confusion matrix rendered as a dataframe</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Model Evaluation Results section</td>
        </tr>
        <tr>
          <td>F — export</td>
          <td>Save Model button that downloads the trained pipeline as a .joblib file</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Export Model section; <code>backend.py</code> — <code>/download</code> endpoint</td>
        </tr>
        <tr>
          <td>G</td>
          <td>FastAPI REST endpoints for training (<code>POST /train</code>) and model retrieval (<code>GET /download/{id}</code>)</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>backend.py</code></td>
        </tr>
        <tr>
          <td>G — HTTP</td>
          <td>Frontend communicates with backend exclusively over HTTP using the <code>requests</code> library</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — pipeline submission block</td>
        </tr>
        <tr>
          <td>—</td>
          <td>Dataset Quality Report: missing value counts, duplicate row detection, feature scale divergence warning</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Dataset Quality Report section</td>
        </tr>
        <tr>
          <td>—</td>
          <td>Class imbalance visualisation (bar chart + ratio warning) on the frontend before training</td>
          <td><span class="badge badge-done">Complete</span></td>
          <td><code>frontend.py</code> — Target Variable Preview section</td>
        </tr>
      </tbody>
    </table>

    <div class="callout callout-warn" style="margin-top:18px;">
      <p><strong>Planned items below are not yet implemented.</strong> See Section 10 for a
      detailed roadmap of enhancements that extend the current baseline.</p>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 5. FRONTEND                                                        -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="frontend">
    <h2 class="section-heading">5. Frontend — Streamlit Application</h2>

    <p>
      The frontend is a single-file Streamlit application (<code>frontend.py</code>) that
      walks the user through the AutoML workflow in a top-to-bottom sequential layout.
      No additional configuration files or static assets are required.
    </p>

    <h3 class="sub-heading">5.1 Data Ingestion Panel</h3>
    <p>
      The application opens with a file uploader that accepts <code>.csv</code> and
      <code>.xlsx</code> files. Upon successful upload, Pandas reads the file into a
      DataFrame and the interface immediately surfaces:
    </p>
    <ul class="styled">
      <li>A data preview table showing the first five rows and overall shape (rows × columns).</li>
      <li>A Dataset Quality Report, split into two columns — missing value counts per column on the left and duplicate row detection on the right.</li>
      <li>A feature-scale divergence warning when the ratio of the maximum to minimum numeric range exceeds 10, informing the user that StandardScaler will be applied automatically.</li>
      <li>A column data-type summary table.</li>
    </ul>

    <h3 class="sub-heading">5.2 Task Selection Panel</h3>
    <p>
      A radio group allows the user to choose one of three machine learning problem types.
      For <em>Classification</em> and <em>Regression</em>, a dependent dropdown appears and
      is populated with the uploaded dataset's column names, prompting target variable
      selection. A bar chart and imbalance ratio warning are rendered for classification
      targets; summary statistics and a line chart of the first 100 values are shown for
      regression targets.
    </p>

    <h3 class="sub-heading">5.3 Pipeline Submission</h3>
    <p>
      A primary-styled "Start Automated ML Pipeline" button triggers client-side
      validation before dispatching the request. If no dataset has been uploaded, or if
      a target column is missing for a supervised task, an error is surfaced immediately
      without contacting the backend. On a valid submission, the file and form parameters
      are forwarded to <code>POST http://localhost:8000/train</code> as multipart form data.
    </p>

    <h3 class="sub-heading">5.4 Results &amp; Model Export</h3>
    <p>
      After a successful training response, three metric cards are rendered in a three-column
      layout alongside task-specific displays (confusion matrix for classification). A
      separate "Download Trained Model" button issues a <code>GET</code> request to the
      backend's download endpoint and converts the binary response to an in-page anchor tag
      using Base64 encoding, allowing the user to save the <code>.joblib</code> file without
      any server-side redirect.
    </p>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 6. BACKEND                                                         -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="backend">
    <h2 class="section-heading">6. Backend — FastAPI Service</h2>

    <p>
      The backend (<code>backend.py</code>) is a self-contained FastAPI application exposing
      two HTTP endpoints. All model artefacts are written to the <code>models/</code>
      directory, which is created automatically on first launch.
    </p>

    <h3 class="sub-heading">6.1 POST /train</h3>
    <p>
      Accepts a multipart form containing the dataset file, the selected task type string,
      and an optional target column name. Executes the full preprocessing and training
      pipeline (described in Section 7), persists the best estimator, and returns:
    </p>
    <ul class="styled">
      <li><code>status</code> — always <code>"success"</code> on a 200 response.</li>
      <li><code>model_id</code> — a UUID v4 string uniquely identifying the saved artefact.</li>
      <li><code>metrics</code> — a dictionary of task-appropriate evaluation scores.</li>
      <li><code>task_type</code> — echoes the submitted task type for frontend awareness.</li>
    </ul>

    <h3 class="sub-heading">6.2 GET /download/{model_id}</h3>
    <p>
      Looks up the model file path by the provided UUID, and returns it as a streaming
      binary response (<code>application/octet-stream</code>) with a descriptive filename.
      Returns HTTP 404 if the model ID does not exist in the in-memory registry.
    </p>

    <div class="callout callout-info">
      <p><strong>In-memory Registry:</strong> Model paths are stored in a Python dictionary
      (<code>models_db</code>) for the lifetime of the server process. In a production
      deployment this would be replaced with a persistent database to survive process
      restarts. The actual model files on disk are not affected by a restart.</p>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 7. DATA & MODELING PIPELINE                                        -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="pipeline">
    <h2 class="section-heading">7. Data &amp; Modeling Pipeline</h2>

    <p>
      All preprocessing is implemented as a Scikit-learn <code>Pipeline</code> and
      <code>ColumnTransformer</code> combination, ensuring that the exact same
      transformations learned on the training split are applied to the test split without
      any data leakage.
    </p>

    <h3 class="sub-heading">7.1 Preprocessing</h3>

    <table class="req-table">
      <thead>
        <tr>
          <th>Column Type</th>
          <th>Step 1 — Imputation</th>
          <th>Step 2 — Transformation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Numeric (<code>int64</code>, <code>float64</code>)</td>
          <td>Mean imputation (<code>SimpleImputer</code>)</td>
          <td>Z-score standardisation (<code>StandardScaler</code>)</td>
        </tr>
        <tr>
          <td>Categorical (<code>object</code>, <code>category</code>)</td>
          <td>Most-frequent imputation (<code>SimpleImputer</code>)</td>
          <td>One-hot encoding (<code>OneHotEncoder</code>, unknown categories ignored)</td>
        </tr>
      </tbody>
    </table>

    <h3 class="sub-heading">7.2 Class Imbalance Handling (Classification Only)</h3>
    <p>
      Prior to constructing the preprocessor, the backend checks whether the ratio of the
      most-frequent to least-frequent class exceeds 1.5. If so, random oversampling with
      replacement is applied to minority classes until all classes reach parity with the
      majority class. This resampling is performed on the raw (pre-scaling) data, and the
      resampled set is then fed into the standard preprocessing pipeline.
    </p>

    <h3 class="sub-heading">7.3 Algorithm Selection</h3>
    <p>
      Two candidate estimators are trained per task. The selection criterion varies by task:
    </p>

    <table class="req-table">
      <thead>
        <tr>
          <th>Task</th>
          <th>Candidates</th>
          <th>Selection Criterion</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Classification</td>
          <td>Random Forest Classifier, Logistic Regression</td>
          <td>Weighted F1-Score (higher is better)</td>
        </tr>
        <tr>
          <td>Regression</td>
          <td>Random Forest Regressor, Linear Regression</td>
          <td>Mean Squared Error (lower is better)</td>
        </tr>
        <tr>
          <td>Clustering</td>
          <td>K-Means (k=3), Agglomerative Clustering (k=3)</td>
          <td>Silhouette Score (higher is better)</td>
        </tr>
      </tbody>
    </table>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 8. EVALUATION                                                      -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="evaluation">
    <h2 class="section-heading">8. Model Evaluation</h2>

    <p>
      Once the best model is selected, the complete set of task-appropriate metrics is
      computed and returned. The table below lists each metric alongside its interpretation.
    </p>

    <h3 class="sub-heading">Classification</h3>
    <table class="req-table">
      <thead><tr><th>Metric</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td>Accuracy</td><td>Proportion of all test samples correctly classified.</td></tr>
        <tr><td>Weighted Precision</td><td>Positive predictive value, averaged by class support.</td></tr>
        <tr><td>Weighted Recall</td><td>True positive rate, averaged by class support.</td></tr>
        <tr><td>Weighted F1-Score</td><td>Harmonic mean of precision and recall. Used as primary selection criterion.</td></tr>
        <tr><td>Confusion Matrix</td><td>N × N matrix of per-class prediction counts, rendered as an interactive DataFrame.</td></tr>
      </tbody>
    </table>

    <h3 class="sub-heading">Regression</h3>
    <table class="req-table">
      <thead><tr><th>Metric</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td>MAE</td><td>Mean Absolute Error — average magnitude of residuals.</td></tr>
        <tr><td>MSE</td><td>Mean Squared Error — penalises large residuals more heavily. Used as selection criterion.</td></tr>
        <tr><td>R² Score</td><td>Coefficient of determination. Proportion of variance in the target explained by the model.</td></tr>
      </tbody>
    </table>

    <h3 class="sub-heading">Clustering</h3>
    <table class="req-table">
      <thead><tr><th>Metric</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td>Silhouette Score</td><td>Measures how similar each point is to its own cluster compared to other clusters. Ranges from −1 to +1; higher values indicate denser, better-separated clusters.</td></tr>
      </tbody>
    </table>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 9. EXPORT                                                          -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="export">
    <h2 class="section-heading">9. Model Export &amp; Download</h2>

    <p>
      After a successful training run, the entire estimator — including the fitted
      preprocessing pipeline — is serialised to disk using <code>joblib.dump</code> under
      a UUID-named file in the <code>models/</code> directory. The frontend presents a
      "Download Trained Model (.joblib)" button. When clicked, it calls the
      <code>GET /download/{model_id}</code> endpoint, receives the binary stream, encodes
      it with Base64, and injects a data-URI anchor tag into the page so the browser can
      save the file locally.
    </p>

    <p>
      The downloaded <code>.joblib</code> artefact can be reloaded in any Python environment
      that has Scikit-learn installed:
    </p>

    <pre><code>import joblib

model = joblib.load("model_&lt;uuid&gt;.joblib")

# For Classification / Regression pipelines
predictions = model.predict(new_dataframe)

# For Clustering (stored as a tuple: preprocessor, algorithm)
preprocessor, algorithm = model
X_transformed = preprocessor.transform(new_dataframe)
labels = algorithm.fit_predict(X_transformed)</code></pre>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 10. PLANNED ENHANCEMENTS                                           -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="planned">
    <h2 class="section-heading">10. Planned Enhancements</h2>

    <p>
      The following capabilities are within scope for the project but have not yet been
      implemented. They represent the next development iteration.
    </p>

    <div class="card-grid">
      <div class="card">
        <div class="card-label">Planned</div>
        <div class="card-title">Feature Importance Visualisation</div>
        <p>For tree-based models (Random Forest), render a horizontal bar chart of the top-N most influential features to aid interpretability.</p>
      </div>
      <div class="card">
        <div class="card-label">Planned</div>
        <div class="card-title">Hyperparameter Tuning</div>
        <p>Integrate a lightweight grid search or random search over key hyperparameters (e.g., <code>n_estimators</code>, <code>max_depth</code>) to push model performance beyond baseline defaults.</p>
      </div>
      <div class="card">
        <div class="card-label">Planned</div>
        <div class="card-title">Additional Algorithms</div>
        <p>Extend the candidate set to include Support Vector Machines, Gradient Boosting (XGBoost / LightGBM), and DBSCAN for clustering.</p>
      </div>
      <div class="card">
        <div class="card-label">Planned</div>
        <div class="card-title">Persistent Model Registry</div>
        <p>Replace the in-memory <code>models_db</code> dictionary with an SQLite-backed registry so that model IDs survive server restarts and can be browsed through a dedicated endpoint.</p>
      </div>
      <div class="card">
        <div class="card-label">Planned</div>
        <div class="card-title">Cluster Visualisation</div>
        <p>Reduce cluster assignments to two dimensions using PCA and render a scatter plot coloured by cluster label, improving interpretability for unsupervised results.</p>
      </div>
      <div class="card">
        <div class="card-label">Planned</div>
        <div class="card-title">Automated k Selection (Clustering)</div>
        <p>Implement elbow-method and silhouette-score sweeps over a range of k values to determine the optimal number of clusters rather than defaulting to three.</p>
      </div>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 11. SETUP                                                          -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="setup">
    <h2 class="section-heading">11. Setup &amp; Running the Application</h2>

    <h3 class="sub-heading">11.1 Prerequisites</h3>
    <div class="pill-row">
      <span class="pill">Python 3.10+</span>
      <span class="pill">pip</span>
      <span class="pill">Virtual environment (recommended)</span>
    </div>

    <h3 class="sub-heading">11.2 Install Dependencies</h3>
    <p>All runtime dependencies are listed in <code>requirements_temp.txt</code>. Install them with:</p>
    <pre><code>pip install fastapi uvicorn streamlit pandas numpy scikit-learn joblib openpyxl requests</code></pre>

    <h3 class="sub-heading">11.3 Start the Backend</h3>
    <p>Open a terminal in the project root and run:</p>
    <pre><code>uvicorn backend:app --reload --host 0.0.0.0 --port 8000</code></pre>
    <p>
      The FastAPI service will be available at <code>http://localhost:8000</code>.
      Interactive API documentation is automatically available at
      <code>http://localhost:8000/docs</code>.
    </p>

    <h3 class="sub-heading">11.4 Start the Frontend</h3>
    <p>Open a second terminal in the project root and run:</p>
    <pre><code>streamlit run frontend.py</code></pre>
    <p>
      Streamlit will open the browser automatically at <code>http://localhost:8501</code>.
    </p>

    <div class="callout callout-note">
      <p><strong>Order matters:</strong> The backend must be running before the frontend
      attempts to submit the pipeline. The frontend will display a descriptive error message
      if it cannot reach <code>localhost:8000</code>.</p>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- 12. STRUCTURE                                                       -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <section id="structure">
    <h2 class="section-heading">12. Repository Structure</h2>

    <pre><code>DE-Project/
├── backend.py                   # FastAPI application — all ML logic and API endpoints
├── frontend.py                  # Streamlit application — user interface
├── models/                      # Runtime directory for serialised .joblib artefacts
│   └── &lt;uuid&gt;.joblib            # Generated at runtime; not committed to source control
├── IS424_Project Requirements-3.pdf   # Original project specification
├── requirements_temp.txt        # Python package requirements
├── README.html                  # This document
└── .gitignore                   # Standard Python ignore rules</code></pre>
  </section>


  <!-- ══════════════════════════════════════════════════════════════════ -->
  <!-- FOOTER                                                             -->
  <!-- ══════════════════════════════════════════════════════════════════ -->
  <footer>
    <p>
      IS424 &mdash; Selected Topics in Data Engineering &mdash; Spring 2026 &mdash;
      Cairo University, Faculty of Computer and Artificial Intelligence
    </p>
  </footer>

</div>
</body>
</html>
