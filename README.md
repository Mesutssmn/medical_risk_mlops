# 🧠 Medical Risk MLOps — Stroke Risk Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medicalriskmlops-gazdhmhjpne5angult4why.streamlit.app/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

An end-to-end **MLOps** project for stroke risk prediction. This system demonstrates a complete machine learning lifecycle, from data processing to deployment, monitoring, and explainability.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture & Code Walkthrough](#-system-architecture--code-walkthrough)
- [Installation & Usage](#-installation--usage)
- [Monitoring & Observability](#-monitoring--observability)
- [Model Performance](#-model-performance)
- [Roadmap & Future Improvements](#-roadmap--future-improvements)

---

## 🎯 Project Overview

This project aims to predict the likelihood of a patient having a **stroke** based on 11 clinical features (Age, BMI, Glucose, Hypertension, etc.).

**Why MLOps?**
Unlike a simple Jupyter Notebook, this project is built as a **production-ready system**:

- **Reproducible**: Training with MLflow.
- **Scalable**: Containerized with Docker.
- **Interpretable**: Explains _why_ a prediction was made (SHAP).
- **Monitored**: Tracks system health (Prometheus) and data quality (Evidently).
- **Safe**: Prevents data leakage and validates inputs.

---

## ✨ Key Features

- **Model**: **CatBoost Classifier** (Gradient Boosting), optimized for categorical data and imbalanced datasets.
- **Scaling**: `RobustScaler` handles outliers in numerical features (Age, BMI, Glucose).
- **Explainability**: **SHAP (SHapley Additive exPlanations)** provides local and global feature importance.
- **API**: **FastAPI** serves predictions with high performance and auto-generated docs.
- **Frontend**: **Streamlit** dashboard for easy interaction and visualization.
- **Monitoring**:
  - **Prometheus**: Tracks API latency, request count, and errors.
  - **Evidently AI**: Generates Data Quality & Drift reports during training.
- **CI/CD**: GitHub Actions for automated testing and linting.

---

## 🏗 System Architecture & Code Walkthrough

This section explains "what code does what" to help you understand the flow.

### 1. Data Pipeline (`src/data/`)

- **`load_data.py`**: Loads raw CSV. **Crucial**: Automatically removes duplicate rows to prevent Data Leakage.
- **`preprocess.py`**:
  - Fills missing BMI values with median.
  - **Feature Engineering**: Creates Age Groups, BMI Categories, and Interaction Terms (e.g., `Age * BMI`).
  - **Scaling**: Applies `RobustScaler` to numerical features (fits on Train, transforms Test).
  - Splits data into Train/Test sets (Stratified).

### 2. Model Pipeline (`src/models/`)

- **`train.py`**: The heart of the training process.
  - Logs params, metrics, and artifacts to **MLflow**.
  - Generates **Evidently** Data Quality Reports.
  - Saves the model (`model.cbm`) and scaler (`scaler.pkl`) for deployment.
- **`evaluate.py`**: Because stroke is rare (imbalanced data), standard accuracy is misleading. This script finds the **Optimal Threshold** that maximizes the **F2-Score** (prioritizing Recall/Sensitivity to catch more cases).

### 3. API & Serving (`src/api/`)

- **`main.py`**: The FastAPI application.
  - Loads `model.cbm`, `scaler.pkl`, and `metadata.json` (threshold).
  - **`/predict`**: Returns class (0/1) and probability.
  - **`/metrics`**: Exposes Prometheus metrics.
- **`schema.py`**: Pydantic models ensuring input data validity (e.g., age must be positive).

### 4. User Interface (`streamlit_app.py`)

- Provides a clean web UI for doctors/users.
- Visualizes **Risk Levels** (Low/Moderate/High).
- Displays **SHAP Waterfall Plots** to explain individual predictions.

---

## ⚙️ Installation & Usage

### Method 1: Local (Python 3.12)

```bash
# 1. Clone & Setup
git clone https://github.com/Mesutssmn/medical-risk-mlops.git
cd medical-risk-mlops
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Running the App
streamlit run streamlit_app.py
```

### Method 2: Docker

```bash
docker-compose up -d
```

- **Streamlit**: http://localhost:8501
- **API**: http://localhost:8000
- **MLflow**: http://localhost:5000

---

## 📡 Monitoring & Observability

### 1. System Metrics (Prometheus)

The API exposes a `/metrics` endpoint compatible with Prometheus. It tracks:

- `http_requests_total`: Total number of predictions.
- `http_request_duration_seconds`: Latency distributions.

### 2. Data Quality (Evidently)

Every time `train.py` runs, an HTML report (`data_quality_report.html`) is generated and logged to MLflow. This helps detect:

- **Data Drift**: Is the new data significantly different from the old data?
- **Missing Values**: Are we seeing unexpected nulls?

---

## 📊 Model Performance

| Metric        | Value      | Description                                                |
| ------------- | ---------- | ---------------------------------------------------------- |
| **ROC-AUC**   | **0.8485** | Strong ability to distinguish stroke vs no-stroke.         |
| **Recall**    | **0.7400** | Catches 74% of actual stroke cases (critically important). |
| **Threshold** | **0.6904** | Optimized decision boundary.                               |

---

## 🔮 Roadmap & Future Improvements

To take this system to the "Enterprise Level", here is the recommended roadmap:

### 1. Advanced Monitoring Stack

- **Grafana Dashboard**: Visualize the Prometheus metrics (Latency, RPS) in real-time dashboards.
- **Alerting**: Set up **Alertmanager** to send Slack/Email notifications if Model Drift is detected or API errors spike.

### 2. Automated Retraining (Continuous Training)

- **Orchestration**: Use **Airflow** or **Prefect** to schedule `train.py` to run weekly automatically using new data.
- **Trigger**: Trigger retraining automatically if Evidently detects significant Data Drift.

### 3. Database Integration

- Currently, the API accepts data but doesn't save requests.
- **Improvement**: Save all incoming prediction requests and model results to a **PostgreSQL** database. This builds a "Ground Truth" dataset for future retraining.

### 4. Canary Deployment

- Use **Kubernetes (K8s)** or Docker Swarm to run two versions of the model simultaneously (v1 and v2) and gradually shift traffic to the new model (A/B Testing).

### 5. Security

- Add **API Key Authentication** to FastAPI endpoints.
- Implement Rate Limiting to prevent abuse.

---

_Built with ❤️ by Mesut_
