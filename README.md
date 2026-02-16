# 🧠 Medical Risk MLOps — Stroke Risk Prediction

An end-to-end **stroke risk prediction** system built with CatBoost + MLflow + FastAPI + Streamlit.

---

## 📋 Table of Contents

- [Project Structure](#-project-structure)
- [Architecture & Design Decisions](#-architecture--design-decisions)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Running with Docker](#-running-with-docker)
- [Deploy to Streamlit Cloud](#-deploy-to-streamlit-cloud)
- [API Usage](#-api-usage)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)

---

## 🗂 Project Structure

```
medical-risk-mlops/
│
├── data/
│   └── raw/stroke_data.csv          # Kaggle Stroke Prediction dataset (5,110 records)
│
├── models/
│   ├── model.cbm                    # Standalone CatBoost model (for Docker/Cloud)
│   └── metadata.json                # Optimal threshold & model metadata
│
├── src/
│   ├── config.py                    # Central configuration: paths, hyperparams, features
│   │
│   ├── data/
│   │   ├── load_data.py             # CSV loader
│   │   ├── validate.py              # Data validation (missing values, dtype checks)
│   │   └── preprocess.py            # Cleaning, BMI imputation, train/test split
│   │
│   ├── models/
│   │   ├── train.py                 # Model training + MLflow logging + SHAP
│   │   ├── evaluate.py              # Metrics + threshold tuning
│   │   └── predict.py               # Model loading + inference
│   │
│   └── api/
│       ├── schema.py                # Pydantic input/output schemas
│       └── main.py                  # FastAPI endpoints (/predict, /explain, /health)
│
├── .streamlit/config.toml           # Streamlit theme & server settings
├── streamlit_app.py                 # 🖥 Streamlit dashboard (visual interface)
├── Dockerfile                       # Multi-stage Docker container
├── docker-compose.yml               # 3 services: API + Streamlit + MLflow UI
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🎯 Architecture & Design Decisions

### 1. `config.py` — Central Configuration

**Why:** All hyperparameters, file paths, and feature names are stored in one place. Need to change something? Edit a single file.

### 2. `load_data.py` → `validate.py` → `preprocess.py` — Data Pipeline

**Why:** Separating data loading → validation → cleaning into distinct modules makes each independently testable and replaceable.

| Step            | What It Does                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------- |
| `load_data.py`  | Reads the raw CSV file                                                                       |
| `validate.py`   | Checks for missing values, target distribution, and dtype consistency                        |
| `preprocess.py` | Drops the `id` column, imputes missing BMI with median, performs stratified train/test split |

### 3. `train.py` — Model Training + MLflow

**Why:** Trains a CatBoost model and records everything to MLflow for reproducibility.

**What gets logged:**

- Hyperparameters (iterations, depth, learning_rate, class_weights)
- Metrics: ROC-AUC, Precision, Recall, F1, optimal threshold
- Artifacts: confusion matrix (PNG + JSON), classification report (TXT), SHAP summary plot (PNG)
- The model itself → registered in MLflow Model Registry
- Standalone export → `models/model.cbm` + `models/metadata.json` (for Docker/Cloud)

### 4. `evaluate.py` — Threshold Tuning

**Why:** The dataset is highly imbalanced (**95% no-stroke** vs **5% stroke**). The default 0.5 threshold misses too many stroke cases. We use **F2-score** to find the optimal threshold (≈0.69), which weighs recall more heavily.

### 5. `predict.py` — Model Loading & Inference

**Why:** Loads the model from the MLflow Registry and runs predictions for a single patient. Used by both the API and Streamlit.

### 6. `schema.py` — Pydantic Schemas

**Why:** Guarantees correct input data types for the API. Invalid types or missing fields return clear error messages.

### 7. `api/main.py` — FastAPI REST API

**Why:** Serves the model as an HTTP service. Any application (web, mobile, microservice) can call this API for predictions.

| Endpoint   | Method | Description                                 |
| ---------- | ------ | ------------------------------------------- |
| `/health`  | GET    | Health check & system status                |
| `/predict` | POST   | Stroke risk prediction for a single patient |
| `/explain` | POST   | SHAP-based prediction explanation           |

### 8. `streamlit_app.py` — Dashboard Interface

**Why:** A visual interface for non-technical users. Fill in patient info → get predictions → see which factors drive the risk via SHAP visualization.

### 9. Dual-Mode Model Loading

**Why:** After training, the model is saved in two locations:

1. **MLflow Registry** → for local development (alongside experiment tracking)
2. **`models/model.cbm`** → for Docker and Cloud deployment (no MLflow dependency)

Both the API and Streamlit check for the `.cbm` file first → fall back to MLflow if not found.

### 10. Class Imbalance Handling

**Why:** 4,861 no-stroke vs 249 stroke samples. We use `class_weights=[1, 20]` to tell CatBoost to treat stroke cases as 20x more important during training.

---

## ⚙️ Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Getting Started

### Step 1 — Train the Model

```bash
python -m src.models.train
```

**What happens:**

- Data is loaded and preprocessed
- CatBoost model is trained (500 iterations)
- Threshold is optimized for recall
- SHAP summary plot is generated
- Everything is logged to MLflow
- Model is registered in MLflow Registry
- `models/model.cbm` and `models/metadata.json` are exported

**Output:** `ROC-AUC: ~0.85 | Recall: ~0.74 | Threshold: ~0.69`

### Step 2a — Streamlit Dashboard (Recommended)

```bash
streamlit run streamlit_app.py --server.port 8890
```

Open **http://localhost:8890** in your browser.

> ⚠️ **Windows Hyper-V Note:** Port 8501 (default) may be blocked by Hyper-V. Use `--server.port 8890` to run on a different port.

### Step 2b — FastAPI API (Alternative)

```bash
uvicorn src.api.main:app --port 8000
```

API docs: **http://localhost:8000/docs** (Swagger UI)

### Step 3 — MLflow UI (Optional)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**http://localhost:5000** → Visually explore all experiments, metrics, and artifacts.

---

## 🐳 Running with Docker

### Single Service

```bash
# Build the image
docker build -t stroke-risk-mlops .

# Run FastAPI
docker run -p 8000:8000 stroke-risk-mlops uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run Streamlit
docker run -p 8501:8501 stroke-risk-mlops streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Compose (All 3 Services)

```bash
docker-compose up -d
```

| Service       | URL                   | Description            |
| ------------- | --------------------- | ---------------------- |
| **API**       | http://localhost:8000 | FastAPI REST endpoint  |
| **Streamlit** | http://localhost:8501 | Dashboard interface    |
| **MLflow**    | http://localhost:5000 | Experiment tracking UI |

```bash
# Stop
docker-compose down
```

> **Note:** Docker containers use the standalone `models/model.cbm` file (no MLflow registry dependency). This means a model trained on Windows runs seamlessly inside a Linux container.

---

## ☁️ Deploy to Streamlit Cloud

1. Push the project to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your GitHub repo → choose `streamlit_app.py`
4. Click **Deploy**

> **Important:** Make sure `models/model.cbm` and `models/metadata.json` are in the repo (not in `.gitignore`).

---

## 📡 API Usage

### Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

### Response

```json
{
  "prediction": 1,
  "probability_stroke": 0.8357
}
```

---

## 🛠 Tech Stack

| Technology       | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| **CatBoost**     | Gradient boosting with native categorical feature support |
| **MLflow**       | Experiment tracking, model registry, artifact storage     |
| **FastAPI**      | High-performance REST API with auto-generated docs        |
| **Streamlit**    | Interactive dashboard interface                           |
| **SHAP**         | Model explainability (which features drive predictions)   |
| **Pydantic**     | Input/output data validation for the API                  |
| **Docker**       | Portable containerized deployment                         |
| **scikit-learn** | Train/test split, metric computation                      |

---

## 📊 Model Performance

| Metric        | Value      |
| ------------- | ---------- |
| ROC-AUC       | **0.8485** |
| Recall        | **0.7400** |
| Threshold     | **0.6904** |
| Class Weights | [1, 20]    |

---

_Built with ❤️ using CatBoost + MLflow + FastAPI + Streamlit_
