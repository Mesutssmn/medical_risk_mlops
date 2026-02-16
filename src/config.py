# src/config.py
import os

# ── Paths ──────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/stroke_data.csv"
PROCESSED_DATA_PATH = "data/processed"

# ── Target ─────────────────────────────────────────────
TARGET_COLUMN = "stroke"

# ── Split ──────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Features ───────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

DROP_COLUMNS = ["id"]

# ── CatBoost Hyperparameters (Improved) ────────────────
CATBOOST_PARAMS = {
    "iterations": 1000,
    "depth": 8,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "class_weights": [1, 20],
    "verbose": 100,
    "early_stopping_rounds": 100,
    "l2_leaf_reg": 5,
    "border_count": 128,
}

# ── MLflow ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "stroke_prediction_catboost"
MODEL_NAME = "catboost_stroke_model"

# ── Standalone model (for Docker / Streamlit Cloud) ────
MODEL_DIR = "models"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.cbm")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
