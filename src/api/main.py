# src/api/main.py

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schema import StrokeInput, StrokeOutput
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

logger = logging.getLogger(__name__)

# ── Global state ──
_model = None
_optimal_threshold: float = 0.5


def _load_model():
    """Load CatBoost model — standalone .cbm file first, MLflow registry fallback."""
    import json
    import os

    from catboost import CatBoostClassifier

    from src.config import MODEL_FILE_PATH, MODEL_METADATA_PATH

    # ── Try standalone .cbm file (Docker / Streamlit Cloud) ──
    if os.path.exists(MODEL_FILE_PATH):
        logger.info("Loading standalone model from %s", MODEL_FILE_PATH)
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE_PATH)
        threshold = 0.5
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH) as f:
                threshold = json.load(f).get("threshold", 0.5)
        return model, threshold

    # ── Fallback: MLflow Registry (local dev) ──
    import mlflow
    import mlflow.catboost

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            run = client.get_run(latest.run_id)
            threshold = run.data.metrics.get("threshold", 0.5)
        else:
            threshold = 0.5
    except Exception:
        threshold = 0.5

    model_uri = f"models:/{MODEL_NAME}/None"
    logger.info("Loading model from MLflow: %s (threshold=%.4f)", model_uri, threshold)
    model = mlflow.catboost.load_model(model_uri)
    return model, threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup."""
    global _model, _optimal_threshold
    try:
        _model, _optimal_threshold = _load_model()
        logger.info("Model loaded (threshold=%.4f)", _optimal_threshold)
    except Exception as e:
        logger.error("Could not load model: %s", e)
        _model = None
    yield
    _model = None


app = FastAPI(
    title="Stroke Risk Prediction API",
    description="CatBoost-based binary classification for stroke risk — with threshold tuning and SHAP",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Prometheus Instrumentation ──
Instrumentator().instrument(app).expose(app)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy" if _model is not None else "model_not_loaded",
        "model": MODEL_NAME,
        "threshold": _optimal_threshold,
    }


@app.post("/predict", response_model=StrokeOutput)
def predict(data: StrokeInput):
    """Predict stroke risk using the optimized threshold."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_dict = data.model_dump()
    if input_dict.get("bmi") is None:
        input_dict["bmi"] = 28.9

    df = pd.DataFrame([input_dict])
    
    # ── Apply Feature Engineering ──
    from src.data.preprocess import create_features
    df = create_features(df)

    probability = float(_model.predict_proba(df)[:, 1][0])
    prediction = 1 if probability >= _optimal_threshold else 0

    return StrokeOutput(
        prediction=prediction,
        probability_stroke=round(probability, 4),
    )


@app.post("/explain")
def explain(data: StrokeInput):
    """Return SHAP values for a single prediction."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import shap
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP not installed")

    input_dict = data.model_dump()
    if input_dict.get("bmi") is None:
        input_dict["bmi"] = 28.9

    df = pd.DataFrame([input_dict])
    from src.data.preprocess import create_features
    df = create_features(df)

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_contributions = {
        col: round(float(val), 4)
        for col, val in zip(df.columns, shap_values[0])
    }

    probability = float(_model.predict_proba(df)[:, 1][0])

    return {
        "probability_stroke": round(probability, 4),
        "prediction": 1 if probability >= _optimal_threshold else 0,
        "shap_values": feature_contributions,
        "base_value": round(float(explainer.expected_value
                                   if not isinstance(explainer.expected_value, list)
                                   else explainer.expected_value[1]), 4),
    }
