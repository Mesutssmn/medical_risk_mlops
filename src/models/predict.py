# src/models/predict.py

import logging
import pandas as pd
import mlflow.catboost

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

logger = logging.getLogger(__name__)


def load_model_from_registry(stage: str = "None"):
    """
    Load a CatBoost model from the MLflow Model Registry.

    Parameters
    ----------
    stage : str
        Model stage — "None", "Staging", "Production", or a version number.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.catboost.load_model(model_uri)
    return model


def predict_single(model, input_data: dict) -> dict:
    """
    Run prediction on a single sample.

    Parameters
    ----------
    model : CatBoostClassifier
    input_data : dict with feature names as keys

    Returns
    -------
    dict with 'prediction' (int) and 'probability_stroke' (float)
    """
    df = pd.DataFrame([input_data])
    prediction = int(model.predict(df).flatten()[0])
    probability = float(model.predict_proba(df)[:, 1][0])

    return {
        "prediction": prediction,
        "probability_stroke": probability,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = load_model_from_registry()
    sample = {
        "gender": "Male",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }
    print(predict_single(m, sample))
