# src/models/train.py

import json
import logging
import os
import mlflow
import mlflow.catboost
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier

from src.config import (
    RAW_DATA_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    CATBOOST_PARAMS,
)
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"


def _save_confusion_matrix(cm, path):
    """Save confusion matrix as a heatmap image."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Stroke", "Stroke"])
    ax.set_yticklabels(["No Stroke", "Stroke"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_shap_summary(model, X_test, path):
    """Generate SHAP summary plot and save to disk."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, shap_values may be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary plot saved to %s", path)
        return True
    except Exception as e:
        logger.warning("SHAP plot generation failed: %s", e)
        return False


def train():
    """Train CatBoost model, evaluate, log to MLflow, and register."""

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── MLflow setup ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Data ──
    df = load_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test, categorical_features = preprocess(df)

    # ── Model ──
    model = CatBoostClassifier(**CATBOOST_PARAMS)

    with mlflow.start_run() as run:
        logger.info("MLflow Run ID: %s", run.info.run_id)

        # ── Log parameters ──
        mlflow.log_params(CATBOOST_PARAMS)

        # ── Train ──
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            eval_set=(X_test, y_test),
        )

        # ── Evaluate (with threshold tuning) ──
        metrics = evaluate_model(model, X_test, y_test)

        # ── Log scalar metrics ──
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("threshold", metrics["threshold"])

        # ── Log confusion matrix as JSON + image ──
        cm_json_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.json")
        with open(cm_json_path, "w") as f:
            json.dump(metrics["confusion_matrix"], f)
        mlflow.log_artifact(cm_json_path)

        cm_img_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        _save_confusion_matrix(np.array(metrics["confusion_matrix"]), cm_img_path)
        mlflow.log_artifact(cm_img_path)

        # ── Log classification report as text ──
        report_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(metrics["classification_report"])
        mlflow.log_artifact(report_path)

        # ── SHAP explainability ──
        shap_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
        if _save_shap_summary(model, X_test, shap_path):
            mlflow.log_artifact(shap_path)

        # ── Feature Importance ──
        try:
            from catboost import Pool
            train_pool = Pool(X_train, y_train, cat_features=categorical_features)
            feature_importance = model.get_feature_importance(train_pool)
            feature_names = X_train.columns
            
            logger.info("Feature importance len: %d, Feature names len: %d", len(feature_importance), len(feature_names))

            fi_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
            fi_df = fi_df.sort_values(by="importance", ascending=False)
            
            # Save as CSV
            fi_csv_path = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")
            fi_df.to_csv(fi_csv_path, index=False)
            mlflow.log_artifact(fi_csv_path)

            # Save as Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(fi_df["feature"][:20], fi_df["importance"][:20], color="skyblue")
            ax.set_xlabel("Importance")
            ax.set_title("Top 20 Feature Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            fi_plot_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
            plt.savefig(fi_plot_path)
            plt.close()
            mlflow.log_artifact(fi_plot_path)
            logger.info("Feature importance saved to %s", fi_csv_path)
        except Exception as e:
            logger.error("Failed to calculate feature importance: %s", e)

        # ── Register model ──
        mlflow.catboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # ── Export standalone model for Docker / Streamlit Cloud ──
        from src.config import MODEL_DIR, MODEL_FILE_PATH, MODEL_METADATA_PATH

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model(MODEL_FILE_PATH)
        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump({"threshold": metrics["threshold"]}, f)
        logger.info("Standalone model exported to %s", MODEL_FILE_PATH)

        logger.info(
            "Training complete — ROC-AUC: %.4f | Recall: %.4f | Threshold: %.4f",
            metrics["roc_auc"], metrics["recall"], metrics["threshold"],
        )

    return model, metrics


if __name__ == "__main__":
    train()
