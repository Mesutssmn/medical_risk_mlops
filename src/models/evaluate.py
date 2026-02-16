# src/models/evaluate.py

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true, y_proba, beta: float = 2.0) -> float:
    """
    Find the probability threshold that maximises F-beta score,
    favouring recall (beta > 1 → recall-oriented).

    Parameters
    ----------
    y_true  : array-like ground truth
    y_proba : array-like predicted probabilities for class 1
    beta    : float, default 2.0, higher values weight recall more

    Returns
    -------
    float – optimal threshold
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)

    # Remove last element (precision=1, recall=0, no threshold)
    precision_arr = precision_arr[:-1]
    recall_arr = recall_arr[:-1]

    # F-beta score for each threshold
    fbeta = ((1 + beta ** 2) * precision_arr * recall_arr) / (
        (beta ** 2) * precision_arr + recall_arr + 1e-8
    )

    best_idx = np.argmax(fbeta)
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold: %.4f  (F%.1f=%.4f, Precision=%.4f, Recall=%.4f)",
        best_threshold, beta, fbeta[best_idx],
        precision_arr[best_idx], recall_arr[best_idx],
    )
    return best_threshold


def evaluate_model(model, X_test, y_test, threshold: float | None = None) -> dict:
    """
    Evaluate a trained CatBoost model on the test set.

    Parameters
    ----------
    model     : CatBoostClassifier
    X_test    : features
    y_test    : labels
    threshold : custom probability threshold; if None, uses default 0.5

    Returns
    -------
    dict with keys: roc_auc, accuracy, precision, recall, f1,
                    confusion_matrix, classification_report, threshold
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Threshold tuning ──
    if threshold is None:
        threshold = find_optimal_threshold(y_test, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    report = classification_report(y_test, y_pred, zero_division=0)
    metrics["classification_report"] = report

    logger.info(
        "Threshold: %.4f | ROC-AUC: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["threshold"], metrics["roc_auc"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )
    logger.info("Classification Report:\n%s", report)

    return metrics
