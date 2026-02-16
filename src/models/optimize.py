import logging
import optuna
import mlflow
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, fbeta_score
from src.config import RANDOM_STATE, CATBOOST_PARAMS
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

def objective(trial):
    # ── Load and split data ──
    df = load_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test, cat_features = preprocess(df)

    # ── Suggest hyperparameters ──
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 1, 255),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        # Optimization for recall on minority class
        "class_weights": [1, trial.suggest_float("class_weight_1", 10, 50)],
    }

    # ── Train model ──
    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        verbose=False
    )

    # ── Evaluate (Maximize F2-Score for Recall focus) ──
    y_pred = model.predict(X_test)
    
    # F2 score weights recall higher than precision
    score = fbeta_score(y_test, y_pred, beta=2)
    
    return score

def run_optimization(n_trials=20):
    logger.info("Starting Optuna optimization with %d trials...", n_trials)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_optimization()
