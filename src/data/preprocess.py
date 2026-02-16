# src/data/preprocess.py

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    DROP_COLUMNS,
    CATEGORICAL_FEATURES,
)

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Clean the DataFrame and split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test, categorical_features
    """
    df = df.copy()

    # ── Drop ID column ──
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    # ── Handle missing BMI with median ──
    if "bmi" in df.columns and df["bmi"].isnull().sum() > 0:
        median_bmi = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(median_bmi)
        logger.info("Filled %d missing BMI values with median (%.1f)",
                     df["bmi"].isnull().sum(), median_bmi)

    # ── Split features / target ──
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # ── Train / test split (stratified) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # ── Categorical feature indices for CatBoost ──
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    return X_train, X_test, y_train, y_test, cat_features
