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

    # ── Feature Engineering ──
    # 1. Age Binning
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 19, 59, 120],
        labels=["Child", "Teen", "Adult", "Senior"]
    ).astype("object")

    # 2. BMI Binning
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    ).astype("object")

    # 3. Glucose Binning
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 99, 139, 300],
        labels=["Normal", "Prediabetic", "Diabetic"]
    ).astype("object")

    # 4. Interaction Terms
    df["age_x_bmi"] = df["age"] * df["bmi"]
    df["glucose_x_age"] = df["avg_glucose_level"] * df["age"]
    # Combine binary risk factors
    df["hypertension_x_heart"] = df["hypertension"] * df["heart_disease"]

    logger.info("Added features: age_group, bmi_category, glucose_category, interactions")

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
    # Re-scan for object/category columns including new features
    cat_features = [
        c for c in X.columns
        if X[c].dtype == "object" or X[c].dtype.name == "category"
    ]
    # Ensure they are strings (CatBoost requirement for categorical features)
    X_train[cat_features] = X_train[cat_features].astype(str)
    X_test[cat_features] = X_test[cat_features].astype(str)

    return X_train, X_test, y_train, y_test, cat_features
