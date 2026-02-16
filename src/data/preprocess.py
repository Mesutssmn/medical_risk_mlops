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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations to a DataFrame.
    Used for both training and inference.
    """
    df = df.copy()

    # ── Handle missing BMI with median (hardcoded to avoid training dependency issues in inference) ──
    # In production, this should ideally be loaded from a saved artifact/metadata.
    # For now, we use a sensible default or the same logic if we can't easily load training stats.
    # Training median was ~28.1. Let's use 28.1 if nan. 
    # Or just keep logic simple: if column exists.
    if "bmi" in df.columns:
         if df["bmi"].isnull().sum() > 0:
            df["bmi"] = df["bmi"].fillna(28.1) # Hardcoded median from training set to ensure consistency

    # ── Feature Engineering ──
    # 1. Age Binning
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 12, 19, 59, 120],
            labels=["Child", "Teen", "Adult", "Senior"]
        ).astype("object")

    # 2. BMI Binning
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        ).astype("object")

    # 3. Glucose Binning
    if "avg_glucose_level" in df.columns:
        df["glucose_category"] = pd.cut(
            df["avg_glucose_level"],
            bins=[0, 99, 139, 300],
            labels=["Normal", "Prediabetic", "Diabetic"]
        ).astype("object")

    # 4. Interaction Terms
    if "age" in df.columns and "bmi" in df.columns:
        df["age_x_bmi"] = df["age"] * df["bmi"]
    
    if "avg_glucose_level" in df.columns and "age" in df.columns:
        df["glucose_x_age"] = df["avg_glucose_level"] * df["age"]
    
    if "hypertension" in df.columns and "heart_disease" in df.columns:
        df["hypertension_x_heart"] = df["hypertension"] * df["heart_disease"]

    # ── Categorical Casting ──
    # Ensure all object columns are strings (CatBoost requirement)
    # We must do this for ALL potential categorical columns, 
    # including the new ones and original ones.
    # But during inference, we might only have "Male" etc.
    # We define the known categorical columns to be safe.
    
    known_cat_cols = [
        "gender", "ever_married", "work_type", "Residence_type", 
        "smoking_status", "age_group", "bmi_category", "glucose_category"
    ]
    
    for c in known_cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    return df


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

    # ── Apply Feature Engineering ──
    # Note: We do this BEFORE split to ensure consistent categories, 
    # but strictly we should fillna based on train split only for BMI.
    # However, create_features uses a fixed value now or logic that works row-wise.
    df = create_features(df)
    
    logger.info("Applied feature engineering via create_features")

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

    # ── Scaling (RobustScaler) ──
    from sklearn.preprocessing import RobustScaler
    from src.config import SCALING_FEATURES
    
    # Filter features that actually exist in the dataframe
    scale_cols = [c for c in SCALING_FEATURES if c in X_train.columns]
    
    if scale_cols:
        scaler = RobustScaler()
        # Fit works on Train only to prevent leakage
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = scaler.transform(X_test[scale_cols])
        logger.info("Scaled features: %s", scale_cols)
    else:
        scaler = None

    # ── Categorical feature indices for CatBoost ──
    # Re-scan for object/category columns including new features
    # (Already cast to str in create_features, but check again)
    cat_features = [
        c for c in X.columns
        if X[c].dtype == "object" or X[c].dtype.name == "category"
    ]
    
    return X_train, X_test, y_train, y_test, cat_features, scaler
