# src/data/validate.py

import logging
import pandas as pd
from src.config import TARGET_COLUMN, CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validate the raw DataFrame and return a diagnostics dict."""

    diagnostics: dict = {}

    # ── Target column exists ──
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    # ── Missing values ──
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning("Missing values detected (%d total)", total_nulls)
        diagnostics["missing"] = null_counts[null_counts > 0].to_dict()
    else:
        logger.info("No missing values")
        diagnostics["missing"] = {}

    # ── Target distribution ──
    target_dist = df[TARGET_COLUMN].value_counts().to_dict()
    diagnostics["target_distribution"] = target_dist
    logger.info("Target distribution: %s", target_dist)

    # ── Feature dtype check ──
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].dtype not in ("object", "category"):
            logger.warning("Expected categorical dtype for '%s', got %s", col, df[col].dtype)

    for col in NUMERIC_FEATURES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning("Expected numeric dtype for '%s', got %s", col, df[col].dtype)

    diagnostics["shape"] = df.shape
    return diagnostics
