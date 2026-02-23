import pytest
import pandas as pd
from src.config import TARGET_COLUMN

def test_data_schema(sample_data):
    """Check if required columns exist."""
    # Updated to match new schema and src/config.py
    required_cols = [
        "Age", "Gender", "SES", "Hypertension", "Heart_Disease", 
        "Avg_Glucose", "BMI", "Diabetes", "Smoking_Status", TARGET_COLUMN
    ]
    for col in required_cols:
        assert col in sample_data.columns, f"Missing column: {col}"

def test_no_null_target(sample_data):
    """Ensure target column has no missing values."""
    assert sample_data[TARGET_COLUMN].isnull().sum() == 0, "Target column contains NULL values"

def test_target_values(sample_data):
    """Ensure target contains only binary values (0, 1)."""
    unique_vals = sample_data[TARGET_COLUMN].unique()
    assert set(unique_vals).issubset({0, 1}), f"Unexpected target values: {unique_vals}"

def test_numeric_ranges(sample_data):
    """Check if numeric features are within reasonable ranges."""
    # Use new column names
    assert (sample_data["Age"] >= 0).all() and (sample_data["Age"] <= 120).all()
    assert (sample_data["Avg_Glucose"] > 0).all()
    
    # Check BMI if present (it's optional in some contexts but usually present in raw data)
    if "BMI" in sample_data.columns:
        valid_bmi = sample_data["BMI"].dropna()
        assert (valid_bmi > 0).all()
