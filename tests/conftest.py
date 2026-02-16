import pytest
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.train import train

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    df = load_data(RAW_DATA_PATH)
    return df.sample(n=50, random_state=42)

@pytest.fixture
def processed_data(sample_data):
    """Return processed data tuple."""
    return preprocess(sample_data)

@pytest.fixture
def model_and_metrics():
    """Train a small model for testing (fast)."""
    # Patch config to run fast
    from src import config
    original_params = config.CATBOOST_PARAMS.copy()
    config.CATBOOST_PARAMS["iterations"] = 1
    config.CATBOOST_PARAMS["early_stopping_rounds"] = None
    
    try:
        model, metrics = train()
        return model, metrics
    finally:
        # Restore original params
        config.CATBOOST_PARAMS = original_params
