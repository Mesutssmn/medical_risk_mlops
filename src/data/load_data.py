# src/data/load_data.py

import logging
import pandas as pd
from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load raw CSV data and return a DataFrame."""
    path = path or RAW_DATA_PATH
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    # ── Drop duplicates to prevent leakage ──
    original_len = len(df)
    df = df.drop_duplicates()
    new_len = len(df)
    logger.info("Loaded %d rows, %d columns (Dropped %d duplicates)", new_len, df.shape[1], original_len - new_len)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_data()
    print(df.head())
    print(df.info())
