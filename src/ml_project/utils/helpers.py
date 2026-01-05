"""Utility functions for persistence, logging, and metrics."""

import json
import logging
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from typing import Dict, Any


def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save dictionary to JSON file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved to {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(obj, file_path: str) -> None:
    """Save object to pickle file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
    print(f"Saved to {file_path}")


def load_pickle(file_path: str):
    """Load object from pickle file."""
    return joblib.load(file_path)


def save_stats(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame statistics to text file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")
        f.write("Data types:\n")
        f.write(str(df.dtypes))
        f.write("\n\nMissing values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nBasic statistics:\n")
        f.write(str(df.describe()))
    print(f"Stats saved to {file_path}")


def get_timestamp() -> str:
    """Get current timestamp string (YYYYMMDD_HHMMSS)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
