from .Base_DataLoader import BaseDataLoader
import pandas as pd
from typing import Tuple


class DataLoader(BaseDataLoader):
    """Concrete data loader."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data."""
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df

    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame.
            test_size: Fraction of data for test set.
        
        Returns:
            Tuple of (train_df, test_df).
        """
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)
        print(f"Train: {len(train)}, Test: {len(test)}")
        return train, test

    def split_train_val(self, df: pd.DataFrame, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets.
        
        Args:
            df: Input DataFrame.
            val_size: Fraction of data for validation set.
        
        Returns:
            Tuple of (train_df, val_df).
        """
        split_idx = int(len(df) * (1 - val_size))
        train = df.iloc[:split_idx].reset_index(drop=True)
        val = df.iloc[split_idx:].reset_index(drop=True)
        print(f"Train: {len(train)}, Validation: {len(val)}")
        return train, val
