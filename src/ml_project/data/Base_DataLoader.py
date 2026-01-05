from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple


class BaseDataLoader(ABC):
    """Base class for data loading and splitting."""

    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        pass

    @abstractmethod
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        pass

    @abstractmethod
    def split_train_val(self, df: pd.DataFrame, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets."""
        pass
