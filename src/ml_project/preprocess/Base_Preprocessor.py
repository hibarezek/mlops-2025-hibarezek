from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    """Abstract base class for preprocess step."""

    @abstractmethod
    def process(self, train: pd.DataFrame, test: pd.DataFrame):
        """Takes training and test DataFrames and returns transformed DataFrames."""
        raise NotImplementedError()
