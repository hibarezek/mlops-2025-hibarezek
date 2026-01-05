from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureEngineer(ABC):
    @abstractmethod
    def fit_transform(self, train: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        raise NotImplementedError()
