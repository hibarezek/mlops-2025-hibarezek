from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple


class BaseTrainer(ABC):
    """Base class for model training."""

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model(s) on training data and evaluate on validation data.
        
        Args:
            train_data: Training dataset with features and target.
            val_data: Validation dataset with features and target.
        
        Returns:
            Dictionary with training results, best model, and metrics.
        """
        pass

    @abstractmethod
    def save_model(self, model, model_path: str) -> None:
        """Save trained model to disk."""
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        """Load trained model from disk."""
        pass
