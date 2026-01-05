from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseInference(ABC):
    """Base class for batch inference."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on input data.
        
        Args:
            data: Input dataset for prediction.
        
        Returns:
            DataFrame with predictions and metadata.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load trained model."""
        pass
