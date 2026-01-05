from .Base_Inference import BaseInference
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.ml_project.preprocess.Preprocessor import Preprocessor
from src.ml_project.features.FeatureEngineer import FeatureEngineer


class Inference(BaseInference):
    """Concrete inference adapted from scripts/inference.py"""

    def __init__(self, config: dict = None, preprocessor: Preprocessor = None, feature_engineer: FeatureEngineer = None):
        self.config = config or {}
        self.preprocessor = preprocessor or Preprocessor(config)
        self.feature_engineer = feature_engineer or FeatureEngineer(config)
        self.model = None

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

    def _load_artifacts(self, artifacts_dir: str) -> None:
        """Load preprocessor and feature engineer artifacts."""
        self.feature_engineer.load_artifacts(artifacts_dir)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run full inference pipeline: preprocess -> feature engineer -> predict.
        
        Args:
            data: Raw input dataset.
        
        Returns:
            DataFrame with id, prediction, and timestamp.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess (inference mode)
        processed = self.preprocessor.process_single(data, is_train=False)

        # Feature engineer (inference mode, uses loaded artifacts)
        engineered, feature_cols = self.feature_engineer.transform(processed)

        # Predict
        X = engineered[feature_cols].fillna(0)
        predictions = self.model.predict(X)

        # Build output
        result = pd.DataFrame({
            'id': processed['id'],
            'prediction': predictions,
            'timestamp': datetime.now().isoformat(),
        })

        return result

    def save_predictions(self, result: pd.DataFrame, output_dir: str) -> str:
        """Save predictions to CSV."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = p / f"{timestamp}_predictions.csv"
        result.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")
        return str(filename)
