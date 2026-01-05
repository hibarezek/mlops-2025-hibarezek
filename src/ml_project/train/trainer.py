from .Base_Trainer import BaseTrainer
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Trainer(BaseTrainer):
    """Concrete trainer adapted from scripts/train.py"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.best_model = None
        self.best_model_name = None
        self.models = {}
        self.metrics = {}

    def _build_models(self):
        """Initialize models."""
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0, max_iter=5000),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        }

    def _evaluate(self, y_true, y_pred):
        """Compute evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def _select_features(self, df: pd.DataFrame):
        """Select feature columns (exclude id, datetime, target)."""
        exclude_cols = ['id', 'trip_duration', 'pickup_datetime', 'dropoff_datetime']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple models and select best by MAE."""
        self._build_models()

        feature_cols = self._select_features(train_data)
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['trip_duration']

        X_val = val_data[feature_cols].fillna(0)
        y_val = val_data['trip_duration']

        best_mae = float('inf')

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics = self._evaluate(y_val, y_pred)
            self.metrics[name] = metrics
            print(f"  MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")

            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                self.best_model = model
                self.best_model_name = name

        print(f"\nBest model: {self.best_model_name} (MAE: {best_mae:.2f})")

        return {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'metrics': self.metrics,
            'feature_columns': feature_cols,
        }

    def save_model(self, model, model_path: str) -> None:
        """Save model to disk."""
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load model from disk."""
        model = joblib.load(model_path)
        self.best_model = model
        return model

    def save_metadata(self, output_dir: str, best_model_name: str, metrics: Dict[str, Any]) -> None:
        """Save training metadata and results."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        training_results = {
            'timestamp': timestamp,
            'best_model': best_model_name,
            'metrics': {name: {k: float(v) for k, v in m.items()} for name, m in metrics.items()},
        }
        results_file = p / f"training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        print(f"Training results saved to {results_file}")

        model_metadata = {
            'timestamp': timestamp,
            'best_model': best_model_name,
            'best_metrics': {k: float(v) for k, v in metrics[best_model_name].items()},
        }
        metadata_file = p / f"model_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        print(f"Model metadata saved to {metadata_file}")
