"""
Main pipeline orchestrator that combines all steps:
preprocessing → feature engineering → training → inference
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from src.ml_project.data.DataLoader import DataLoader
from src.ml_project.preprocess.preprocessor import Preprocessor
from src.ml_project.features.FeatureEngineer import FeatureEngineer
from src.ml_project.train.trainer import Trainer
from src.ml_project.inference.inference import Inference
from src.ml_project.utils.helpers import get_timestamp, save_json, save_stats


class MLPipeline:
    """
    Orchestrates the complete ML pipeline:
    Load → Preprocess → Feature Engineer → Train → Evaluate → Predict
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.trainer = Trainer(self.config)
        self.inference = Inference(self.config, self.preprocessor, self.feature_engineer)

        self._setup_directories()

    def _setup_directories(self):
        """Create output directories."""
        for key in ['artifacts_dir', 'outputs_dir', 'preprocessed_dir', 'engineered_dir', 'predictions_dir']:
            path = Path(self.config['paths'][key])
            path.mkdir(parents=True, exist_ok=True)

    def train_pipeline(
        self,
        train_csv: str,
        test_csv: str,
        val_size: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run training pipeline: load → preprocess → feature engineer → train → save artifacts.

        Args:
            train_csv: Path to raw training CSV.
            test_csv: Path to raw test CSV.
            val_size: Validation split size.
            seed: Random seed.

        Returns:
            Dictionary with results (model path, metrics, etc).
        """
        print("\n" + "="*70)
        print("TRAINING PIPELINE")
        print("="*70)

        # 1. Load raw data
        print("\n[1/6] Loading raw data...")
        train_raw = self.data_loader.load_data(train_csv)
        test_raw = self.data_loader.load_data(test_csv)
        print(f"  Train: {len(train_raw)} rows, Test: {len(test_raw)} rows")

        # 2. Preprocess in train mode
        print("\n[2/6] Preprocessing (train mode)...")
        train_proc = self.preprocessor.process_single(train_raw, is_train=True)
        test_proc = self.preprocessor.process_single(test_raw, is_train=False)
        print(f"  Train: {len(train_proc)} rows, Test: {len(test_proc)} rows")

        # Save preprocessed data
        preproc_dir = Path(self.config['paths']['preprocessed_dir'])
        preproc_dir.mkdir(parents=True, exist_ok=True)
        timestamp = get_timestamp()
        train_proc.to_csv(preproc_dir / f"train_preprocessed_{timestamp}.csv", index=False)
        test_proc.to_csv(preproc_dir / f"test_preprocessed_{timestamp}.csv", index=False)
        print(f"  Saved preprocessed data to {preproc_dir}")

        # Save preprocessing stats
        stats_file = preproc_dir / f"preprocessing_stats_{timestamp}.txt"
        save_stats(train_proc, str(stats_file))

        # 3. Feature engineer in train mode (fit + save encoders/scalers)
        print("\n[3/6] Feature engineering (train mode)...")
        train_eng, train_feature_cols = self.feature_engineer.fit_transform(train_proc)
        test_eng, _ = self.feature_engineer.transform(test_proc)
        print(f"  Train features: {len(train_feature_cols)} columns")

        # Save engineered data
        eng_dir = Path(self.config['paths']['engineered_dir'])
        eng_dir.mkdir(parents=True, exist_ok=True)
        train_eng.to_csv(eng_dir / f"train_engineered_{timestamp}.csv", index=False)
        test_eng.to_csv(eng_dir / f"test_engineered_{timestamp}.csv", index=False)
        print(f"  Saved engineered data to {eng_dir}")

        # Save feature columns list
        feature_cols_file = eng_dir / f"feature_columns_{timestamp}.txt"
        with open(feature_cols_file, 'w') as f:
            f.write('\n'.join(train_feature_cols))
        print(f"  Saved feature columns to {feature_cols_file}")

        # Save encoder and scaler artifacts
        artifacts_dir = self.config['paths']['artifacts_dir']
        self.feature_engineer.save_artifacts(artifacts_dir)
        print(f"  Saved encoder/scaler artifacts to {artifacts_dir}")

        # 4. Split training data for validation
        print("\n[4/6] Splitting training data (80/20 train/val)...")
        X_train = train_eng[train_feature_cols].fillna(0)
        y_train = train_eng['trip_duration']

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=val_size, random_state=seed
        )
        print(f"  Train: {len(X_train_split)}, Val: {len(X_val_split)}")

        # Combine for trainer
        train_data_for_trainer = X_train_split.copy()
        train_data_for_trainer['trip_duration'] = y_train_split
        val_data_for_trainer = X_val_split.copy()
        val_data_for_trainer['trip_duration'] = y_val_split

        # 5. Train models and select best
        print("\n[5/6] Training models...")
        results = self.trainer.train(train_data_for_trainer, val_data_for_trainer)

        # Save best model
        best_model = results['best_model']
        model_path = Path(self.config['paths']['artifacts_dir']) / "best_model.pkl"
        self.trainer.save_model(best_model, str(model_path))

        # Save training metadata
        print("\n[6/6] Saving artifacts...")
        self.trainer.save_metadata(self.config['paths']['artifacts_dir'], results['best_model_name'], results['metrics'])

        print("\n" + "="*70)
        print("✓ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"Best Model:  {results['best_model_name']}")
        print(f"Best MAE:    {results['metrics'][results['best_model_name']]['mae']:.2f}")
        print(f"Artifacts:   {self.config['paths']['artifacts_dir']}")
        print("="*70 + "\n")

        return {
            'model_path': str(model_path),
            'best_model_name': results['best_model_name'],
            'metrics': results['metrics'],
            'feature_columns': train_feature_cols,
        }

    def inference_pipeline(
        self,
        test_csv: str,
        model_path: str = None,
    ) -> Dict[str, Any]:
        """
        Run inference pipeline: load → preprocess → feature engineer → predict → save.

        Args:
            test_csv: Path to raw test CSV.
            model_path: Path to best model (default: artifacts/best_model.pkl).

        Returns:
            Dictionary with predictions path and stats.
        """
        if model_path is None:
            model_path = Path(self.config['paths']['artifacts_dir']) / "best_model.pkl"

        print("\n" + "="*70)
        print("INFERENCE PIPELINE")
        print("="*70)

        # 1. Load raw test data
        print("\n[1/3] Loading raw test data...")
        test_raw = self.data_loader.load_data(test_csv)
        print(f"  Loaded: {len(test_raw)} rows")

        # 2. Load model and artifacts
        print("\n[2/3] Loading model and artifacts...")
        self.inference.load_model(str(model_path))
        self.inference._load_artifacts(self.config['paths']['artifacts_dir'])

        # 3. Run full inference pipeline (preprocess -> feature engineer -> predict)
        print("\n[3/3] Running inference (preprocess → feature engineer → predict)...")
        predictions = self.inference.predict(test_raw)
        print(f"  Predictions: {len(predictions)} rows")

        # Save predictions
        pred_dir = Path(self.config['paths']['predictions_dir'])
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = self.inference.save_predictions(predictions, str(pred_dir))

        # Print summary stats
        print(f"\n  Prediction stats:")
        print(f"    Mean:  {predictions['prediction'].mean():.2f}")
        print(f"    Std:   {predictions['prediction'].std():.2f}")
        print(f"    Min:   {predictions['prediction'].min():.2f}")
        print(f"    Max:   {predictions['prediction'].max():.2f}")

        print("\n" + "="*70)
        print("✓ INFERENCE PIPELINE COMPLETE!")
        print("="*70)
        print(f"Predictions saved to: {pred_path}")
        print("="*70 + "\n")

        return {
            'predictions_path': pred_path,
            'num_predictions': len(predictions),
            'stats': {
                'mean': float(predictions['prediction'].mean()),
                'std': float(predictions['prediction'].std()),
                'min': float(predictions['prediction'].min()),
                'max': float(predictions['prediction'].max()),
            },
        }
