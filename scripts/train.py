"""
Training script
- Reads engineered CSV (must include `trip_duration` as target)
- Splits into train/validation
- Trains multiple models from config
- Evaluates on validation set and selects best model based on chosen metric
- Logs runs to MLflow if enabled in config
- Saves best model to artifacts
"""

import argparse
import logging
import sys
from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import yaml
try:
    import mlflow
except Exception:
    mlflow = None
from datetime import datetime


# ------------------------- Logging -------------------------

def setup_logging():
    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger


# ------------------------- Config -------------------------

def load_config(path: str = "configs/config.yaml"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, 'r') as f:
        return yaml.safe_load(f)


# ------------------------- Metrics -------------------------

def evaluate(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred))
    }


# ------------------------- Training -------------------------

def train_and_select(X_train, y_train, X_valid, y_valid, cfg, logger):
    models_cfg = cfg.get('training', {}).get('model_configs', {})
    metric_name = cfg.get('training', {}).get('metric', 'mae')

    model_constructors = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'rf': RandomForestRegressor,
        'gb': GradientBoostingRegressor
    }

    models = {}
    for name, params in models_cfg.items():
        if name in model_constructors:
            try:
                models[name] = model_constructors[name](**params)
            except Exception as e:
                logger.warning(f"Could not instantiate model {name}: {e}")

    if not models:
        raise RuntimeError("No valid models to train. Check config.model_configs")

    best_model = None
    best_name = None
    best_score = float('inf') if metric_name in ['mae', 'rmse'] else -float('inf')

    mlflow_enabled = cfg.get('mlflow', {}).get('enabled', False)
    if mlflow_enabled:
        mlflow.set_tracking_uri(cfg.get('mlflow', {}).get('tracking_uri', None))
        mlflow.set_experiment(cfg.get('mlflow', {}).get('experiment_name', 'experiment'))

    results = {}

    for name, model in models.items():
        logger.info(f"Training model: {name}")
        if mlflow_enabled:
            run = mlflow.start_run(run_name=name)
            mlflow.log_params(models_cfg.get(name, {}))

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        metrics = evaluate(y_valid, preds)
        results[name] = metrics

        logger.info(f"Metrics for {name}: {metrics}")

        if mlflow_enabled:
            mlflow.log_metrics(metrics)
            mlflow.end_run()

        current = metrics.get(metric_name)
        if metric_name in ['mae', 'rmse']:
            is_better = current < best_score
        else:
            is_better = current > best_score

        if is_better:
            best_score = current
            best_model = model
            best_name = name

    return best_model, best_name, best_score, results


# ------------------------- MAIN -------------------------

def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--input', required=True, help='Path to engineered CSV (must include target `trip_duration`)')
    parser.add_argument('--output', required=True, help='Path to save best model (pkl)')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--test-size', type=float, default=None, help='Override test size from config')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config')

    args = parser.parse_args()
    logger = setup_logging()

    cfg = load_config(args.config)

    df = pd.read_csv(args.input)
    if 'trip_duration' not in df.columns:
        raise ValueError('Input engineered CSV must contain `trip_duration` target column')

    # Features: drop columns possibly containing raw datetimes or ids if present
    X = df.drop(columns=['trip_duration', 'id', 'pickup_datetime', 'dropoff_datetime'], errors='ignore')
    y = df['trip_duration']

    test_size = args.test_size if args.test_size is not None else cfg.get('training', {}).get('test_size', 0.2)
    seed = args.seed if args.seed is not None else cfg.get('training', {}).get('seed', 42)

    logger.info(f"Splitting data: test_size={test_size}, seed={seed}")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=seed)

    logger.info(f"Training set: {len(X_train):,} rows | Validation set: {len(X_valid):,} rows")

    best_model, best_name, best_score, results = train_and_select(X_train, y_train, X_valid, y_valid, cfg, logger)

    # Save best model
    artifacts_dir = Path(cfg.get('paths', {}).get('artifacts_dir', 'artifacts/'))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output)
    if out_path.is_dir():
        out_path = out_path / 'best_model.pkl'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, out_path)
    logger.info(f"Saved best model ({best_name}) to: {out_path}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = artifacts_dir / f'training_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump({'best_model': best_name, 'best_score': best_score, 'results': results}, f, indent=2)
    logger.info(f"Saved training results: {results_path}")

    # Also save model metadata
    meta_path = artifacts_dir / f'model_metadata_{timestamp}.json'
    meta = {
        'best_model': best_name,
        'best_score': best_score,
        'metric': cfg.get('training', {}).get('metric', 'mae'),
        'timestamp': timestamp,
        'model_path': str(out_path)
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved model metadata: {meta_path}")

    logger.info('Training complete ✔')


if __name__ == '__main__':
    main()
