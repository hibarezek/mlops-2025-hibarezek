"""
Batch Inference Script
- Loads raw test CSV
- Runs preprocessing (inference mode - no target/outlier cleaning)
- Runs feature engineering (inference mode - loads fitted encoders/scalers)
- Loads best model
- Makes predictions on test data
- Saves predictions to outputs/predictions/YYYYMMDD_predictions.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import tempfile

import joblib
import pandas as pd
import yaml


# Import preprocessing and feature engineering modules
sys.path.insert(0, str(Path(__file__).parent))
import preprocess as preprocess_module
import feature_engineer as fe_module


# ========================= Logging =========================

def setup_logging():
    logger = logging.getLogger("BatchInference")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger


# ========================= Config =========================

def load_config(path: str = "configs/config.yaml"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, 'r') as f:
        return yaml.safe_load(f)


# ========================= Main Inference =========================

def batch_inference(test_csv_path, model_path, output_path, config, logger):
    """
    Run full inference pipeline:
    1. Load raw test CSV
    2. Preprocess (inference mode)
    3. Feature engineer (inference mode)
    4. Load best model
    5. Make predictions
    6. Save predictions
    """
    
    # --- Step 1: Load raw test data ---
    logger.info(f"Loading raw test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(test_df):,} rows")
    
    # --- Step 2: Preprocess (inference mode) ---
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING (Inference Mode)")
    logger.info("="*80)
    test_df_preprocessed, preprocess_stats = preprocess_module.preprocess(
        test_df.copy(),
        mode='inference',
        config=config,
        logger=logger
    )
    logger.info(f"Preprocessed data: {len(test_df_preprocessed):,} rows")
    
    # --- Step 3: Feature Engineering (inference mode) ---
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING (Inference Mode)")
    logger.info("="*80)
    X, y, encoder, scaler, feature_cols, full_df = fe_module.feature_engineer(
        test_df_preprocessed,
        mode='inference',
        config=config,
        logger=logger
    )
    logger.info(f"Engineered features: {len(feature_cols)} columns")
    
    # --- Step 4: Load model ---
    logger.info(f"\nLoading best model from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully ✔")
    
    # --- Step 5: Make predictions ---
    logger.info(f"\nMaking predictions on {len(X):,} records...")
    predictions = model.predict(X)
    logger.info(f"Predictions complete ✔")
    
    # --- Step 6: Prepare output ---
    output_df = full_df[['id']].copy() if 'id' in full_df.columns else pd.DataFrame(index=full_df.index)
    output_df['prediction'] = predictions
    output_df['timestamp'] = datetime.now().isoformat()
    
    # --- Step 7: Save predictions ---
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Predictions saved to: {output_file}")
    logger.info(f"   Total predictions: {len(output_df):,}")
    logger.info(f"   Columns: {list(output_df.columns)}")
    
    return output_df


# ========================= CLI =========================

def main():
    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument('--input', required=True, help='Path to raw test CSV (e.g., src/ml_project/data/test.csv)')
    parser.add_argument('--model', required=True, help='Path to best model (pkl)')
    parser.add_argument('--output', required=True, help='Path to save predictions CSV')
    parser.add_argument('--config', default='configs/config.yaml')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    try:
        cfg = load_config(args.config)
        
        output_df = batch_inference(
            args.input,
            args.model,
            args.output,
            cfg,
            logger
        )
        
        logger.info("\n✅ Batch inference completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Inference failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
