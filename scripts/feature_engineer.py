"""
Feature engineering script
- Reads preprocessed CSV
- Builds distance and datetime features
- Encodes categorical variables and scales numeric ones
- In `train` mode fits encoders/scalers and saves them to artifacts
- In `inference` mode loads encoders/scalers from artifacts
- Saves engineered CSV to outputs/engineered/
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------- Logging -------------------------

def setup_logging():
    logger = logging.getLogger("FeatureEngineer")
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


# ------------------------- Geospatial / Distance -------------------------

def haversine_vectorized(lat1, lon1, lat2, lon2, R=6371.0088):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def build_distance_feature(df: pd.DataFrame, chunk_size: int = 100000, logger=None):
    logger = logger or logging.getLogger("FeatureEngineer")
    logger.info("Calculating Haversine distance (km) ...")
    distances = np.zeros(len(df))
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        distances[start:end] = haversine_vectorized(
            df['pickup_latitude'].iloc[start:end].values,
            df['pickup_longitude'].iloc[start:end].values,
            df['dropoff_latitude'].iloc[start:end].values,
            df['dropoff_longitude'].iloc[start:end].values
        )
    df['distance_km'] = distances
    # fallback median
    df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())
    logger.info("Distance feature created")
    return df


# ------------------------- Datetime features -------------------------

def build_datetime_features(df: pd.DataFrame, logger=None):
    logger = logger or logging.getLogger("FeatureEngineer")
    logger.info("Building datetime features...")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_month'] = df['pickup_datetime'].dt.month
    # fill missing with median
    for col in ['pickup_hour','pickup_day','pickup_weekday','pickup_month']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    logger.info("Datetime features added")
    return df


# ------------------------- Encoding & Scaling -------------------------

def encode_categorical(df: pd.DataFrame, cat_cols, encoder: OneHotEncoder = None, fit=False, logger=None):
    logger = logger or logging.getLogger("FeatureEngineer")
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[cat_cols])
    else:
        if encoder is None:
            raise ValueError("Encoder must be provided for inference mode")
        encoded = encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    logger.info(f"Categorical encoded: {cat_cols}")
    return df, encoder


def scale_numeric(df: pd.DataFrame, num_cols, scaler: StandardScaler = None, fit=False, logger=None):
    logger = logger or logging.getLogger("FeatureEngineer")
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for inference mode")
        df[num_cols] = scaler.transform(df[num_cols])
    logger.info(f"Numeric columns scaled: {num_cols}")
    return df, scaler


# ------------------------- Pipeline -------------------------

def feature_engineer(
    df: pd.DataFrame,
    mode: str,
    config: dict,
    logger: logging.Logger
):
    # 1) Distance
    df = build_distance_feature(df, chunk_size=config['feature_engineering'].get('distance_calculation_chunk_size', 100000), logger=logger)

    # 2) Datetime
    df = build_datetime_features(df, logger=logger)

    # 3) Encode categorical
    cat_cols = config['feature_engineering'].get('categorical_columns', ['vendor_id','store_and_fwd_flag'])

    encoder = None
    scaler = None

    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    encoder_artifact = config['feature_engineering'].get('encoder_artifact', 'onehot_encoder.pkl')
    scaler_artifact = config['feature_engineering'].get('scaler_artifact', 'standard_scaler.pkl')

    if mode == 'train':
        df, encoder = encode_categorical(df, cat_cols, encoder=None, fit=True, logger=logger)
    else:
        # inference: load encoder
        enc_path = artifacts_dir / encoder_artifact
        if not enc_path.exists():
            raise FileNotFoundError(f"Encoder artifact not found: {enc_path}")
        encoder = joblib.load(enc_path)
        df, encoder = encode_categorical(df, cat_cols, encoder=encoder, fit=False, logger=logger)

    # 4) Scale numerics
    num_cols = config['feature_engineering'].get('numeric_columns', ['distance_km','pickup_hour','pickup_day','pickup_weekday','pickup_month','passenger_count'])

    if mode == 'train':
        df, scaler = scale_numeric(df, num_cols, scaler=None, fit=True, logger=logger)
    else:
        scl_path = artifacts_dir / scaler_artifact
        if not scl_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found: {scl_path}")
        scaler = joblib.load(scl_path)
        df, scaler = scale_numeric(df, num_cols, scaler=scaler, fit=False, logger=logger)

    # 5) Prepare X (drop leakage columns)
    X = df.drop(columns=['trip_duration', 'id', 'pickup_datetime', 'dropoff_datetime'], errors='ignore')
    y = df['trip_duration'] if 'trip_duration' in df.columns else None

    feature_cols = list(X.columns)

    return X, y, encoder, scaler, feature_cols, df


# ------------------------- Save helpers -------------------------

def save_artifacts(encoder, scaler, config, logger):
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    enc_name = config['feature_engineering'].get('encoder_artifact', 'onehot_encoder.pkl')
    scl_name = config['feature_engineering'].get('scaler_artifact', 'standard_scaler.pkl')
    enc_path = artifacts_dir / enc_name
    scl_path = artifacts_dir / scl_name
    if encoder is not None:
        joblib.dump(encoder, enc_path)
        logger.info(f"Saved encoder: {enc_path}")
    if scaler is not None:
        joblib.dump(scaler, scl_path)
        logger.info(f"Saved scaler: {scl_path}")
    return str(enc_path), str(scl_path)


def save_engineered_df(df: pd.DataFrame, output_path: str, logger: logging.Logger):
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    logger.info(f"Engineered dataset saved: {outp}")
    return str(outp)


def save_feature_list(feature_cols, output_dir: str, logger: logging.Logger):
    p = Path(output_dir) / f"feature_columns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        for c in feature_cols:
            f.write(f"{c}\n")
    logger.info(f"Saved feature list: {p}")
    return str(p)


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature engineering script")
    parser.add_argument('--input', required=True, help='Path to preprocessed CSV')
    parser.add_argument('--output', required=True, help='Path to save engineered CSV')
    parser.add_argument('--mode', choices=['train','inference'], default='train')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    logger.info(f"Loading preprocessed data from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} rows")

    X, y, encoder, scaler, feature_cols, full_df = feature_engineer(df, args.mode, config, logger)

    # Save engineered dataset (include original columns for traceability)
    save_engineered_df(full_df, args.output, logger)

    # Save artifacts if training
    if args.mode == 'train':
        enc_path, scl_path = save_artifacts(encoder, scaler, config, logger)
        save_feature_list(feature_cols, Path(config['paths']['engineered_dir']), logger)
    else:
        save_feature_list(feature_cols, Path(config['paths']['engineered_dir']), logger)

    logger.info("Feature engineering finished ✔")


if __name__ == '__main__':
    main()
