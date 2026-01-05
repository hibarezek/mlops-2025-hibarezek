"""
Preprocessing Script for NYC Taxi Trip Duration Dataset
Handles data cleaning, validation, and preparation for feature engineering
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import shutil

import pandas as pd
import numpy as np
import yaml


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging for the preprocessing pipeline."""
    logger = logging.getLogger("PreprocessingPipeline")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# CONFIG LOADING
# ============================================================================
def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def print_row_counts(
    df: pd.DataFrame,
    stage: str,
    removed: int,
    logger: logging.Logger
) -> None:
    """Print detailed row count information."""
    current_rows = len(df)
    logger.info(f"  Stage: {stage}")
    logger.info(f"    Rows removed: {removed}")
    logger.info(f"    Current rows: {current_rows}")


def create_output_dirs(config: Dict, logger: logging.Logger) -> None:
    """Create all required output directories."""
    output_dirs = [
        config["paths"]["raw_dir"],
        config["paths"]["preprocessed_dir"],
        config["paths"]["engineered_dir"],
        config["paths"]["predictions_dir"],
        config["paths"]["artifacts_dir"]
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directories created/verified ✔")


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list,
    mode: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Validate that all required columns are present."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns for {mode} mode: {missing_cols}")
    
    logger.info(f"✓ All required columns present ({len(required_cols)} columns)")
    return df


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================
def drop_missing_locations(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Remove rows with missing location data."""
    location_cols = [
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude'
    ]
    
    initial_rows = len(df)
    df = df.dropna(subset=location_cols)
    removed = initial_rows - len(df)
    
    if removed > 0:
        logger.info(f"✗ Dropped {removed} rows with missing location data")
    else:
        logger.info(f"✓ No missing location data")
    
    return df, removed


def fill_missing_passenger_count(
    df: pd.DataFrame,
    default_value: int,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Fill missing passenger count with default value."""
    if 'passenger_count' not in df.columns:
        logger.info("⊘ passenger_count column not found")
        return df, 0
    
    missing_count = df['passenger_count'].isna().sum()
    
    if missing_count > 0:
        df['passenger_count'] = df['passenger_count'].fillna(default_value)
        logger.info(f"✓ Filled {missing_count} missing passenger_count values with {default_value}")
    else:
        logger.info(f"✓ No missing passenger_count values")
    
    return df, 0  # Not removing rows, just filling


def remove_invalid_coordinates(
    df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Remove rows with invalid geographic coordinates."""
    lon_range = config["preprocessing"]["longitude_range"]
    lat_range = config["preprocessing"]["latitude_range"]
    
    initial_rows = len(df)
    
    df = df[
        (df['pickup_longitude'].between(lon_range[0], lon_range[1])) &
        (df['dropoff_longitude'].between(lon_range[0], lon_range[1])) &
        (df['pickup_latitude'].between(lat_range[0], lat_range[1])) &
        (df['dropoff_latitude'].between(lat_range[0], lat_range[1]))
    ]
    
    removed = initial_rows - len(df)
    
    if removed > 0:
        logger.info(f"✗ Removed {removed} rows with invalid coordinates")
        logger.info(f"  (Valid longitude range: {lon_range}, latitude range: {lat_range})")
    else:
        logger.info(f"✓ No invalid coordinates found")
    
    return df, removed


def remove_invalid_trip_durations(
    df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Remove rows with invalid trip durations (must be positive)."""
    if 'trip_duration' not in df.columns:
        logger.info("⊘ trip_duration column not found (inference mode?)")
        return df, 0
    
    initial_rows = len(df)
    min_duration = config["preprocessing"]["min_duration_seconds"]
    
    df = df[df['trip_duration'] > min_duration]
    
    removed = initial_rows - len(df)
    
    if removed > 0:
        logger.info(f"✗ Removed {removed} rows with invalid trip duration (<= {min_duration}s)")
    else:
        logger.info(f"✓ No invalid trip durations found")
    
    return df, removed


def remove_duration_outliers(
    df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Remove extreme outliers in trip duration using quantiles."""
    if 'trip_duration' not in df.columns:
        logger.info("⊘ trip_duration column not found (inference mode?)")
        return df, 0
    
    initial_rows = len(df)
    lower_q = config["preprocessing"]["duration_lower_quantile"]
    upper_q = config["preprocessing"]["duration_upper_quantile"]
    
    q_lower = df['trip_duration'].quantile(lower_q)
    q_upper = df['trip_duration'].quantile(upper_q)
    
    df = df[(df['trip_duration'] >= q_lower) & (df['trip_duration'] <= q_upper)]
    
    removed = initial_rows - len(df)
    
    if removed > 0:
        logger.info(f"✗ Removed {removed} outlier rows (duration outside {lower_q}-{upper_q} quantiles)")
        logger.info(f"  (Duration range: {q_lower:.0f}s - {q_upper:.0f}s retained)")
    else:
        logger.info(f"✓ No duration outliers detected")
    
    return df, removed


def remove_duplicates(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate records."""
    initial_rows = len(df)
    
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'])
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"✗ Removed {removed} duplicate records (by ID)")
        else:
            logger.info(f"✓ No duplicate records found")
    else:
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"✗ Removed {removed} duplicate records")
        else:
            logger.info(f"✓ No duplicate records found")
    
    return df, removed


# ============================================================================
# TIME FEATURE EXTRACTION
# ============================================================================
def parse_datetime(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """Parse datetime columns and remove rows with invalid dates."""
    initial_rows = len(df)
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    
    if 'dropoff_datetime' in df.columns:
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    
    df = df.dropna(subset=['pickup_datetime'])
    removed = initial_rows - len(df)
    
    if removed > 0:
        logger.info(f"✗ Removed {removed} rows with invalid pickup_datetime")
    else:
        logger.info(f"✓ All datetime values parsed successfully")
    
    return df, removed


def extract_time_features(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """Extract time-based features from pickup_datetime."""
    if 'pickup_datetime' not in df.columns:
        logger.warning("⚠ pickup_datetime column not found, skipping time features")
        return df
    
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_month'] = df['pickup_datetime'].dt.month
    
    logger.info(f"✓ Extracted 4 time-based features (hour, day, weekday, month)")
    
    return df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================
def preprocess(
    df: pd.DataFrame,
    mode: str,
    config: Dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the full preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        mode: 'train' or 'inference'
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Preprocessed DataFrame and statistics dictionary
    """
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE STARTED")
    logger.info("=" * 80)
    
    initial_rows = len(df)
    logger.info(f"\n📊 Initial dataset: {initial_rows:,} rows")
    
    # Get required columns based on mode
    required_cols = config["preprocessing"]["required_columns"][mode]
    
    # Step 1: Validate columns
    logger.info("\n[1/10] Validating required columns...")
    df = validate_required_columns(df, required_cols, mode, logger)
    
    # Step 2: Drop missing locations
    logger.info("\n[2/10] Removing rows with missing location data...")
    df, removed = drop_missing_locations(df, logger)
    print_row_counts(df, "Missing locations", removed, logger)
    
    # Step 3: Fill missing passenger count
    logger.info("\n[3/10] Handling missing passenger count...")
    df, _ = fill_missing_passenger_count(
        df,
        config["preprocessing"]["fill_passenger_count_default"],
        logger
    )
    
    # Step 4: Remove invalid coordinates
    logger.info("\n[4/10] Removing invalid geographic coordinates...")
    df, removed = remove_invalid_coordinates(df, config, logger)
    print_row_counts(df, "Invalid coordinates", removed, logger)
    
    # Step 5-6: Trip duration cleaning (only for training)
    if mode == "train":
        logger.info("\n[5/10] Removing invalid trip durations...")
        df, removed = remove_invalid_trip_durations(df, config, logger)
        print_row_counts(df, "Invalid durations", removed, logger)
        
        logger.info("\n[6/10] Removing duration outliers...")
        df, removed = remove_duration_outliers(df, config, logger)
        print_row_counts(df, "Duration outliers", removed, logger)
    else:
        logger.info("\n[5/10] Skipping duration validation (inference mode)")
        logger.info("\n[6/10] Skipping outlier removal (inference mode)")
    
    # Step 7: Remove duplicates
    logger.info("\n[7/10] Removing duplicate records...")
    df, removed = remove_duplicates(df, logger)
    print_row_counts(df, "Duplicates", removed, logger)
    
    # Step 8: Parse datetime
    logger.info("\n[8/10] Parsing datetime columns...")
    df, removed = parse_datetime(df, logger)
    print_row_counts(df, "Invalid datetimes", removed, logger)
    
    # Step 9: Extract time features
    logger.info("\n[9/10] Extracting time-based features...")
    df = extract_time_features(df, logger)
    
    # Step 10: Final cleanup
    logger.info("\n[10/10] Final cleanup and reindexing...")
    df = df.reset_index(drop=True)
    logger.info(f"✓ DataFrame reindexed")
    
    # Summary statistics
    final_rows = len(df)
    removed_total = initial_rows - final_rows
    retention_pct = (final_rows / initial_rows * 100) if initial_rows > 0 else 0
    
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n📊 Final dataset: {final_rows:,} rows")
    logger.info(f"   Total rows removed: {removed_total:,} ({100-retention_pct:.1f}%)")
    logger.info(f"   Data retention rate: {retention_pct:.1f}%")
    logger.info(f"\n📋 Available columns: {list(df.columns)}")
    
    stats = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "removed_rows": removed_total,
        "retention_pct": retention_pct,
        "mode": mode,
        "timestamp": datetime.now().isoformat()
    }
    
    return df, stats


# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================
def save_preprocessed_data(
    df: pd.DataFrame,
    output_path: str,
    logger: logging.Logger
) -> str:
    """Save preprocessed data to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Preprocessed data saved: {output_file}")
    
    return str(output_file)


def save_raw_copy(
    df: pd.DataFrame,
    input_path: str,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """Copy raw input file for traceability."""
    input_file = Path(input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"{input_file.stem}_raw_{timestamp}.csv"
    raw_path = Path(output_dir) / raw_filename
    
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_path, raw_path)
    
    logger.info(f"📁 Raw data copy saved: {raw_path}")
    
    return str(raw_path)


def save_stats(
    stats: Dict,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """Save preprocessing statistics to text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_filename = f"preprocessing_stats_{timestamp}.txt"
    stats_path = Path(output_dir) / stats_filename
    
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path, 'w') as f:
        f.write("PREPROCESSING STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {stats['timestamp']}\n")
        f.write(f"Mode: {stats['mode']}\n")
        f.write(f"Initial Rows: {stats['initial_rows']:,}\n")
        f.write(f"Final Rows: {stats['final_rows']:,}\n")
        f.write(f"Removed Rows: {stats['removed_rows']:,}\n")
        f.write(f"Retention Rate: {stats['retention_pct']:.1f}%\n")
    
    logger.info(f"📊 Statistics saved: {stats_path}")
    
    return str(stats_path)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Preprocessing script for NYC Taxi Trip Duration dataset"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where preprocessed CSV will be saved"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Preprocessing mode: 'train' (with outlier removal) or 'inference' (no target cleaning)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--save-raw",
        action="store_true",
        default=True,
        help="Save a copy of raw input data for traceability"
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    config = load_config(args.config)
    create_output_dirs(config, logger)
    
    try:
        # Load input data
        logger.info(f"📂 Loading input data from: {args.input}")
        input_df = pd.read_csv(args.input)
        logger.info(f"✓ Loaded {len(input_df):,} rows, {len(input_df.columns)} columns")
        
        # Save raw copy for traceability
        if args.save_raw:
            raw_copy_path = save_raw_copy(
                input_df,
                args.input,
                config["paths"]["raw_dir"],
                logger
            )
        
        # Execute preprocessing pipeline
        preprocessed_df, stats = preprocess(
            input_df.copy(),
            args.mode,
            config,
            logger
        )
        
        # Save preprocessed data
        output_path = save_preprocessed_data(
            preprocessed_df,
            args.output,
            logger
        )
        
        # Save statistics
        stats_path = save_stats(
            stats,
            config["paths"]["preprocessed_dir"],
            logger
        )
        
        logger.info("\n✅ Preprocessing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Preprocessing failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
