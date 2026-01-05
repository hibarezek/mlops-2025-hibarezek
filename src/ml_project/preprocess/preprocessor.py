from .Base_Preprocessor import BasePreprocessor
import pandas as pd
import numpy as np


class Preprocessor(BasePreprocessor):
    """Concrete preprocessor adapted from scripts/preprocess.py"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def validate_required_columns(self, df: pd.DataFrame, is_train: bool = True):
        required = [
            'pickup_latitude','pickup_longitude',
            'dropoff_latitude','dropoff_longitude','pickup_datetime'
        ]
        if is_train:
            required.append('trip_duration')
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def process_single(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        # basic cleaning steps
        df = df.copy()
        df = self.validate_required_columns(df, is_train=is_train)

        # drop missing locations
        df = df.dropna(subset=['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'])

        # fill passenger_count
        if 'passenger_count' in df.columns:
            default = self.config.get('preprocessing', {}).get('fill_passenger_count_default', 1)
            df['passenger_count'] = df['passenger_count'].fillna(default)

        # remove invalid coordinates
        lon_min, lon_max = self.config.get('preprocessing', {}).get('longitude_range', [-180,180])
        lat_min, lat_max = self.config.get('preprocessing', {}).get('latitude_range', [-90,90])
        df = df[
            df['pickup_longitude'].between(lon_min, lon_max) &
            df['dropoff_longitude'].between(lon_min, lon_max) &
            df['pickup_latitude'].between(lat_min, lat_max) &
            df['dropoff_latitude'].between(lat_min, lat_max)
        ]

        # trip duration cleaning for training
        if is_train and 'trip_duration' in df.columns:
            df = df[df['trip_duration'] > self.config.get('preprocessing', {}).get('min_duration_seconds', 0)]
            lower = self.config.get('preprocessing', {}).get('duration_lower_quantile', 0.01)
            upper = self.config.get('preprocessing', {}).get('duration_upper_quantile', 0.99)
            ql = df['trip_duration'].quantile(lower)
            qu = df['trip_duration'].quantile(upper)
            df = df[(df['trip_duration'] >= ql) & (df['trip_duration'] <= qu)]

        # remove duplicates
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        else:
            df = df.drop_duplicates()

        # parse datetime
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        if 'dropoff_datetime' in df.columns:
            df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
        df = df.dropna(subset=['pickup_datetime'])

        # extract time features
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
        df['pickup_month'] = df['pickup_datetime'].dt.month

        df = df.reset_index(drop=True)
        return df

    def process(self, train: pd.DataFrame, test: pd.DataFrame):
        train_clean = self.process_single(train, is_train=True)
        test_clean = self.process_single(test, is_train=False)
        return train_clean, test_clean
