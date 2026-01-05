import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from pathlib import Path
from .Base_FeatureEngineer import BaseFeatureEngineer


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ohe = None
        self.scaler = None

    def haversine_vectorized(self, lat1, lon1, lat2, lon2, R=6371.0088):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def build_distance(self, df: pd.DataFrame):
        distances = self.haversine_vectorized(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values,
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values
        )
        df['distance_km'] = distances
        df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())
        return df

    def build_datetime(self, df: pd.DataFrame):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
        df['pickup_month'] = df['pickup_datetime'].dt.month
        for col in ['pickup_hour','pickup_day','pickup_weekday','pickup_month']:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def fit_transform(self, train: pd.DataFrame):
        df = train.copy()
        df = self.build_distance(df)
        df = self.build_datetime(df)

        cat_cols = self.config.get('feature_engineering', {}).get('categorical_columns', ['vendor_id','store_and_fwd_flag'])
        df[cat_cols] = df[cat_cols].fillna('Unknown')
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = self.ohe.fit_transform(df[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=self.ohe.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, enc_df], axis=1)

        num_cols = self.config.get('feature_engineering', {}).get('numeric_columns', ['distance_km','pickup_hour','pickup_day','pickup_weekday','pickup_month','passenger_count'])
        self.scaler = StandardScaler()
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        feature_cols = list(df.drop(columns=['trip_duration','id','pickup_datetime','dropoff_datetime'], errors='ignore').columns)
        return df, feature_cols

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self.build_distance(df)
        df = self.build_datetime(df)

        cat_cols = self.config.get('feature_engineering', {}).get('categorical_columns', ['vendor_id','store_and_fwd_flag'])
        df[cat_cols] = df[cat_cols].fillna('Unknown')
        if self.ohe is None:
            raise RuntimeError('Encoder not fitted')
        encoded = self.ohe.transform(df[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=self.ohe.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, enc_df], axis=1)

        num_cols = self.config.get('feature_engineering', {}).get('numeric_columns', ['distance_km','pickup_hour','pickup_day','pickup_weekday','pickup_month','passenger_count'])
        if self.scaler is None:
            raise RuntimeError('Scaler not fitted')
        df[num_cols] = self.scaler.transform(df[num_cols])

        feature_cols = list(df.drop(columns=['trip_duration','id','pickup_datetime','dropoff_datetime'], errors='ignore').columns)
        return df, feature_cols

    def save_artifacts(self, artifacts_dir: str):
        p = Path(artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        if self.ohe is not None:
            joblib.dump(self.ohe, p / self.config.get('feature_engineering', {}).get('encoder_artifact', 'onehot_encoder.pkl'))
        if self.scaler is not None:
            joblib.dump(self.scaler, p / self.config.get('feature_engineering', {}).get('scaler_artifact', 'standard_scaler.pkl'))

    def load_artifacts(self, artifacts_dir: str):
        p = Path(artifacts_dir)
        enc_path = p / self.config.get('feature_engineering', {}).get('encoder_artifact', 'onehot_encoder.pkl')
        scl_path = p / self.config.get('feature_engineering', {}).get('scaler_artifact', 'standard_scaler.pkl')
        if enc_path.exists():
            self.ohe = joblib.load(enc_path)
        if scl_path.exists():
            self.scaler = joblib.load(scl_path)
