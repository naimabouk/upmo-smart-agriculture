"""Data loading, KNN imputation and temporal feature engineering."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer

TARGET_ENERGY = "energy_index"
TARGET_YIELD = "yield_efficiency_index"
TARGETS = [TARGET_ENERGY, TARGET_YIELD]


def load_data(path: str) -> pd.DataFrame:
    """Load the sensor CSV and drop the timestamp column (rows stay in time order)."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    return df


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Fill missing sensor readings with KNN imputation."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


def add_temporal_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add lag-1 and rolling-mean features for every non-target column.

    Rows made incomplete by the lag/rolling window are dropped.
    """
    base_cols = [c for c in df.columns if c not in TARGETS]
    out = df.copy()
    for col in base_cols:
        out[f"{col}_lag1"] = out[col].shift(1)
    for col in base_cols:
        out[f"{col}_roll{window}"] = out[col].rolling(window).mean()
    return out.dropna().reset_index(drop=True)


def split_features_targets(df_fe: pd.DataFrame):
    """Return (X, y_energy, y_yield)."""
    X = df_fe.drop(columns=TARGETS)
    return X, df_fe[TARGET_ENERGY], df_fe[TARGET_YIELD]


def prepare(path: str):
    """Full preprocessing: load -> impute -> temporal features."""
    df = load_data(path)
    df = impute_knn(df)
    return add_temporal_features(df)
