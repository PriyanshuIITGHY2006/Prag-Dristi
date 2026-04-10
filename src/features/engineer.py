"""
Feature engineering for the flood forecasting model.

Takes the merged daily DataFrame from build_dataset.py and produces
a feature matrix ready for the LSTM dataset.

Features created:
  - Lag features: discharge and rainfall at t-1 to t-7
  - Rolling rainfall accumulations: 3-day, 7-day, 14-day, 30-day sums
  - Rolling discharge statistics: 3-day and 7-day means
  - Wind speed magnitude from u/v components
  - Day of year (sin/cos encoding for seasonality)
  - Log-transformed discharge (target is log(Q), predicted log back)

All features are normalised with StandardScaler fitted on the training
split only (to avoid leakage).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


DISCHARGE_COL = "discharge_m3s"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features to the raw merged DataFrame."""
    df = df.copy()
    df.sort_index(inplace=True)

    # --- Wind speed magnitude ---
    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)
        df.drop(columns=["wind_u", "wind_v"], inplace=True)

    # --- Lag features ---
    for lag in [1, 2, 3, 5, 7]:
        df[f"discharge_lag{lag}"] = df[DISCHARGE_COL].shift(lag)
        if "precip_mm" in df.columns:
            df[f"precip_lag{lag}"] = df["precip_mm"].shift(lag)

    # --- Rolling rainfall accumulations ---
    if "precip_mm" in df.columns:
        for window in [3, 7, 14, 30]:
            df[f"precip_roll{window}d"] = (
                df["precip_mm"].rolling(window, min_periods=1).sum()
            )

    # --- Rolling discharge statistics ---
    for window in [3, 7]:
        df[f"discharge_roll{window}d_mean"] = (
            df[DISCHARGE_COL].rolling(window, min_periods=1).mean()
        )

    # --- Log discharge (stabilises variance) ---
    df["log_discharge"] = np.log1p(df[DISCHARGE_COL])

    # --- Seasonality: day-of-year encoded as sin/cos ---
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Drop rows with NaN introduced by lagging
    df.dropna(inplace=True)

    return df


def split_temporal(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal (chronological) train / val / test split.
    Never shuffles -- future data must never contaminate the past.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    log.info(
        "Split: train=%d  val=%d  test=%d  (total=%d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "log_discharge",
    scaler_path: str | Path | None = None,
) -> tuple[StandardScaler, StandardScaler]:
    """
    Fit StandardScaler on the training split for features and target separately.
    Optionally save to disk for later inference.

    Returns:
        (feature_scaler, target_scaler)
    """
    feat_scaler = StandardScaler()
    feat_scaler.fit(train_df[feature_cols].values)

    tgt_scaler = StandardScaler()
    tgt_scaler.fit(train_df[[target_col]].values)

    if scaler_path is not None:
        path = Path(scaler_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"feature": feat_scaler, "target": tgt_scaler}, path)
        log.info("Scalers saved -> %s", path)

    return feat_scaler, tgt_scaler


def load_scalers(scaler_path: str | Path) -> tuple[StandardScaler, StandardScaler]:
    saved = joblib.load(scaler_path)
    return saved["feature"], saved["target"]


def get_feature_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return list of feature column names (everything except raw discharge)."""
    if exclude is None:
        exclude = [DISCHARGE_COL, "log_discharge"]
    return [c for c in df.columns if c not in exclude]
