"""
PyTorch Dataset for the LSTM encoder-decoder flood forecasting model.

Produces sliding-window samples:
  X  : (encoder_len, n_features)  -- input sequence fed to the encoder
  y  : (decoder_len,)              -- target log-discharge values to predict
  y_raw : (decoder_len,)           -- un-normalised discharge (m³/s), for metrics

The dataset is constructed from an already-engineered, normalised DataFrame.
Call engineer_features() and fit_scalers() before constructing this dataset.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class FloodDataset(Dataset):
    """
    Sliding-window dataset.

    Args:
        df: Engineered DataFrame (all columns including target).
        feature_cols: List of input feature column names.
        target_col: Column name for the (log-transformed) target.
        raw_target_col: Column name for the un-normalised target (for metrics).
        feat_scaler: Fitted StandardScaler for features.
        tgt_scaler: Fitted StandardScaler for the target.
        encoder_len: Number of timesteps in the encoder input.
        decoder_len: Number of timesteps to forecast.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        raw_target_col: str,
        feat_scaler: StandardScaler,
        tgt_scaler: StandardScaler,
        encoder_len: int = 30,
        decoder_len: int = 7,
    ):
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.raw_target_col = raw_target_col

        # Scale features and target
        features = feat_scaler.transform(df[feature_cols].values).astype(np.float32)
        targets = tgt_scaler.transform(df[[target_col]].values).astype(np.float32).squeeze()
        raw_targets = df[raw_target_col].values.astype(np.float32)

        self.features = features      # (T, F)
        self.targets = targets        # (T,)
        self.raw_targets = raw_targets  # (T,)

        # Valid start indices: we need encoder_len past + decoder_len future
        self.n_samples = len(df) - encoder_len - decoder_len + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc_start = idx
        enc_end = idx + self.encoder_len
        dec_end = enc_end + self.decoder_len

        X = torch.from_numpy(self.features[enc_start:enc_end])          # (enc_len, F)
        y = torch.from_numpy(self.targets[enc_end:dec_end])             # (dec_len,)
        y_raw = torch.from_numpy(self.raw_targets[enc_end:dec_end])     # (dec_len,)

        return {"x": X, "y": y, "y_raw": y_raw}


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    feat_scaler: StandardScaler,
    tgt_scaler: StandardScaler,
    encoder_len: int = 30,
    decoder_len: int = 7,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple:
    """Build train, val, and test DataLoaders."""
    from torch.utils.data import DataLoader

    common = dict(
        feature_cols=feature_cols,
        target_col="log_discharge",
        raw_target_col="discharge_m3s",
        feat_scaler=feat_scaler,
        tgt_scaler=tgt_scaler,
        encoder_len=encoder_len,
        decoder_len=decoder_len,
    )

    train_ds = FloodDataset(train_df, **common)
    val_ds = FloodDataset(val_df, **common)
    test_ds = FloodDataset(test_df, **common)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
