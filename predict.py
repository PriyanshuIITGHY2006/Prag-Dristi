"""
Generate a 7-day discharge forecast for a monitoring station.

Loads the trained model and scalers, takes the most recent 30 days of
data from the processed CSV, and outputs a forecast.

Usage:
  python predict.py --station Bahadurabad
  python predict.py --station Guwahati --date 2022-06-15
"""

import argparse
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from src.features.engineer import engineer_features, load_scalers, get_feature_cols
from src.models.lstm_seq2seq import FloodForecastModel
from omegaconf import OmegaConf


def _remap_state_dict(sd: dict) -> dict:
    """
    Remap keys from the Kaggle notebook's flat model layout to the local
    nested module layout (LSTMEncoder / LSTMDecoder wrappers).

    Kaggle notebook:  encoder.*  decoder.*  attention.*  fc.*
    Local model:      encoder.lstm.*  decoder.lstm.*  decoder.attention.*  decoder.fc.*
    """
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder.") and not k.startswith("encoder.lstm."):
            new_sd["encoder.lstm." + k[len("encoder."):]] = v
        elif k.startswith("decoder.") and not k.startswith("decoder.lstm.") \
                and not k.startswith("decoder.attention.") \
                and not k.startswith("decoder.fc."):
            new_sd["decoder.lstm." + k[len("decoder."):]] = v
        elif k.startswith("attention."):
            new_sd["decoder." + k] = v
        elif k.startswith("fc."):
            new_sd["decoder." + k] = v
        else:
            new_sd[k] = v
    return new_sd


def load_model(ckpt_dir: Path, input_size: int, model_cfg) -> FloodForecastModel:
    model = FloodForecastModel(
        input_size=input_size,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        encoder_len=model_cfg.encoder_len,
        decoder_len=model_cfg.decoder_len,
        fc_hidden=model_cfg.fc_hidden,
        use_attention=model_cfg.attention,
    )
    weights_path = ckpt_dir / "best_model.pt"
    sd = torch.load(weights_path, map_location="cpu")
    # Handle checkpoints saved from the Kaggle notebook (flat key layout)
    if any(k.startswith("encoder.") and not k.startswith("encoder.lstm.") for k in sd):
        sd = _remap_state_dict(sd)
    model.load_state_dict(sd)
    model.eval()
    return model


def forecast(
    station: str,
    as_of_date: str | None = None,
    project_root: Path = Path("."),
) -> dict:
    """
    Generate a 7-day ahead discharge forecast.

    Args:
        station: Station name (must match configs/data.yaml)
        as_of_date: ISO date string. Defaults to last available date in data.
        project_root: Project root directory.

    Returns:
        dict with 'dates' and 'discharge_m3s' lists.
    """
    ckpt_dir = project_root / "checkpoints"
    data_path = project_root / f"data/processed/merged_{station.lower()}.csv"
    model_cfg_path = project_root / "configs" / "model.yaml"
    data_cfg_path = project_root / "configs" / "data.yaml"

    model_cfg = OmegaConf.load(model_cfg_path)
    data_cfg = OmegaConf.load(data_cfg_path)
    encoder_len = model_cfg.encoder_len
    decoder_len = model_cfg.decoder_len

    # Load and engineer features
    df = pd.read_csv(data_path, index_col="date", parse_dates=True)
    df = engineer_features(df)

    feat_scaler, tgt_scaler = load_scalers(ckpt_dir / "scalers.pkl")
    feature_cols = get_feature_cols(df)

    # Determine the "as of" date
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
    else:
        cutoff = df.index[-1]

    window = df[df.index <= cutoff].iloc[-encoder_len:]
    if len(window) < encoder_len:
        raise ValueError(f"Not enough history: need {encoder_len} days, got {len(window)}")

    # Scale features
    x = feat_scaler.transform(window[feature_cols].values).astype(np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, enc_len, features)

    # Load model
    model = load_model(ckpt_dir, len(feature_cols), model_cfg)

    with torch.no_grad():
        preds = model(x_tensor)  # (1, dec_len)

    preds_np = preds.numpy()
    preds_inv = tgt_scaler.inverse_transform(preds_np.reshape(-1, 1)).reshape(preds_np.shape)
    preds_m3s = np.expm1(preds_inv).squeeze().tolist()

    # Forecast dates
    start_date = cutoff + timedelta(days=1)
    forecast_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(decoder_len)]

    # Lookup danger level from config
    danger_discharge = None
    for s in data_cfg.stations:
        if s.name.lower() == station.lower():
            danger_discharge = s.danger_discharge
            break

    result = {
        "station": station,
        "as_of": cutoff.strftime("%Y-%m-%d"),
        "forecast_horizon_days": decoder_len,
        "dates": forecast_dates,
        "discharge_m3s": [round(q, 1) for q in preds_m3s],
        "danger_discharge_m3s": danger_discharge,
        "flood_alert": [q >= danger_discharge for q in preds_m3s] if danger_discharge else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate flood forecast")
    parser.add_argument("--station", default="Bahadurabad", help="Station name")
    parser.add_argument("--date", default=None, help="As-of date (YYYY-MM-DD)")
    args = parser.parse_args()

    result = forecast(args.station, args.date)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
