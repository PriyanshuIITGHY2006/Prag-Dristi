"""
Training script for the Assam flood forecasting LSTM encoder-decoder.

Usage:
  python train.py                          # uses defaults in configs/
  python train.py train.epochs=50          # override any config value
  python train.py train.target_station=Guwahati

Outputs:
  checkpoints/best_model.pt   -- best val-loss weights
  checkpoints/scalers.pkl     -- fitted feature and target scalers
  checkpoints/results.json    -- test-set metrics
"""

import json
import logging
import random
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        preds = model(x, teacher_forcing_ratio=teacher_forcing_ratio, targets=y)
        loss = criterion(preds, y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    tgt_scaler,
    flood_threshold: float,
) -> dict:
    from src.evaluation.metrics import evaluate_all

    model.eval()
    total_loss = 0.0
    all_obs, all_sim = [], []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_raw = batch["y_raw"].numpy()

        preds = model(x)
        loss = criterion(preds, y)
        total_loss += loss.item() * x.size(0)

        # Inverse-transform predictions to m³/s
        preds_np = preds.cpu().numpy()  # (batch, dec_len)
        preds_inv = tgt_scaler.inverse_transform(preds_np.reshape(-1, 1)).reshape(preds_np.shape)
        preds_m3s = np.expm1(preds_inv)

        all_obs.append(y_raw)
        all_sim.append(preds_m3s)

    obs = np.concatenate(all_obs).ravel()
    sim = np.concatenate(all_sim).ravel()

    metrics = evaluate_all(obs, sim, flood_threshold)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Hydra changes cwd to outputs/ -- fix paths relative to project root
    project_root = Path(hydra.utils.get_original_cwd())

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------ #
    # Load and prepare data
    # ------------------------------------------------------------------ #
    import pandas as pd
    from src.features.engineer import (
        engineer_features, split_temporal, fit_scalers, get_feature_cols,
    )
    from src.data.dataset import build_dataloaders

    data_path = project_root / f"data/processed/merged_{cfg.target_station.lower()}.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}\n"
            "Run: python -m src.data.build_dataset"
        )

    df = pd.read_csv(data_path, index_col="date", parse_dates=True)
    log.info("Loaded %d rows from %s", len(df), data_path)

    df = engineer_features(df)
    train_df, val_df, test_df = split_temporal(df, cfg.train_frac, cfg.val_frac)

    ckpt_dir = project_root / cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = ckpt_dir / "scalers.pkl"

    feature_cols = get_feature_cols(df)
    feat_scaler, tgt_scaler = fit_scalers(train_df, feature_cols, scaler_path=scaler_path)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        feature_cols=feature_cols,
        feat_scaler=feat_scaler,
        tgt_scaler=tgt_scaler,
        encoder_len=cfg.get("encoder_len", 30),
        decoder_len=cfg.get("decoder_len", 7),
        batch_size=cfg.batch_size,
    )

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model_cfg_path = project_root / "configs" / "model.yaml"
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.load(model_cfg_path)

    from src.models.lstm_seq2seq import FloodForecastModel
    model = FloodForecastModel(
        input_size=len(feature_cols),
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        encoder_len=model_cfg.encoder_len,
        decoder_len=model_cfg.decoder_len,
        fc_hidden=model_cfg.fc_hidden,
        use_attention=model_cfg.attention,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    # Flood-weighted MSE: errors on high-discharge timesteps are penalised more.
    # Weight = 1 + flood_weight_multiplier * (y / flood_threshold).
    # Normal days get weight ~1, flood-peak days get weight up to ~5x.
    flood_threshold_normalised = float(cfg.flood_threshold)
    flood_weight_multiplier = 4.0  # tune this: higher = more focus on peaks

    def criterion(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets are log-normalised; reconstruct approximate raw scale for weighting
        targets_raw = torch.expm1(
            torch.tensor(
                tgt_scaler.inverse_transform(
                    targets.detach().cpu().numpy().reshape(-1, 1)
                ).reshape(targets.shape),
                device=targets.device,
            )
        )
        weights = 1.0 + flood_weight_multiplier * torch.clamp(
            targets_raw / flood_threshold_normalised, min=0.0, max=1.0
        )
        return (weights * (preds - targets) ** 2).mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = ckpt_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        # Teacher forcing decreases from 0.5 to 0 over first 30 epochs
        tf_ratio = max(0.0, 0.5 * (1.0 - epoch / 30))

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.gradient_clip, tf_ratio
        )
        val_metrics = evaluate(model, val_loader, criterion, device, tgt_scaler, cfg.flood_threshold)
        val_loss = val_metrics["loss"]

        scheduler.step(val_loss)

        if epoch % cfg.log_every_n_epochs == 0 or epoch == 1:
            log.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_NSE=%.3f | val_KGE=%.3f",
                epoch, train_loss, val_loss, val_metrics["NSE"], val_metrics["KGE"],
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    # ------------------------------------------------------------------ #
    # Test evaluation
    # ------------------------------------------------------------------ #
    log.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, tgt_scaler, cfg.flood_threshold)

    log.info("=== TEST RESULTS ===")
    for k, v in test_metrics.items():
        log.info("  %-15s %.4f", k, v)

    results_path = ckpt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({k: round(float(v), 4) for k, v in test_metrics.items()}, f, indent=2)
    log.info("Results saved -> %s", results_path)


if __name__ == "__main__":
    main()
