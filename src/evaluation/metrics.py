"""
Hydrological evaluation metrics for flood forecasting.

Standard metrics:
  NSE   -- Nash-Sutcliffe Efficiency (1.0 = perfect, 0 = as good as mean, <0 = worse)
  KGE   -- Kling-Gupta Efficiency (1.0 = perfect)
  RMSE  -- Root Mean Squared Error (m³/s)
  PBIAS -- Percent Bias (%)

Flood-specific (threshold-based, treat as binary classification):
  CSI   -- Critical Success Index (Threat Score) = TP / (TP + FP + FN)
  POD   -- Probability of Detection (Recall)     = TP / (TP + FN)
  FAR   -- False Alarm Ratio                     = FP / (TP + FP)
  HSS   -- Heidke Skill Score

All functions accept numpy arrays.
"""

import numpy as np


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    num = np.sum((obs - sim) ** 2)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - num / denom)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009)."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-9)
    beta = np.mean(sim) / (np.mean(obs) + 1e-9)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent bias (positive = model over-predicts)."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    total_obs = np.sum(obs)
    if total_obs == 0:
        return np.nan
    return float(100.0 * (np.sum(sim) - total_obs) / total_obs)


def flood_contingency(
    obs: np.ndarray, sim: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """
    Compute contingency table for flood events above a discharge threshold.

    Returns: (TP, FP, FN, TN)
    """
    obs_flood = obs >= threshold
    sim_flood = sim >= threshold
    TP = int(np.sum(obs_flood & sim_flood))
    FP = int(np.sum(~obs_flood & sim_flood))
    FN = int(np.sum(obs_flood & ~sim_flood))
    TN = int(np.sum(~obs_flood & ~sim_flood))
    return TP, FP, FN, TN


def csi(obs: np.ndarray, sim: np.ndarray, threshold: float) -> float:
    """Critical Success Index (Threat Score)."""
    TP, FP, FN, _ = flood_contingency(obs, sim, threshold)
    denom = TP + FP + FN
    return float(TP / denom) if denom > 0 else np.nan


def pod(obs: np.ndarray, sim: np.ndarray, threshold: float) -> float:
    """Probability of Detection."""
    TP, _, FN, _ = flood_contingency(obs, sim, threshold)
    denom = TP + FN
    return float(TP / denom) if denom > 0 else np.nan


def far(obs: np.ndarray, sim: np.ndarray, threshold: float) -> float:
    """False Alarm Ratio."""
    TP, FP, _, _ = flood_contingency(obs, sim, threshold)
    denom = TP + FP
    return float(FP / denom) if denom > 0 else np.nan


def hss(obs: np.ndarray, sim: np.ndarray, threshold: float) -> float:
    """Heidke Skill Score."""
    TP, FP, FN, TN = flood_contingency(obs, sim, threshold)
    n = TP + FP + FN + TN
    if n == 0:
        return np.nan
    expected = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / (n * n)
    accuracy = (TP + TN) / n
    denom = 1.0 - expected
    if denom == 0:
        return np.nan
    return float((accuracy - expected) / denom)


def evaluate_all(
    obs: np.ndarray,
    sim: np.ndarray,
    flood_threshold: float,
) -> dict[str, float]:
    """Return a dict of all metrics."""
    return {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "RMSE_m3s": rmse(obs, sim),
        "PBIAS_%": pbias(obs, sim),
        "CSI": csi(obs, sim, flood_threshold),
        "POD": pod(obs, sim, flood_threshold),
        "FAR": far(obs, sim, flood_threshold),
        "HSS": hss(obs, sim, flood_threshold),
    }
