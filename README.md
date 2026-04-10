# Prag-Dristi — Assam Flood Forecasting Engine

> **"Prag-Dristi"** (প্রাগ্‌-দৃষ্টি) — Sanskrit for *foresight*. A deep learning system for 7-day ahead river discharge forecasting across the Brahmaputra basin, Assam, India.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![NSE](https://img.shields.io/badge/NSE-0.924-brightgreen)](checkpoints/results.json)
[![KGE](https://img.shields.io/badge/KGE-0.920-brightgreen)](checkpoints/results.json)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

Prag-Dristi is a research-grade flood forecasting engine built for the Brahmaputra river system in Assam — one of the most flood-prone regions on Earth, where annual floods displace millions and cause billions in losses. The system generates 7-day probabilistic discharge forecasts for three key monitoring stations using an **LSTM Encoder-Decoder with Bahdanau Attention**, trained on 23 years of reanalysis data.

### Key Results (Test Set — Unseen Years)

| Metric | Score | Benchmark |
|--------|-------|-----------|
| NSE (Nash-Sutcliffe Efficiency) | **0.924** | >0.90 = Excellent |
| KGE (Kling-Gupta Efficiency) | **0.920** | >0.90 = Excellent |
| RMSE | 5,175 m³/s | — |
| CSI (Critical Success Index) | **0.623** | Flood event detection |
| POD (Probability of Detection) | **0.651** | Catches 65% of floods |
| FAR (False Alarm Ratio) | **0.065** | Only 6.5% false alarms |

These results are within the published state-of-the-art range for LSTM-based flood forecasting (Kratzert et al., 2018; Nevo et al., 2022).

---

## Architecture

### LSTM Encoder-Decoder with Bahdanau Attention

```
Input Sequence (30 days × N features)
        │
        ▼
┌─────────────────────────────┐
│        LSTM Encoder         │
│   2 layers · hidden=128     │
│                             │
│  x₁ → h₁                   │
│  x₂ → h₂                   │
│   ⋮      ⋮   (enc outputs)  │
│  x₃₀→ h₃₀                  │
└──────────┬──────────────────┘
           │  encoder outputs + final (h_n, c_n)
           ▼
┌─────────────────────────────────────────────────┐
│              Bahdanau Attention                  │
│                                                  │
│  e_{t,s} = vᵀ tanh(W₁ hₛᵉⁿᶜ + W₂ hₜᵈᵉᶜ)       │
│  α_{t,s} = softmax(e_{t,s})                     │
│  context_t = Σₛ α_{t,s} · hₛᵉⁿᶜ                │
└──────────────────┬──────────────────────────────┘
                   │ context vector
                   ▼
┌─────────────────────────────────────────────────┐
│           LSTM Decoder (autoregressive)          │
│   2 layers · hidden=128                         │
│                                                  │
│  Step t: input = [prev_pred ‖ context_t]        │
│          output → FC head → Q̂_t                │
│          Q̂_t becomes input for step t+1        │
│                                                  │
│  Q̂₁, Q̂₂, ..., Q̂₇  (7-day forecast)           │
└─────────────────────────────────────────────────┘
           │
           ▼
    FC Output Head
    Linear(128→64) → ReLU → Dropout → Linear(64→1)
           │
           ▼
    log1p(Q̂) → inverse_transform → Q̂ [m³/s]
```

### Why this architecture?

1. **LSTM over simple RNN** — LSTMs mitigate the vanishing gradient problem critical for hydrological systems with long memory (soil saturation, snowmelt) spanning weeks to months (Hochreiter & Schmidhuber, 1997).

2. **Encoder-Decoder over many-to-one** — Multi-step forecasting requires propagating uncertainty through time. The seq2seq structure allows independent error at each forecast horizon while sharing a learned context representation (Sutskever et al., 2014).

3. **Bahdanau Attention** — The Brahmaputra's peak discharge lags upstream rainfall by 2–5 days depending on catchment location. Attention lets the decoder dynamically focus on the most relevant encoder timesteps rather than compressing 30 days into a single fixed vector (Bahdanau et al., 2015).

4. **Teacher Forcing (annealed)** — Ratio decays from 0.5 → 0 over 30 epochs, preventing exposure bias while training stably (Williams & Zipser, 1989).

5. **Flood-weighted MSE loss** — Floods occur on only ~3–8% of days. Standard MSE produces high accuracy (>99%) by ignoring all peaks. Our loss:
   ```
   w_i = 1 + 4 · clip(Q_i / Q_danger, 0, 1)
   L = mean(w_i · (Q̂_i − Q_i)²)
   ```
   gives up to 5× more penalty on flood-peak days.

---

## Model Configuration

```yaml
encoder_len:  30     # days of history
decoder_len:   7     # days ahead
hidden_size:  128    # LSTM hidden units
num_layers:     2    # stacked LSTM layers
dropout:      0.3
attention:    true   # Bahdanau attention
fc_hidden:     64    # FC output head
```

Total trainable parameters: ~660,000

---

## Data Pipeline

### Sources

| Source | Variable | Resolution | Years |
|--------|----------|-----------|-------|
| ERA5 (Copernicus CDS) | Precipitation, Temperature, Pressure, Soil moisture, Wind (u, v) | 0.25° × 6-hourly | 2000–2022 |
| GloFAS-ERA5 v4.0 (Copernicus EWDS) | River discharge (dis24) | 0.05° × daily | 2000–2022 |

### Stations

| Station | River | Lat | Lon | Danger Level |
|---------|-------|-----|-----|-------------|
| Bahadurabad | Brahmaputra | 25.17°N | 89.67°E | 98,600 m³/s |
| Guwahati | Brahmaputra | 26.18°N | 91.73°E | 72,000 m³/s |
| Dibrugarh | Brahmaputra | 27.48°N | 95.00°E | 45,000 m³/s |

### Feature Engineering

From the raw ERA5 + GloFAS merge, the following features are constructed:

**Meteorological (from ERA5):**
- `precip_mm` — daily precipitation (m→mm converted)
- `temp_c` — 2m temperature (K→°C)
- `pressure_pa` — surface pressure
- `soil_moisture` — volumetric soil water layer 1
- `wind_u`, `wind_v` — 10m wind components

**Lag features** (discharge autoregression):
- `discharge_lag_{1,2,3,5,7}` — Q at t-1 through t-7

**Rolling statistics:**
- `precip_roll_{3,7,14,30}d` — antecedent rainfall accumulations
- `discharge_roll_{7,14}d_mean` — medium-term discharge trend

**Target variable:**
- `log_discharge = log1p(discharge_m3s)` — log-transform stabilises the heavily right-skewed distribution (σ/μ ≈ 0.9 in raw space)

**Seasonality encoding:**
- `doy_sin = sin(2π · doy / 365)`
- `doy_cos = cos(2π · doy / 365)`

All features are StandardScaler-normalised (fit on training set only — no data leakage).

### Train / Val / Test Split

Temporal (chronological) split — never shuffled:

```
2000 ──────────────────────────── 2022
│         Train (70%)          │Val│Test│
                                 15%  15%
```

- Train: 2000–2015 (~5,880 days)
- Val:   2016–2018 (~1,260 days)
- Test:  2019–2022 (~1,260 days)

---

## Project Structure

```
prag-dristi/
├── configs/
│   ├── data.yaml          # ERA5/GloFAS params, station metadata
│   ├── model.yaml         # LSTM architecture hyperparameters
│   └── train.yaml         # Training hyperparameters
│
├── src/
│   ├── data/
│   │   ├── era5_download.py     # Downloads ERA5 via Copernicus CDS API
│   │   ├── glofas_download.py   # Downloads GloFAS via Copernicus EWDS API
│   │   ├── build_dataset.py     # Merges ERA5 + GloFAS → processed CSVs
│   │   └── dataset.py           # PyTorch Dataset + DataLoader builder
│   │
│   ├── features/
│   │   └── engineer.py          # Lag/rolling features, scalers, splits
│   │
│   ├── models/
│   │   └── lstm_seq2seq.py      # BahdanauAttention, LSTMEncoder, LSTMDecoder
│   │
│   ├── evaluation/
│   │   └── metrics.py           # NSE, KGE, RMSE, PBIAS, CSI, POD, FAR, HSS
│   │
│   └── api/
│       └── main.py              # FastAPI REST endpoints
│
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
│
├── notebooks/
│   └── train_kaggle.ipynb       # Self-contained Kaggle training notebook
│
├── data/
│   ├── raw/
│   │   ├── era5/                # era5_YYYY.nc (23 files, ~500 MB total)
│   │   └── glofas/              # discharge_{station}.csv
│   └── processed/
│       ├── merged_bahadurabad.csv
│       ├── merged_guwahati.csv
│       └── merged_dibrugarh.csv
│
├── checkpoints/
│   ├── best_model.pt            # Trained weights
│   ├── scalers.pkl              # Fitted StandardScalers
│   └── results.json             # Test-set metrics
│
├── train.py                     # Training script (Hydra config)
├── predict.py                   # Inference script
├── RESEARCH.md                  # Literature review and citations
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/priyanshudebnathiitg/prag-dristi.git
cd prag-dristi
pip install -r requirements.txt
```

### 2. Set up data credentials

**ERA5 (Copernicus CDS):**
```
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api
key: your-cds-api-key
```

**GloFAS (Copernicus EWDS):**
```
# ~/.ewdsapirc
url: https://ewds.climate.copernicus.eu/api
key: your-ewds-api-key
```

Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) and [ewds.climate.copernicus.eu](https://ewds.climate.copernicus.eu).

### 3. Download data

```bash
# Download ERA5 reanalysis (23 years, ~500 MB)
python -m src.data.era5_download

# Download GloFAS discharge (per-station bounding boxes, ~3 MB total)
python -m src.data.glofas_download

# Build merged dataset
python -m src.data.build_dataset
```

### 4. Train

```bash
# Local training
python train.py

# Override config values
python train.py train.target_station=Guwahati train.epochs=50
```

Or use the Kaggle notebook (`notebooks/train_kaggle.ipynb`) with GPU for faster training (~20 min on T4).

### 5. Forecast

```bash
# 7-day forecast from the last available date
python predict.py --station Bahadurabad

# Forecast as of a specific historical date
python predict.py --station Bahadurabad --date 2022-07-15
```

Output:
```json
{
  "station": "Bahadurabad",
  "as_of": "2022-07-15",
  "forecast_horizon_days": 7,
  "dates": ["2022-07-16", ..., "2022-07-22"],
  "discharge_m3s": [45230.1, 48910.3, 52340.7, ...],
  "danger_discharge_m3s": 98600.0,
  "flood_alert": [false, false, true, ...]
}
```

### 6. Launch dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **NSE** | `1 − Σ(Qobs−Qsim)² / Σ(Qobs−Q̄obs)²` | 1=perfect, 0=mean baseline, <0=worse than mean |
| **KGE** | `1 − √((r−1)² + (α−1)² + (β−1)²)` | Balances correlation, bias, variability |
| **RMSE** | `√(mean((Qobs−Qsim)²))` | m³/s error |
| **CSI** | `TP / (TP+FP+FN)` | Flood event skill score |
| **POD** | `TP / (TP+FN)` | Hit rate (sensitivity) |
| **FAR** | `FP / (TP+FP)` | False alarm fraction |

Flood events are defined as Q > 57,512 m³/s (95th percentile of Bahadurabad discharge).

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping | patience=15 epochs |
| Gradient clipping | max_norm=1.0 |
| Teacher forcing | 0.5 → 0.0 (annealed over 30 epochs) |
| Batch size | 64 |
| Best epoch | 10 (of 100 max) |

---

## References

1. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. Sutskever, I., Vinyals, O. & Le, Q. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS*.
3. Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
4. Kratzert, F., Klotz, D., Brenner, C., Schulz, K. & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22(11), 6005–6022.
5. Nevo, S. et al. (2022). Flood forecasting with machine learning models in an operational framework. *Hydrology and Earth System Sciences*, 26(15), 4013–4032.
6. Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
7. Copernicus Emergency Management Service (2019). GloFAS-ERA5 operational global river discharge reanalysis 1979–present. *Copernicus Climate Change Service*.

Full literature review with 20+ papers: [RESEARCH.md](RESEARCH.md)

---

## License

MIT License. Data from ERA5 and GloFAS is subject to the [Copernicus Licence](https://cds.climate.copernicus.eu/disclaimer-privacy).

---

*Built by [Priyanshu Debnath](https://github.com/priyanshudebnathiitg) — Indian Institute of Technology Guwahati*
