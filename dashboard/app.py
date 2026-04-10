"""
Streamlit dashboard for Prag-Dristi: Assam Flood Forecasting Engine.

Run:
  streamlit run dashboard/app.py

Reads from:
  - data/processed/merged_{station}.csv  (historical data)
  - checkpoints/best_model.pt + scalers.pkl  (trained model)
  - configs/  (station metadata, model config)

If the model is not trained yet, shows only historical data.
"""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from omegaconf import OmegaConf

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
DATA_CFG = OmegaConf.load(PROJECT_ROOT / "configs" / "data.yaml")
STATION_NAMES = [s.name for s in DATA_CFG.stations]
STATION_META = {s.name: s for s in DATA_CFG.stations}

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Prag-Dristi | Assam Flood Forecasting",
    page_icon="🌊",
    layout="wide",
)

st.title("Prag-Dristi — Assam Flood Forecasting Engine")
st.caption("7-day river discharge forecasts for the Brahmaputra basin using LSTM Encoder-Decoder with Attention")

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Settings")
    selected_station = st.selectbox("Monitoring Station", STATION_NAMES)
    station_meta = STATION_META[selected_station]

    st.markdown(f"""
    **River:** {station_meta.river}
    **Lat/Lon:** {station_meta.lat}°N, {station_meta.lon}°E
    **Danger level:** {station_meta.danger_discharge:,.0f} m³/s
    """)

    st.divider()
    history_days = st.slider("Historical context (days)", min_value=30, max_value=365, value=90)
    show_uncertainty = st.checkbox("Show ±10% uncertainty band", value=True)


# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
@st.cache_data
def load_history(station: str) -> pd.DataFrame | None:
    path = PROJECT_ROOT / f"data/processed/merged_{station.lower()}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


@st.cache_resource
def load_model_and_scalers():
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not (ckpt_dir / "best_model.pt").exists():
        return None, None, None

    import joblib
    import torch
    from src.models.lstm_seq2seq import FloodForecastModel
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "model.yaml")
    scalers = joblib.load(ckpt_dir / "scalers.pkl")
    feat_scaler = scalers["feature"]
    tgt_scaler = scalers["target"]

    # Infer input_size from scaler
    input_size = feat_scaler.n_features_in_

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
    sd = torch.load(ckpt_dir / "best_model.pt", map_location="cpu")
    # Remap flat keys from Kaggle notebook to nested local layout
    if any(k.startswith("encoder.") and not k.startswith("encoder.lstm.") for k in sd):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder.") and not k.startswith("encoder.lstm."):
                new_sd["encoder.lstm." + k[len("encoder."):]] = v
            elif k.startswith("decoder.") and not any(k.startswith(f"decoder.{p}.") for p in ("lstm","attention","fc")):
                new_sd["decoder.lstm." + k[len("decoder."):]] = v
            elif k.startswith("attention."):
                new_sd["decoder." + k] = v
            elif k.startswith("fc."):
                new_sd["decoder." + k] = v
            else:
                new_sd[k] = v
        sd = new_sd
    model.load_state_dict(sd)
    model.eval()
    return model, feat_scaler, tgt_scaler


def run_forecast(station: str) -> dict | None:
    try:
        from predict import forecast
        return forecast(station, project_root=PROJECT_ROOT)
    except Exception as e:
        return None


# ------------------------------------------------------------------ #
# Main content
# ------------------------------------------------------------------ #
hist_df = load_history(selected_station)

if hist_df is None:
    st.warning(
        f"No processed data found for **{selected_station}**. "
        "Run `python -m src.data.build_dataset` first."
    )
    st.stop()

model, feat_scaler, tgt_scaler = load_model_and_scalers()
model_ready = model is not None

# ------------------------------------------------------------------ #
# KPI row
# ------------------------------------------------------------------ #
col1, col2, col3, col4 = st.columns(4)
danger = station_meta.danger_discharge

recent_q = hist_df["discharge_m3s"].iloc[-1] if "discharge_m3s" in hist_df.columns else None
recent_rain = hist_df["precip_mm"].iloc[-1] if "precip_mm" in hist_df.columns else None
max_7d = hist_df["discharge_m3s"].iloc[-7:].max() if "discharge_m3s" in hist_df.columns else None

with col1:
    st.metric("Latest Discharge", f"{recent_q:,.0f} m³/s" if recent_q else "N/A",
              delta=f"{recent_q - danger:,.0f} vs danger" if recent_q else None,
              delta_color="inverse")
with col2:
    st.metric("Yesterday Rainfall", f"{recent_rain:.1f} mm" if recent_rain else "N/A")
with col3:
    st.metric("7-day Peak Discharge", f"{max_7d:,.0f} m³/s" if max_7d else "N/A")
with col4:
    status = "⚠ ABOVE DANGER" if recent_q and recent_q >= danger else "✓ Below Danger"
    st.metric("Flood Status", status)

st.divider()

# ------------------------------------------------------------------ #
# Forecast section
# ------------------------------------------------------------------ #
if model_ready:
    forecast_result = run_forecast(selected_station)
else:
    forecast_result = None
    st.info("Model not trained yet. Train with `python train.py` to enable forecasts.")

# ------------------------------------------------------------------ #
# Main chart: historical + forecast
# ------------------------------------------------------------------ #
plot_hist = hist_df["discharge_m3s"].iloc[-history_days:] if "discharge_m3s" in hist_df.columns else None

fig = go.Figure()

# Historical line
if plot_hist is not None:
    fig.add_trace(go.Scatter(
        x=plot_hist.index,
        y=plot_hist.values,
        mode="lines",
        name="Observed discharge",
        line=dict(color="#1f77b4", width=1.5),
    ))

    # Danger level line
    fig.add_hline(
        y=danger,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Danger level ({danger:,.0f} m³/s)",
        annotation_position="top right",
    )

# Forecast
if forecast_result:
    fc_dates = pd.to_datetime(forecast_result["dates"])
    fc_q = forecast_result["discharge_m3s"]
    flood_alert = forecast_result.get("flood_alert") or [False] * len(fc_q)

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_q,
        mode="lines+markers",
        name="7-day forecast",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        marker=dict(
            size=8,
            color=["#d62728" if a else "#ff7f0e" for a in flood_alert],
            symbol=["diamond" if a else "circle" for a in flood_alert],
        ),
    ))

    if show_uncertainty:
        upper = [q * 1.10 for q in fc_q]
        lower = [q * 0.90 for q in fc_q]
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±10% band",
            showlegend=True,
        ))

fig.update_layout(
    title=f"Brahmaputra Discharge at {selected_station}",
    xaxis_title="Date",
    yaxis_title="Discharge (m³/s)",
    hovermode="x unified",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Forecast table
# ------------------------------------------------------------------ #
if forecast_result:
    st.subheader("7-Day Forecast Table")
    fc_df = pd.DataFrame({
        "Date": forecast_result["dates"],
        "Discharge (m³/s)": [f"{q:,.0f}" for q in forecast_result["discharge_m3s"]],
        "Flood Alert": ["🔴 YES" if a else "🟢 No" for a in (forecast_result["flood_alert"] or [])],
    })
    st.dataframe(fc_df, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ #
# Historical rainfall chart
# ------------------------------------------------------------------ #
if "precip_mm" in hist_df.columns:
    rain_hist = hist_df["precip_mm"].iloc[-history_days:]
    fig_rain = go.Figure(go.Bar(
        x=rain_hist.index,
        y=rain_hist.values,
        name="Rainfall",
        marker_color="#2ca02c",
    ))
    fig_rain.update_layout(
        title=f"Daily Precipitation near {selected_station} (ERA5)",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        height=250,
    )
    st.plotly_chart(fig_rain, use_container_width=True)

# ------------------------------------------------------------------ #
# Model metrics (if results.json exists)
# ------------------------------------------------------------------ #
results_path = PROJECT_ROOT / "checkpoints" / "results.json"
if results_path.exists():
    import json
    with open(results_path) as f:
        results = json.load(f)

    st.subheader("Model Performance (Test Set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("NSE", f"{results.get('NSE', 'N/A'):.3f}")
    m2.metric("KGE", f"{results.get('KGE', 'N/A'):.3f}")
    rmse_val = results.get('RMSE_m3s', results.get('RMSE', None))
    m3.metric("RMSE (m³/s)", f"{rmse_val:,.0f}" if rmse_val is not None else "N/A")
    m4.metric("CSI (floods)", f"{results.get('CSI', 'N/A'):.3f}")
    m5.metric("POD", f"{results.get('POD', 'N/A'):.3f}")

# ------------------------------------------------------------------ #
# Station map
# ------------------------------------------------------------------ #
st.subheader("Station Location")
import folium
from streamlit_folium import st_folium

m = folium.Map(location=[26.5, 92.5], zoom_start=6, tiles="CartoDB positron")
for s in DATA_CFG.stations:
    color = "red" if s.name == selected_station else "blue"
    folium.CircleMarker(
        location=[s.lat, s.lon],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=f"{s.name} ({s.river})\nDanger: {s.danger_discharge:,.0f} m³/s",
        popup=folium.Popup(
            f"<b>{s.name}</b><br>River: {s.river}<br>Danger level: {s.danger_discharge:,.0f} m³/s",
            max_width=200,
        ),
    ).add_to(m)

st_folium(m, height=350, use_container_width=True)

st.caption("Prag-Dristi v1.0 · Data: ERA5 + GloFAS (Copernicus CDS) · Model: LSTM Encoder-Decoder with Bahdanau Attention")
