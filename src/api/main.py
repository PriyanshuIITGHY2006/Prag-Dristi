"""
FastAPI prediction server for the Assam Flood Forecasting Engine.

Endpoints:
  GET  /health              -- liveness check
  GET  /stations            -- list available stations
  POST /forecast            -- generate a 7-day discharge forecast
  GET  /forecast/{station}  -- convenience GET endpoint

Usage:
  uvicorn src.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from predict import forecast as run_forecast
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CFG = OmegaConf.load(PROJECT_ROOT / "configs" / "data.yaml")
VALID_STATIONS = [s.name for s in DATA_CFG.stations]

app = FastAPI(
    title="Prag-Dristi: Assam Flood Forecasting API",
    description="7-day river discharge forecasts for the Brahmaputra basin.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    station: str
    as_of_date: str | None = None

    @field_validator("station")
    @classmethod
    def validate_station(cls, v: str) -> str:
        if v not in VALID_STATIONS:
            raise ValueError(f"Unknown station '{v}'. Valid options: {VALID_STATIONS}")
        return v

    @field_validator("as_of_date")
    @classmethod
    def validate_date(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError("as_of_date must be ISO format: YYYY-MM-DD")
        return v


class ForecastResponse(BaseModel):
    station: str
    as_of: str
    forecast_horizon_days: int
    dates: list[str]
    discharge_m3s: list[float]
    danger_discharge_m3s: float | None
    flood_alert: list[bool] | None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations")
def list_stations():
    return {
        "stations": [
            {
                "name": s.name,
                "lat": s.lat,
                "lon": s.lon,
                "river": s.river,
                "danger_discharge_m3s": s.danger_discharge,
            }
            for s in DATA_CFG.stations
        ]
    }


@app.post("/forecast", response_model=ForecastResponse)
def post_forecast(req: ForecastRequest):
    try:
        result = run_forecast(req.station, req.as_of_date, project_root=PROJECT_ROOT)
        return ForecastResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{station}", response_model=ForecastResponse)
def get_forecast(station: str, as_of_date: str | None = None):
    if station not in VALID_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station}' not found.")
    try:
        result = run_forecast(station, as_of_date, project_root=PROJECT_ROOT)
        return ForecastResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
