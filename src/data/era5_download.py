"""
Download ERA5 reanalysis data for the Brahmaputra/Assam region via the
Copernicus Climate Data Store (CDS) API.

Setup (one-time):
  1. Register at https://cds.climate.copernicus.eu
  2. Accept the ERA5 licence on the dataset page
  3. Create ~/.cdsapirc with your UID and API key:
       url: https://cds.climate.copernicus.eu/api/v2
       key: <UID>:<API-KEY>
  4. pip install cdsapi

Usage:
  python -m src.data.era5_download          # downloads all years in configs/data.yaml
  python -m src.data.era5_download year=2020 # single year
"""

import os
import logging
from pathlib import Path

import cdsapi
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Variable name mapping: friendly name -> CDS short name
VARIABLE_MAP = {
    "total_precipitation": "total_precipitation",
    "2m_temperature": "2m_temperature",
    "surface_pressure": "surface_pressure",
    "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1",
    "u_component_of_wind_10m": "10m_u_component_of_wind",
    "v_component_of_wind_10m": "10m_v_component_of_wind",
}


def download_era5_year(client: cdsapi.Client, cfg: DictConfig, year: int) -> Path:
    """Download one year of ERA5 daily data and save as NetCDF."""
    out_dir = Path(cfg.era5.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_{year}.nc"

    if out_path.exists():
        log.info("ERA5 %d already downloaded, skipping.", year)
        return out_path

    cds_variables = [VARIABLE_MAP[v] for v in cfg.era5.variables if v in VARIABLE_MAP]
    area = list(cfg.era5.area)  # convert ListConfig -> plain list for JSON serialisation

    log.info("Downloading ERA5 for year %d ...", year)
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": cds_variables,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": area,
            "grid": [float(cfg.era5.resolution), float(cfg.era5.resolution)],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )
    log.info("Saved -> %s", out_path)
    return out_path


@hydra.main(config_path="../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    client = cdsapi.Client()
    for year in range(cfg.era5.start_year, cfg.era5.end_year + 1):
        download_era5_year(client, cfg, year)


if __name__ == "__main__":
    main()
