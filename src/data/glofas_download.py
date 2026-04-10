"""
Download GloFAS-ERA5 river discharge reanalysis for specific station points.

Instead of downloading the global domain (~2GB/year), we download one small
bounding box per station (0.2° padding around the station lat/lon = ~4 grid
cells). This reduces each download from ~2GB to ~1MB.

Setup:
  1. Register at https://ewds.climate.copernicus.eu
  2. Accept GloFAS licence at:
     https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical
  3. Create C:/Users/<you>/.ewdsapirc:
       url: https://ewds.climate.copernicus.eu/api
       key: your-ewds-uuid-key

Usage:
  python -m src.data.glofas_download
"""

import logging
from pathlib import Path

import cdsapi
import hydra
import xarray as xr
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)
EWDS_RC = Path.home() / ".ewdsapirc"
PADDING = 0.5   # degrees of padding around each station point


def get_ewds_client() -> cdsapi.Client:
    if not EWDS_RC.exists():
        raise FileNotFoundError(
            f"EWDS credentials not found at {EWDS_RC}.\n"
            "Create it with:\n"
            "  url: https://ewds.climate.copernicus.eu/api\n"
            "  key: your-ewds-api-key"
        )
    url, key = None, None
    with open(EWDS_RC) as f:
        for line in f:
            line = line.strip()
            if line.startswith("url:"):
                url = line.split(":", 1)[1].strip()
            elif line.startswith("key:"):
                key = line.split(":", 1)[1].strip()
    if not url or not key:
        raise ValueError(f"Could not parse url/key from {EWDS_RC}")
    return cdsapi.Client(url=url, key=key)


def download_station_year(
    client: cdsapi.Client,
    station_name: str,
    lat: float,
    lon: float,
    year: int,
    out_dir: Path,
) -> Path:
    """
    Download GloFAS discharge for a tiny bounding box around one station.
    Area = [N, W, S, E] with PADDING degrees on each side.
    File size: ~1-5 MB vs ~2 GB for global.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"glofas_{station_name.lower()}_{year}.nc"

    if out_path.exists():
        log.info("GloFAS %s %d already downloaded, skipping.", station_name, year)
        return out_path

    # Tight bounding box around the station
    area = [
        round(lat + PADDING, 2),   # N
        round(lon - PADDING, 2),   # W
        round(lat - PADDING, 2),   # S
        round(lon + PADDING, 2),   # E
    ]

    log.info(
        "Downloading GloFAS %s %d  bbox=[N%.1f W%.1f S%.1f E%.1f] ...",
        station_name, year, *area
    )
    client.retrieve(
        "cems-glofas-historical",
        {
            "system_version": "version_4_0",
            "hydrological_model": "lisflood",
            "product_type": "consolidated",
            "variable": "river_discharge_in_the_last_24_hours",
            "hyear": str(year),
            "hmonth": [f"{m:02d}" for m in range(1, 13)],
            "hday":   [f"{d:02d}" for d in range(1, 32)],
            "area":   area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )
    log.info("Saved -> %s", out_path)
    return out_path


def extract_station_timeseries(nc_path: Path, lat: float, lon: float) -> pd.DataFrame:
    """Extract nearest-neighbour daily discharge time series from a station NetCDF."""
    ds = xr.open_dataset(nc_path)
    varname = next(
        (v for v in ["dis24", "dis", "river_discharge", "DIS"] if v in ds.data_vars),
        list(ds.data_vars)[0]
    )
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    pt = ds[varname].sel({lat_dim: lat, lon_dim: lon}, method="nearest")
    df = pt.to_dataframe(name="discharge_m3s")[["discharge_m3s"]]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    ds.close()
    return df


@hydra.main(config_path="../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    client = get_ewds_client()
    out_base = Path(cfg.glofas.output_dir)

    for station in cfg.stations:
        name = station.name
        lat, lon = float(station.lat), float(station.lon)
        station_dir = out_base / name.lower()

        log.info("=== Station: %s (%.2f°N, %.2f°E) ===", name, lat, lon)

        yearly_dfs = []
        for year in range(cfg.glofas.start_year, cfg.glofas.end_year + 1):
            nc = download_station_year(client, name, lat, lon, year, station_dir)
            yearly_dfs.append(extract_station_timeseries(nc, lat, lon))

        # Merge all years into one CSV per station
        merged = pd.concat(yearly_dfs).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]

        csv_path = out_base / f"discharge_{name.lower()}.csv"
        merged.to_csv(csv_path)
        log.info("Station %s: %d days saved -> %s", name, len(merged), csv_path)


if __name__ == "__main__":
    main()
