"""
Build the merged, analysis-ready dataset from raw downloads.

Pipeline:
  ERA5 NetCDF files (data/raw/era5/era5_YYYY.nc)
      → extract nearest grid point to each station
      → resample 6-hourly → daily mean
      → unit conversion (m→mm precip, K→°C temp)

  GloFAS per-station CSVs (data/raw/glofas/discharge_{station}.csv)
      → already daily, just read and align

  Merge ERA5 + GloFAS on date index → inner join → drop NaN
  Save → data/processed/merged_{station}.csv

Usage:
  python -m src.data.build_dataset
"""

import io
import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

ERA5_VAR_RENAME = {
    "tp":    "precip_m",
    "t2m":   "temp_k",
    "sp":    "pressure_pa",
    "swvl1": "soil_moisture",
    "u10":   "wind_u",
    "v10":   "wind_v",
}


def _open_era5_file(path: Path) -> xr.Dataset:
    """
    Open an ERA5 file that may be either a plain NetCDF4 or a ZIP archive
    containing multiple NetCDF4 files (new CDS API format returns ZIPs with
    two files: one for instantaneous variables, one for accumulated).
    Returns a merged xarray Dataset.
    """
    with open(path, "rb") as fh:
        magic = fh.read(2)

    if magic != b"PK":
        # Plain NetCDF4
        ds = xr.open_dataset(path, engine="netcdf4")
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        return ds

    # ZIP archive -- extract each .nc to a temp file and open with netcdf4
    datasets = []
    with zipfile.ZipFile(path) as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise ValueError(f"No .nc files found inside ZIP: {path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in nc_names:
                tmp_path = Path(tmpdir) / name
                tmp_path.write_bytes(zf.read(name))
                ds = xr.open_dataset(tmp_path, engine="netcdf4")
                # Load into memory before temp dir is deleted
                datasets.append(ds.load())
                ds.close()

    merged = xr.merge(datasets, compat="override")

    # New CDS API names the time dimension 'valid_time' instead of 'time'
    if "valid_time" in merged.dims and "time" not in merged.dims:
        merged = merged.rename({"valid_time": "time"})

    return merged


def extract_point_era5(era5_dir: Path, lat: float, lon: float) -> pd.DataFrame:
    """
    Load all ERA5 annual files, extract the nearest grid point to (lat, lon),
    resample from 6-hourly to daily mean, and return a DataFrame.

    Handles both plain NetCDF4 and ZIP-packaged NetCDF (new CDS API format).
    """
    files = sorted(era5_dir.glob("era5_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No ERA5 files found in {era5_dir}\n"
            "Run: python -m src.data.era5_download"
        )

    chunks = []
    for f in files:
        log.info("  ERA5 reading %s", f.name)
        ds = _open_era5_file(f)

        # Nearest-neighbour point extraction
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        pt = ds.sel({lat_dim: lat, lon_dim: lon}, method="nearest")

        # 6-hourly → daily mean
        daily = pt.resample(time="1D").mean()
        df = daily.to_dataframe().reset_index().rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        keep = [c for c in ERA5_VAR_RENAME if c in df.columns]
        df = df[keep].rename(columns=ERA5_VAR_RENAME)
        chunks.append(df)
        ds.close()

    era5 = pd.concat(chunks).sort_index()
    era5 = era5[~era5.index.duplicated(keep="first")]

    # Unit conversions
    if "precip_m" in era5.columns:
        era5["precip_mm"] = era5["precip_m"] * 1000.0
        era5.drop(columns=["precip_m"], inplace=True)
    if "temp_k" in era5.columns:
        era5["temp_c"] = era5["temp_k"] - 273.15
        era5.drop(columns=["temp_k"], inplace=True)

    log.info(
        "  ERA5: %d daily rows  (%s → %s)",
        len(era5), era5.index[0].date(), era5.index[-1].date(),
    )
    return era5


def load_glofas_station(glofas_dir: Path, station_name: str) -> pd.DataFrame:
    """
    Load the pre-built per-station GloFAS discharge CSV produced by
    glofas_download.py.

    Falls back to searching for per-year NetCDF files (legacy path) if the
    CSV does not exist yet.
    """
    # Primary path: per-station CSV written by glofas_download.py
    csv_path = glofas_dir / f"discharge_{station_name.lower()}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df = df[["discharge_m3s"]].sort_index()
        df = df[~df.index.duplicated(keep="first")]
        log.info(
            "  GloFAS: %d daily rows  (%s → %s)",
            len(df), df.index[0].date(), df.index[-1].date(),
        )
        return df

    # Legacy fallback: per-year NetCDF files in a station sub-directory
    station_dir = glofas_dir / station_name.lower()
    nc_files = sorted(station_dir.glob(f"glofas_{station_name.lower()}_*.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No GloFAS data for station '{station_name}'.\n"
            f"Expected: {csv_path}\n"
            f"Or NetCDF files in: {station_dir}\n"
            "Run: python -m src.data.glofas_download"
        )

    chunks = []
    for f in nc_files:
        ds = xr.open_dataset(f)
        varname = _detect_discharge_var(ds)
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        # Take the spatial mean over the tiny bbox (only a few cells)
        pt = ds[varname].mean(dim=[lat_dim, lon_dim])
        df = pt.to_dataframe(name="discharge_m3s")[["discharge_m3s"]]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        chunks.append(df)
        ds.close()

    merged = pd.concat(chunks).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    log.info(
        "  GloFAS (NC): %d daily rows  (%s → %s)",
        len(merged), merged.index[0].date(), merged.index[-1].date(),
    )
    return merged


def _detect_discharge_var(ds: xr.Dataset) -> str:
    for candidate in ["dis24", "dis", "river_discharge", "DIS", "q"]:
        if candidate in ds.data_vars:
            return candidate
    var = list(ds.data_vars)[0]
    log.warning("Could not detect discharge variable, using: %s", var)
    return var


@hydra.main(config_path="../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    era5_dir   = Path(cfg.era5.output_dir)
    glofas_dir = Path(cfg.glofas.output_dir)
    out_dir    = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for station in cfg.stations:
        name     = station.name
        lat, lon = float(station.lat), float(station.lon)
        log.info("Building dataset for: %s (%.2f°N, %.2f°E)", name, lat, lon)

        era5_df   = extract_point_era5(era5_dir, lat, lon)
        glofas_df = load_glofas_station(glofas_dir, name)

        # Inner join on date -- only keep days where both sources have data
        merged = era5_df.join(glofas_df, how="inner")
        before = len(merged)
        merged.dropna(inplace=True)
        log.info(
            "  Merged: %d rows  (dropped %d NaN rows)",
            len(merged), before - len(merged),
        )

        out_path = out_dir / f"merged_{name.lower()}.csv"
        merged.to_csv(out_path)
        log.info("  Saved -> %s\n", out_path)

    log.info("All stations done. Run train.py next.")


if __name__ == "__main__":
    main()
