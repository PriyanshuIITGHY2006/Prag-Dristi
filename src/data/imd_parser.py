"""
Parse IMD (India Meteorological Department) gridded daily rainfall NetCDF files
and extract area-mean rainfall over Assam.

IMD publishes 0.25° gridded daily rainfall from 1901-present.
Download: https://www.imdpune.gov.in/lrfindex.php  (free, requires registration)
File naming: RF25_ind{year}_rfp25.nc

This module reads those files and produces a daily time series of
spatially-averaged rainfall over the Assam bounding box.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# Assam bounding box
ASSAM_BBOX = {"lat_min": 24.0, "lat_max": 28.0, "lon_min": 89.5, "lon_max": 97.0}


def load_imd_rainfall(nc_path: str | Path, bbox: dict = ASSAM_BBOX) -> pd.DataFrame:
    """
    Load IMD gridded rainfall NetCDF and return daily area-mean over bbox.

    Returns:
        DataFrame with columns ['date', 'imd_rainfall_mm'] indexed by date.
    """
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"IMD file not found: {path}")

    log.info("Loading IMD rainfall from %s", path)
    ds = xr.open_dataset(path)

    # IMD NetCDFs typically use 'RAINFALL' or 'rf' as the variable name
    varname = _detect_rainfall_var(ds)
    log.info("Detected rainfall variable: %s", varname)

    # Subset to Assam bounding box
    da = ds[varname].sel(
        latitude=slice(bbox["lat_min"], bbox["lat_max"]),
        longitude=slice(bbox["lon_min"], bbox["lon_max"]),
    )

    # Spatial mean, ignoring fill values (IMD uses -999.0 as missing)
    da = da.where(da > 0)
    daily_mean = da.mean(dim=["latitude", "longitude"]).to_pandas()

    df = pd.DataFrame({"imd_rainfall_mm": daily_mean})
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df


def _detect_rainfall_var(ds: xr.Dataset) -> str:
    candidates = ["RAINFALL", "rf", "precip", "precipitation", "rain"]
    for c in candidates:
        if c in ds.data_vars:
            return c
    # Fall back to first variable
    var = list(ds.data_vars)[0]
    log.warning("Could not detect rainfall variable, using: %s", var)
    return var


def load_all_imd(directory: str | Path, bbox: dict = ASSAM_BBOX) -> pd.DataFrame:
    """Load and concatenate multiple IMD annual NetCDF files from a directory."""
    files = sorted(Path(directory).glob("RF25_ind*_rfp25.nc"))
    if not files:
        raise FileNotFoundError(f"No IMD files found in {directory}")

    chunks = [load_imd_rainfall(f, bbox) for f in files]
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    log.info("Loaded IMD rainfall: %d days (%s to %s)", len(df), df.index[0].date(), df.index[-1].date())
    return df
