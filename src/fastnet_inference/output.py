"""Forecast output utilities.

Supports concurrent writes to a single zarr store.
Safe because: chunk_size=1 on init_time + non-overlapping sample indices = no conflicts.

References:
- Dataset creation: https://docs.xarray.dev/en/stable/user-guide/data-structures.html
- Zarr region writes: https://docs.xarray.dev/en/stable/user-guide/io.html
"""

from pathlib import Path

import numpy as np
import xarray as xr


def create_output_store(
    path: Path,
    init_times: np.ndarray,
    rollout_steps: int,
    variables: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    freq_hours: int = 6,
) -> None:
    """
    Create empty zarr store for forecast output.

    In distributed setting, call from rank 0 only, then barrier before writes.
    """

    n_samples = len(init_times)
    n_vars = len(variables)
    n_gridpoints = len(lats)

    # Coordinates: dimension coords are just arrays, non-dimension coords use (dim, data) tuple
    lead_time_hours = np.arange(1, rollout_steps + 1) * freq_hours
    lead_time_td = lead_time_hours.astype("timedelta64[h]")

    coords = {
        "init_time": init_times,
        "lead_time": lead_time_td,
        "variable": variables,
        "latitude": ("grid", lats),
        "longitude": ("grid", lons),
    }

    # Create dataset with no data variable yet - just coordinates
    ds = xr.Dataset(coords=coords)

    # Define shape and encoding for the forecast variable
    shape = (n_samples, rollout_steps, n_vars, n_gridpoints)
    # Chunk by init_time=1 to enable safe concurrent writes
    encoding = {
        "forecast": {
            "chunks": (1, rollout_steps, n_vars, n_gridpoints),
            "dtype": np.float32,
        }
    }

    # Add forecast variable with placeholder data using dask to avoid memory allocation
    import dask.array as da

    ds["forecast"] = (
        ["init_time", "lead_time", "variable", "grid"],
        da.zeros(shape, dtype=np.float32, chunks=encoding["forecast"]["chunks"]),
    )

    # mode="w" creates new store, compute=False writes only metadata/structure
    ds.to_zarr(path, mode="w", compute=False, encoding=encoding)


def write_batch(
    path: Path,
    forecasts: np.ndarray,
    init_times: np.ndarray,
) -> None:
    """
    Write a batch of forecasts to pre-allocated store.

    Args:
        path: Path to zarr store
        forecasts: (batch, rollout_steps, variable, gridpoints) array
        init_times: Init time coordinates for this batch
    """
    # ensure init_times is cast to an array if scalar
    init_times = np.atleast_1d(init_times)
    ds = xr.Dataset(
        {"forecast": (["init_time", "lead_time", "variable", "grid"], forecasts)},
        coords={"init_time": init_times},
    )

    # region="auto" matches init_time coords against store to determine write region
    # safe for concurrent access when init_times are unique across ranks
    ds.to_zarr(path, mode="r+", region="auto")
