"""Forecast output utilities.

Supports concurrent writes to a single zarr store.
Safe because: chunk_size=1 on init_time + non-overlapping sample indices = no conflicts.

References:
- Dataset creation: https://docs.xarray.dev/en/stable/user-guide/data-structures.html
- Zarr region writes: https://docs.xarray.dev/en/stable/user-guide/io.html
"""

import numpy as np
import xarray as xr
from pathlib import Path
from collections.abc import Sequence


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

    # Data variable: tuple of (dims, data)
    # See: https://docs.xarray.dev/en/stable/user-guide/data-structures.html
    data_vars = {
        "forecast": (
            ["init_time", "lead_time", "variable", "grid"],
            np.empty(
                (n_samples, rollout_steps, n_vars, n_gridpoints), dtype=np.float32
            ),
        )
    }

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

    ds = xr.Dataset(data_vars, coords=coords)

    # mode="w" creates new store (overwrites if exists)
    ds.to_zarr(path, mode="w")


def write_batch(
    path: Path,
    forecasts: np.ndarray,
    idxs: Sequence[int],
) -> None:
    """
    Write a batch of forecasts to pre-allocated store.

    Args:
        path: Path to zarr store
        forecasts: (batch, rollout_steps, variable, gridpoints) array
        idxs: Init_time indices for this batch (list, array, or tensor)
    """
    # minimal dataset for region write - coords not needed
    ds = xr.Dataset(
        {"forecast": (["init_time", "lead_time", "variable", "grid"], forecasts)}
    )

    # region write is safe for concurrent access when init_times are unique
    start_idx = idxs[0]
    end_idx = idxs[-1] + 1  # slice end is exclusive
    ds.to_zarr(
        path,
        mode="r+",
        region={
            "init_time": slice(start_idx, end_idx),
            "lead_time": slice(None),
            "variable": slice(None),
            "grid": slice(None),
        },
    )
