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
    coords = {
        "init_time": init_times,
        "lead_time": np.arange(1, rollout_steps + 1) * freq_hours,
        "variable": variables,
        "latitude": ("grid", lats),
        "longitude": ("grid", lons),
    }

    ds = xr.Dataset(data_vars, coords=coords)
    ds["lead_time"].attrs["units"] = "hours"

    # Chunk by single init_time for safe concurrent writes
    # -1 means "entire dimension in one chunk"
    chunks = {"init_time": 1, "lead_time": -1, "variable": -1, "grid": -1}

    # mode="w" creates new store (overwrites if exists)
    # compute=False writes structure only, no data
    ds.chunk(chunks).to_zarr(path, mode="w", compute=False)


def write_batch(
    path: Path,
    forecasts: np.ndarray,
    idxs: list[int] | np.ndarray,
) -> None:
    """
    Write a batch of forecasts to pre-allocated store.

    Args:
        path: Path to zarr store
        forecasts: (batch, rollout_steps, variable, gridpoints) array
        idxs: Init_time indices for this batch (list, array, or tensor)
    """
    # Handle torch tensor of indices from DataLoader
    if hasattr(idxs, "tolist"):
        idxs = idxs.tolist()

    # Minimal dataset for region write - coords not needed
    ds = xr.Dataset(
        {"forecast": (["init_time", "lead_time", "variable", "grid"], forecasts)}
    )

    # mode="r+" for modifying existing store (default for region writes)
    # Region write is safe for concurrent access when writing to distinct chunks
    start_idx, end_idx = idxs[0], idxs[-1] + 1
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
