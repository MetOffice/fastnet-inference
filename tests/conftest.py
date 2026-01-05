import numpy as np
import pytest
import zarr
from pathlib import Path


def create_synthetic_anemoi_zarr(
    path: Path,
    variables: list[str],
    n_times: int,
    n_gridpoints: int,
    freq_hours: int = 6,
    seed: int = 670,
) -> Path:
    """
    Create minimal anemoi-compatible zarr for testing.

    NOTE: This was reverse-engineered to match the spec of the dataset generated with anemoi-datasets==0.5.28.
          It may be that this fails in future if this package is updated.
          Including a small file in the repo that is re-generated with future anemoi-datasets versions is also a solution.

    Generates random data with computed statistics.
    """
    rng = np.random.default_rng(seed)
    n_vars = len(variables)

    # Generate random data (time, variable, ensemble=1, cell)
    data = rng.standard_normal((n_times, n_vars, 1, n_gridpoints)).astype(np.float32)

    # Dates and coordinates
    start = np.datetime64("2020-01-01T00:00:00")
    dates = np.array(
        [start + np.timedelta64(i * freq_hours, "h") for i in range(n_times)]
    )
    lats = rng.uniform(-90, 90, n_gridpoints)
    lons = rng.uniform(-180, 180, n_gridpoints)

    # Statistics (computed from data)
    mean = data.mean(axis=(0, 2, 3))
    stdev = data.std(axis=(0, 2, 3))
    minimum = data.min(axis=(0, 2, 3))
    maximum = data.max(axis=(0, 2, 3))
    sums = data.sum(axis=(0, 2, 3))
    squares = (data**2).sum(axis=(0, 2, 3))
    count = np.full(n_vars, n_times * n_gridpoints, dtype=np.float64)
    has_nans = np.zeros(n_vars, dtype=np.float64)

    # Create zarr store
    root = zarr.open(str(path), mode="w")
    root.create_dataset("data", data=data, chunks=(1, n_vars, 1, n_gridpoints))
    root.create_dataset("dates", data=dates)
    root.create_dataset("latitudes", data=lats)
    root.create_dataset("longitudes", data=lons)
    root.create_dataset("mean", data=mean)
    root.create_dataset("stdev", data=stdev)
    root.create_dataset("minimum", data=minimum)
    root.create_dataset("maximum", data=maximum)
    root.create_dataset("sums", data=sums)
    root.create_dataset("squares", data=squares)
    root.create_dataset("count", data=count)
    root.create_dataset("has_nans", data=has_nans)

    # Required attributes
    root.attrs.update(
        {
            "version": "0.30",
            "frequency": f"{freq_hours}h",
            "variables": variables,
            "ensemble_dimension": 1,
            "field_shape": [n_gridpoints],
            "flatten_grid": True,
        }
    )

    # Array dimension attributes (required by anemoi-datasets)
    root["data"].attrs["_ARRAY_DIMENSIONS"] = ["time", "variable", "ensemble", "cell"]
    root["dates"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["latitudes"].attrs["_ARRAY_DIMENSIONS"] = ["cell"]
    root["longitudes"].attrs["_ARRAY_DIMENSIONS"] = ["cell"]

    return path


@pytest.fixture
def synthetic_era5(tmp_path) -> Path:
    """Create a synthetic ERA5-like dataset for testing."""
    from fastnet_inference.data import get_fastnet_var_order

    forecast_vars, nonforecast_vars = get_fastnet_var_order()
    variables = [*forecast_vars, *nonforecast_vars]

    return create_synthetic_anemoi_zarr(
        path=tmp_path / "synthetic_era5.zarr",
        variables=variables,
        n_times=3,
        n_gridpoints=40320,  # O96
    )
