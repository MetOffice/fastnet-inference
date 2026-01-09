import typer
import yaml
from typing import Annotated
from fastnet_inference.eval import InferenceConfig, run_inference


__all__ = ("app",)
app = typer.Typer()

def inspect_output(path):
    import xarray as xr
    """Load and inspect the saved forecast dataset."""
    ds = xr.open_zarr(path)

    print("=== Dataset Overview ===")
    print(ds)
    print()

    print("=== Dimensions ===")
    for dim, size in ds.sizes.items():
        print(f"  {dim}: {size}")
    print()

    print("=== Coordinates ===")
    print(f"  init_time: {ds.init_time.values[0]} to {ds.init_time.values[-1]}")
    print(f"  lead_time: {ds.lead_time.values} ({ds.lead_time.attrs.get('units', 'unknown')})")
    print(f"  variables: {ds.variable.values[:5]}... ({len(ds.variable)} total)")
    print()

    print("=== Forecast Data ===")
    forecast = ds.forecast
    print(f"  shape: {forecast.shape}")
    print(f"  dtype: {forecast.dtype}")
    print(f"  min: {float(forecast.min()):.3f}")
    print(f"  max: {float(forecast.max()):.3f}")
    print(f"  mean: {float(forecast.mean()):.3f}")
    print()

    # Check for NaNs (unfilled regions)
    nan_count = forecast.isnull().sum().values
    print(f"  NaN count: {nan_count} ({100 * nan_count / forecast.size:.1f}%)")


@app.command()
def run_pipeline(
    config_path: Annotated[str, typer.Argument()],
    dataset_path: Annotated[str | None, typer.Option()] = None,
):
    """Load config + run inference and save to zarr."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        if dataset_path is not None:
            cfg["dataset_path"] = dataset_path
        config = InferenceConfig(**cfg)
    print(dataset_path)
    print(config)
    run_inference(config)
    inspect_output(config.dataset_path)
