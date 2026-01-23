from typing import Annotated

import typer
import yaml

from fastnet_inference.eval import InferenceConfig, run_inference

__all__ = ("app",)
app = typer.Typer()


@app.command()
def run_pipeline(
    config_path: Annotated[str, typer.Argument()],
    dataset_path: Annotated[str | None, typer.Option()] = None,
):
    """Load config + run inference and save to zarr."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        if dataset_path is not None:
            cfg["dataset_path"] = dataset_path
        config = InferenceConfig(**cfg)
    run_inference(config)
