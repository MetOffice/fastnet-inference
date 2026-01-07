import typer
import yaml
from typing import Annotated
from fastnet_inference.eval import InferenceConfig, run_inference


__all__ = ("app",)
app = typer.Typer()


@app.command()
def run_pipeline(
    config_path: Annotated[str, typer.Argument()],
):
    """Load config + run inference and save to zarr."""
    with open(config_path, "r") as f:
        config = InferenceConfig(**yaml.safe_load(f))
    run_inference(config)
