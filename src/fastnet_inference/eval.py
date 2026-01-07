import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass
from rich.logging import RichHandler
from rich.progress import track

from fastnet_inference.data import get_fastnet_var_order, AnemoiERA5Dataset
from fastnet_inference.output import create_output_store, write_batch
from fastnet_inference.model import load_model


logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferenceConfig:
    dataset_path: str
    batch_size: int
    num_workers: int
    rollout_steps: int
    output_path: Path = Path(".") / "outputs"
    start: str | None = None
    end: str | None = None
    freq_hours: int = 6
    model_version = 1.1

    def __post_init__(self):
        # auto-convert str for convenience
        self.output_path = Path(self.output_path)


@torch.inference_mode
def _rollout_loop(
    model: torch.nn.Module,
    forecast_features: torch.Tensor,
    non_forecast_features: torch.Tensor,
    rollout_steps: int,
) -> torch.Tensor:
    # make tensor for outputs: batch, rollout, grid, variable
    B, _, G, V = forecast_features.shape
    forecasts = torch.empty(B, rollout_steps, G, V, device=forecast_features.device)
    # get initial condition
    current_state = forecast_features[:, 0]
    # roll out model
    for t in range(rollout_steps):
        # supply the non-forecast vars according to rollout timestep
        # (we keep the rollout dim in non_forecast_features due to a for-loop inside the model)
        current_state = model(
            non_forecast_features[:, t : t + 1],
            current_state,  # autoregressive loop
        ).squeeze(1)
        forecasts[:, t] = current_state
    return forecasts


def run_inference(config: InferenceConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logger.info("Running inference with the following settings:")
    logger.info("%r", config)
    model = load_model(version=config.model_version, device=DEVICE)
    # model = torch.jit.load("model_file_cpu")

    # data setup
    forecast_vars, nonforecast_vars = get_fastnet_var_order()
    ds = AnemoiERA5Dataset(
        dataset_path=config.dataset_path,
        forecast_vars=forecast_vars,
        nonforecast_vars=nonforecast_vars,
        start=config.start,
        end=config.end,
        rollout_steps=config.rollout_steps,
    )
    # write-out empty zarr store of correct shape
    logger.info("Writing output store to %s", config.output_path)
    create_output_store(
        path=config.output_path,
        init_times=ds.ds.dates[: len(ds)],  # accounts for rollout_steps
        rollout_steps=config.rollout_steps,
        variables=forecast_vars,
        lats=ds.ds.latitudes,
        lons=ds.ds.longitudes,
        freq_hours=config.freq_hours,
    )
    logger.info("Written output store!")

    # define post-processing logic
    num_forecast_vars = len(forecast_vars)
    forecast_mean, forecast_std = (
        ds.mean[:num_forecast_vars],
        ds.std[:num_forecast_vars],
    )

    def unnormalize(batch):
        return batch * forecast_std + forecast_mean

    def postprocess(batch):
        batch = unnormalize(batch)
        # output dataset has (init_time, lead_time, variable, grid) dim order
        return batch.permute(0, 1, 3, 2)

    # loop through data
    dl = DataLoader(
        ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    for batch, idxs in track(dl, "Running inference..."):
        init_time = ds.ds.dates[idxs]
        logger.info("Rolling out from %s", init_time)
        # FastNet expects forecast vars & forcings/constants as separate inputs
        forecast_features, non_forecast_features = (
            batch[..., :num_forecast_vars],
            batch[..., num_forecast_vars:],
        )
        # with pin_memory=True, we can async transfer things to/from device.
        # non_blocking=True will let other CPU operations work in parallel before hitting a sync point
        # (i.e. when (non_)forecast_features is next used.)
        forecast_features = forecast_features.to(DEVICE, non_blocking=True)
        non_forecast_features = non_forecast_features.to(DEVICE, non_blocking=True)
        predictions = _rollout_loop(
            model=model,
            forecast_features=forecast_features,
            non_forecast_features=non_forecast_features,
            rollout_steps=config.rollout_steps,
        )
        logger.info(
            "...model evaluated!",
        )
        predictions = postprocess(predictions.cpu())
        logger.info(
            "Writing predictions from %s to %s...", init_time, config.output_path
        )
        write_batch(config.output_path, predictions, idxs)
    logger.info("Inference complete! Results have been saved to %s", config.output_path)


def _plot(predictions: torch.Tensor, ds: AnemoiERA5Dataset) -> None:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=ds.ds.longitudes, y=ds.ds.latitudes, c=predictions[0, -1, :, 3])
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig("test.png")
