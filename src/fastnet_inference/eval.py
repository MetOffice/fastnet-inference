import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass
from fastnet_inference.data import AnemoiERA5Dataset
from fastnet_inference.variables import FORECAST_VARS_ORDER_FASTNET
from fastnet_inference.output import create_output_store, write_batch


logger = logging.getLogger(__name__)
F_LEN = len(FORECAST_VARS_ORDER_FASTNET)
DEVICE = "cpu"


@dataclass
class Config:
    dataset_path: str
    inference_model_path: str
    batch_size: int
    num_workers: int
    rollout_steps: int
    output_path: Path = Path(".") / "outputs"
    start_time: str | None = None
    end_time: str | None = None
    freq_hours: int = 6


@torch.inference_mode
def _rollout_loop(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    # FastNet expects forecast vars & forcings/constants as separate inputs
    forecast_features, non_forecast_features = batch[..., :F_LEN], batch[..., F_LEN:]
    # make tensor for outputs: batch, rollout, grid, variable
    B, _, G, V = forecast_features.shape
    rollout_steps = non_forecast_features.shape[1]
    forecasts = torch.empty(B, rollout_steps, G, V)
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


def full_loop(config: Config):
    # model = load_model(device=DEVICE)
    model = torch.jit.load("model_file_cpu")
    ds = AnemoiERA5Dataset(
        dataset_path=config.dataset_path,
        start=config.start_time,
        end=config.end_time,
        rollout_steps=config.rollout_steps,
    )

    forecast_mean, forecast_std = ds.mean[:F_LEN], ds.std[:F_LEN]

    def unnormalize(batch):
        return batch * forecast_std + forecast_mean

    def postprocess(batch):
        batch = unnormalize(batch)
        # output dataset has (init_time, lead_time, variable, grid) dim order
        return batch.permute(0, 1, 3, 2)

    dl = DataLoader(ds, batch_size=config.batch_size, num_workers=config.num_workers)

    create_output_store(
        path=config.output_path,
        init_times=ds.ds.dates[:len(ds)],  # accounts for rollout_steps
        rollout_steps=config.rollout_steps,
        variables=ds.ds.variables,
        lats=ds.ds.latitudes,
        lons=ds.ds.longitudes,
        freq_hours=config.freq_hours,
    )
    for batch, idxs in dl:
        predictions = _rollout_loop(model, batch)
        predictions = postprocess(predictions.cpu())
        write_batch(config.output_path, predictions, idxs)


def _plot(predictions: torch.Tensor, ds: AnemoiERA5Dataset) -> None:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=ds.ds.longitudes, y=ds.ds.latitudes, c=predictions[0, -1, :, 3])
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig("test.png")
