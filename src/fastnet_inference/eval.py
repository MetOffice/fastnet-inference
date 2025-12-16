import torch
import numpy
import logging
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from fastnet_inference.data import AnemoiERA5Dataset 
from fastnet_inference.variables import FORECAST_VARS_ORDER_FASTNET 

logger = logging.getLogger(__name__)
F_LEN = len(FORECAST_VARS_ORDER_FASTNET)


@dataclass
class Config:
    dataset_path: str
    inference_model_path: str
    batch_size: int
    num_workers: int
    rollout_steps: int
    start_time: str | None = None
    end_time: str | None = None


@torch.inference_mode
def _rollout_loop(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    # FastNet expects forecast vars & forcings/constants as separate inputs
    forecast_features, non_forecast_features = batch[..., :F_LEN], batch[..., F_LEN:]
    # make tensor for outputs
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
            non_forecast_features[:, t:t+1, :, :],
            current_state,  # autoregressive loop
        ).squeeze(1)
        forecasts[:, t] = current_state
        
    return forecasts


def full_loop(config: Config):
    model = torch.jit.load(config.inference_model_path)

    ds = AnemoiERA5Dataset(
        dataset_path=config.dataset_path,
        start=config.start_time,
        end=config.end_time,
        rollout_steps=config.rollout_steps,
    )

    def unnormalize(batch):
        return batch * ds.std[:F_LEN] + ds.mean[:F_LEN]

    dl = DataLoader(ds, batch_size=config.batch_size, num_workers=config.num_workers)

    for batch in dl:
        predictions = _rollout_loop(model, batch)
        # for i in range(predictions.shape[-1]):
        #     data = predictions[..., i]
        #     print(torch.min(data), torch.max(data))
        predictions = unnormalize(predictions)
        _plot(predictions, ds)

def _plot(predictions: torch.Tensor, ds: AnemoiERA5Dataset) -> None:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=ds.ds.longitudes, y=ds.ds.latitudes, c=predictions[0, 1, :, 3])
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig("test.png")
