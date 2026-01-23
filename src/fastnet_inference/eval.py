import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import zarr
from rich.logging import RichHandler
from rich.progress import track
from torch import distributed as dist
from torch.utils.data import DataLoader

from fastnet_inference.data import AnemoiERA5Dataset, create_dataloader, get_fastnet_var_order
from fastnet_inference.model import load_model
from fastnet_inference.output import create_output_store, write_batch

logger = logging.getLogger(__name__)

# see if we're in torchrun context
local_rank = os.environ.get("LOCAL_RANK", None)  # local rank = GPU id within a node
if local_rank is not None:
    local_rank = int(local_rank)
    DEVICE = f"cuda:{local_rank}"  # define device via local rank
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_distributed = world_size > 1
rank = int(os.environ.get("RANK", 0))  # global


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
    zip: bool = False

    def __post_init__(self):
        # auto-convert str for convenience
        self.output_path = Path(self.output_path)


@torch.inference_mode()
def _rollout_loop(
    model: torch.nn.Module,
    forecast_features: torch.Tensor,
    non_forecast_features: torch.Tensor,
    rollout_steps: int,
) -> torch.Tensor:
    device = forecast_features.device
    # make tensor for outputs: batch, rollout, grid, variable
    B, _, G, V = forecast_features.shape
    forecasts = torch.empty(B, rollout_steps, G, V, device=device)
    # get initial condition
    current_state = forecast_features[:, 0]
    logger.info("_rollout_loop: starting %d rollout steps on %r", rollout_steps, device)
    # roll out model
    for t in range(rollout_steps):
        logger.info("_rollout_loop [%s]: step %d/%d - calling model...", device, t + 1, rollout_steps)
        # supply the non-forecast vars according to rollout timestep
        # (we keep the rollout dim in non_forecast_features due to a for-loop inside the model)
        current_state = model(
            non_forecast_features[:, t : t + 1],
            current_state,  # autoregressive loop
        ).squeeze(1)
        # Force CUDA sync to ensure operation completed and catch hangs accurately
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forecasts[:, t] = current_state
        logger.info("_rollout_loop [%s]: step %d/%d - done", device, t + 1, rollout_steps)
    logger.info("_rollout_loop [%s]: finished all steps", device)
    return forecasts


def run_inference(config: InferenceConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger.info("Running inference with the following settings:")
    logger.info("%r", config)

    # handle distributed
    if is_distributed:
        from datetime import timedelta
        backend = "gloo" if DEVICE == "cpu" else "nccl"
        timeout = timedelta(hours=1)
        if backend == "gloo":
            dist.init_process_group(backend=backend, timeout=timeout)
        else:
            dist.init_process_group(backend=backend, timeout=timeout, device_id=local_rank or rank)

    model = load_model(version=config.model_version, device=DEVICE)

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

    if rank == 0:
        # write-out empty zarr store of correct shape
        logger.info("Writing output store to %r on rank %r", config.output_path, rank)
        create_output_store(
            path=config.output_path,
            init_times=ds.ds.dates[: ds.num_samples],  # accounts for rollout_steps
            rollout_steps=config.rollout_steps,
            variables=forecast_vars,
            lats=ds.ds.latitudes,
            lons=ds.ds.longitudes,
            freq_hours=config.freq_hours,
        )
        logger.info("Written output store!")
    
    # make sure other ranks don't eagerly write to store before it's ready
    if is_distributed: dist.barrier()

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

    dl = create_dataloader(
       ds,
       batch_size=config.batch_size,
       num_workers=config.num_workers,
       prefetch_factor=2 if config.num_workers > 0 else None,  # explicit prefetch
       persistent_workers=config.num_workers > 0,  # keep workers alive between batches
    )
    dl_iter = iter(dl)
    batch_num = 0
    while True:
        logger.info("Rank %r requesting batch %r from DataLoader...", rank, batch_num)
        try:
            batch, idxs = next(dl_iter)
        except StopIteration:
            logger.info("Rank %r exhausted DataLoader after %r batches", rank, batch_num)
            break
        init_time = ds.ds.dates[idxs]
        logger.info("Rolling out on rank %r from %r", rank, init_time)
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
        logger.info("Rank %r model evaluated!", rank)
        predictions = postprocess(predictions.cpu())
        logger.info(
            "Rank %r writing predictions from %r to %r...", rank, init_time, config.output_path
        )
        write_batch(config.output_path, predictions, init_time)
        logger.info(
            "Rank %r wrote batch %r (idxs=%r) to file.", rank, batch_num, idxs
        )
        batch_num += 1

    logger.info("Rank %r finished inference - waiting for other ranks...", rank)
    # wait for all ranks to finish before final saving and consolidation
    if is_distributed: dist.barrier()

    logger.info("Inference complete! Results have been saved to %r", config.output_path)

    if rank == 0:
        zarr.consolidate_metadata(config.output_path.with_suffix(".zarr"))
        # if config.zip:
        #     import shutil
        #     outfile_name = str(config.output_path)
        #     shutil.make_archive(outfile_name, "zip", base_dir=config.output_path.with_suffix(".zarr"))
        #
        #     logger.info("Since config.zip=%s, results also compressed to %s", config.zip, outfile_name + ".zip.")
    if is_distributed:
        dist.destroy_process_group()

# def _plot(predictions: torch.Tensor, ds: AnemoiERA5Dataset) -> None:
#     import cartopy.crs as ccrs
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
#     p = ax.scatter(x=ds.ds.longitudes, y=ds.ds.latitudes, c=predictions[0, -1, :, 3])
#     ax.coastlines()
#     ax.gridlines(draw_labels=True)
#     plt.colorbar(p, label="K", orientation="horizontal")
#     plt.savefig("test.png")
