"""Define data-related functions and classes.

Note: the Dataset defined here loads rollout_steps data points for every timestep.
That means that there's rollout_steps - 1 redundant I/O calls that aren't cached for every lead time.
Improving this would be desirable in a high-throughput scenario.
(e.g. loading chunks of data and accessing those multiple times in an IterableDataset).
"""

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import Dataset, Dataloader
from anemoi.datasets import open_dataset

from fastnet_inference.variables import (
    FORECAST_VARS_ORDER_FASTNET,
    NONFORECAST_VARS_ORDER_FASTNET,
    LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND,
)


def get_fastnet_var_order() -> tuple[list[str], list[str]]:
    forecast_ordered_vars: list[str] = []
    nonforecast_ordered_vars: list[str] = []
    for var, level in FORECAST_VARS_ORDER_FASTNET:
        short_name = LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND[var]
        if level != 0:
            short_name += f"_{level}"
        forecast_ordered_vars.append(short_name)

    for var, level in NONFORECAST_VARS_ORDER_FASTNET:
        short_name = LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND[var]
        if level != 0:
            short_name += f"_{level}"
        nonforecast_ordered_vars.append(short_name)
    return forecast_ordered_vars, nonforecast_ordered_vars


class AnemoiERA5Dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        forecast_vars: list[str],
        nonforecast_vars: list[str],
        start: str | None = None,
        end: str | None = None,
        rollout_steps: int = 0,
    ) -> None:
        # will load data with feature dim order of:
        # forecast features, then nonforecast features
        ordered_vars = [*forecast_vars, *nonforecast_vars]
        self.ds = open_dataset(dataset_path, select=ordered_vars, start=start, end=end)
        if len(np.unique(np.diff(self.ds.dates))) > 1:
            msg = "Time periods in specified time range are not contiguous!"
            raise ValueError(msg)
        self.rollout_steps = rollout_steps
        self.mean = self.ds.statistics["mean"]
        self.std = self.ds.statistics["stdev"]
        # calculate actual number of data points based on rollout window
        ds_size = len(self.ds)
        if ds_size <= self.rollout_steps:
            msg = f"Dataset only has {ds_size} entries - impossible to roll out to {rollout_steps} lead times!"
            raise ValueError(msg)
        self.num_samples = len(self.ds) - self.rollout_steps

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        batch = self.ds[idx : idx + self.rollout_steps + 1]
        # t = time, v = variable, e = ensemble, g = grid
        # change dims from tveg to tgv (what FastNet expects)
        # (no ensemble, as deterministic model)
        # we'll be throwing away all the forecast features besides the init condition!
        batch = torch.from_numpy(batch).squeeze(2).permute(0, 2, 1)
        # normalize by stats (anemoi-datasets yaml recipe controls window for stats calc)
        normalized = ((batch - self.mean) / self.std).float()
        return normalized, idx


def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int, **kwargs) -> Dataloader:
    """Check to see if we're in a distributed / GPU context, and init accordingly."""
    sampler = None
    # check dist context
    if dist.is_available() and dist.is_initialized():
        sampler = dist.DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=False,
            drop_last=False,
        )
    return Dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
        **kwargs,
    )

