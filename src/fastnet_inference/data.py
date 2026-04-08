"""Define data-related functions and classes.

Note: the Dataset defined here loads rollout_steps data points for every timestep.
That means that there's rollout_steps - 1 redundant I/O calls that aren't cached for every lead time.
Improving this would be desirable in a high-throughput scenario.
(e.g. loading chunks of data and accessing those multiple times in an IterableDataset).
"""

import importlib
import json

import numpy as np
import torch
from anemoi.datasets import open_dataset
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset

from fastnet_inference.sampler import ContiguousDistributedSampler
from fastnet_inference.variables import (
    FORECAST_VARS_ORDER_FASTNET,
    LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND,
    NONFORECAST_VARS_ORDER_FASTNET,
)

GRAVITATIONAL_ACCELERATION = 9.80665
UNSTANDARDIZED_FEATURES = {"cos_latitude", "cos_longitude", "sin_longitude"}


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
        """
        Parameters
        ----------
        dataset_path : str
            Path to the Anemoi dataset (local or remote).
        forecast_vars : list[str]
            List of forecast variable names to load (predicted by the model).
        nonforecast_vars : list[str]
            List of non-forecast variable names (forcings/constants).
        start : str | None
            Start date for data selection (ISO format). None for dataset start.
        end : str | None
            End date for data selection (ISO format). None for dataset end.
        rollout_steps : int
            Number of autoregressive steps to roll out. Determines window size.
        """
        # will load data with feature dim order of:
        # forecast features, then nonforecast features
        ordered_vars = [*forecast_vars, *nonforecast_vars]
        self.ordered_vars = ordered_vars
        self.ds = open_dataset(dataset_path, select=ordered_vars, start=start, end=end)
        if len(np.unique(np.diff(self.ds.dates))) > 1:
            msg = "Time periods in specified time range are not contiguous!"
            raise ValueError(msg)
        self.rollout_steps = rollout_steps
        # import our own stats
        base_path = importlib.resources.files("fastnet_inference")
        with (base_path / "stats.json").open() as f:
            stats = json.load(f)
        self.mean = np.array(list(stats["mean"].values()))
        self.std = np.array(list(stats["stdev"].values()))
        self.orography_idx = ordered_vars.index("orography")
        self.unstandardized_indices = [ordered_vars.index(v) for v in UNSTANDARDIZED_FEATURES]
        # calculate actual number of data points based on rollout window
        ds_size = len(self.ds)
        if ds_size <= self.rollout_steps:
            msg = f"Dataset only has {ds_size} entries - impossible to roll out to {rollout_steps} lead times!"
            raise ValueError(msg)
        # last init time is self.rollout_steps away from end
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
        # Align orography units with FastNet expectations before standardization.
        batch[..., self.orography_idx] *= GRAVITATIONAL_ACCELERATION
        # normalize by stats (anemoi-datasets yaml recipe controls window for stats calc)
        normalized = ((batch - self.mean) / self.std).float()
        # Explicitly bypass standardization for geometry features.
        normalized[..., self.unstandardized_indices] = batch[..., self.unstandardized_indices]
        return normalized, idx


def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int, **kwargs) -> DataLoader:
    """Check to see if we're in a distributed / GPU context, and init accordingly."""
    sampler = None
    # check dist context
    if dist.is_available() and dist.is_initialized():
        sampler = ContiguousDistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
        **kwargs,
    )
