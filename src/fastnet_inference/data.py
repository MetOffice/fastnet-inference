"""Define data-related functions and classes.

Note: the Dataset defined here loads rollout_steps data points for every timestep.
That means that there's rollout_steps - 1 redundant I/O calls that aren't cached for every lead time.
Improving this would be desirable in a high-throughput scenario.
(e.g. loading chunks of data and accessing those multiple times in an IterableDataset).
"""

import numpy as np
import torch
from anemoi.datasets import open_dataset
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from fastnet_inference.variables import (
    FORECAST_VARS_ORDER_FASTNET,
    LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND,
    NONFORECAST_VARS_ORDER_FASTNET,
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
        # TODO: load in downloaded stats
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


class AnemoiERA5IterableDataset(IterableDataset):
    """Implements contiguous loading of samples across ranks, avoiding redundant I/O reads for consecutive init times"""
    def __init__(
        self,
        dataset_path: str,
        forecast_vars: list[str],
        nonforecast_vars: list[str],
        start: str | None = None,
        end: str | None = None,
        rollout_steps: int = 0,
        rank: int = 0,
        world_size: int = 1,
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
        # distributed bookkeeping
        self.rank = rank
        self.world_size = world_size

    def _preprocess(self, batch: np.ndarray) -> torch.Tensor:
        # t = time, v = variable, e = ensemble, g = grid
        # change dims from tveg to tgv (what FastNet expects)
        # (no ensemble, as deterministic model)
        # we'll be throwing away all the forecast features besides the init condition!
        batch = torch.from_numpy(batch).squeeze(2).permute(0, 2, 1)
        # normalize by stats (anemoi-datasets yaml recipe controls window for stats calc)
        return ((batch - self.mean) / self.std).float()

    def __iter__(self):
        rank_chunk = self.num_samples // self.world_size
        # will have rank remainder & worker remainder — unsure how to process? Can we add them to the last rank?
        rank_remainder = self.num_samples % self.world_size
        rank_start = self.rank * rank_chunk
        rank_end = rank_start + rank_chunk
        if self.rank == self.world_size - 1:
            rank_end += rank_remainder
        rank_chunk = rank_end - rank_start  # could be bigger for the last rank!
        worker_info = get_worker_info()
        if worker_info is None:
            start, end = rank_start, rank_end
        else:
            worker_chunk = rank_chunk // worker_info.num_workers
            worker_remainder = rank_chunk % worker_info.num_workers
            start = rank_start + worker_info.id * worker_chunk
            end = start + worker_chunk
            if worker_info.id == worker_info.num_workers - 1 :
                end += worker_remainder
        # yield data and dates
        buffer = self.ds[start:start+self.rollout_steps+1]  # init time + rollout steps
        yield self._preprocess(buffer), start
 
        for i in range(start+1, end):
            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = self.ds[i+self.rollout_steps]
            yield self._preprocess(buffer), i

def create_dataloader(dataset: Dataset, batch_size: int, num_workers: int, **kwargs) -> DataLoader:
    """Check to see if we're in a distributed / GPU context, and init accordingly."""
    sampler = None
    # check dist context
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # partition dataset into chunks for each rank
        chunk_size = len(dataset) // world_size
        leftover_size = len(dataset) % world_size

        start, end = chunk_size

        sampler = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=False,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler,
        **kwargs,
    )

