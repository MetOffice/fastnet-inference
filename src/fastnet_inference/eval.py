import torch
import numpy
import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass
from fastnet_inference.data import AnemoiERA5Dataset 
from fastnet_inference.variables import FORECAST_VARS_ORDER_FASTNET 


F_LEN = len(FORECAST_VARS_ORDER_FASTNET)


class Config:
    dataset_path: str
    batch_size: int
    num_workers: int
    rollout_steps: int
    start_time: str | None = None
    end_time: str | None = None

def full_loop():
    config = Config()
    ds = AnemoiERA5Dataset(
        dataset_path=config.dataset_path,
        start=config.start_time,
        end=config.end_time,
        rollout_steps=config.rollout_steps,
    )

    def unnormalize(batch):
        return batch * ds.std + ds.mean

    dl = DataLoader(ds, batch_size=config.batch_size, num_workers=config.num_workers)

    for batch in dl:
        # FastNet expects forecast vars & forcings/constants as separate inputs
        forecast_features, non_forecast_features = batch[..., :F_LEN], batch[..., F_LEN:]
        # separate out what we need for rollout
        # grab initial conditions
        init_forecast_features = forecast_features[:, 0, :, :]

