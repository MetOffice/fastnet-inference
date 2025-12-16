import torch
import numpy
import logging
from jaxtyping import Float
from anemoi.datasets.data import Dataset
from fastnet_inference.data import load_anemoi_era5
from fastnet_inference.variables import FORECAST_VARS_ORDER_FASTNET 


F_LEN = len(FORECAST_VARS_ORDER_FASTNET)

def process_batch(batch: Float[numpy.ndarray, "b v t g"], ds: Dataset) -> Float[torch.Tensor, "b t g v"]:
    """b = batch, v = variable, t = time, g = grid""" 
    batch = torch.from_numpy(batch).permute(0, 2, 3, 1)
    batch = ((batch - ds.statistics['mean']) / ds.statistics['stdev']).float() 


def run_model():
    


    for batch in ds:
        # b = batch, v = variable, t = time, g = grid
        # change dims from bvtg to btgv (what FastNet expects)
        batch = torch.from_numpy(batch).permute(0, 2, 3, 1)
        batch = ((batch - ds.statistics['mean']) / ds.statistics['stdev']).float() 
        forecast_features, non_forecast_features = batch[..., :F_LEN], batch[..., F_LEN:]
        forecast_features = forecast_features.squeeze(1)
