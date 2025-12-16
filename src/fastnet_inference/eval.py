import torch
import logging
from jaxtyping import Float
from fastnet_inference.data import load_anemoi_era5


def process_batch(batch: Float[torch.Tensor, "b v t g"]) -> Float[torch.Tensor, "b t g v"]:

