import logging
from enum import StrEnum

import torch
from huggingface_hub import hf_hub_download

HF_ACCOUNT_NAME = "MetOffice"
logger = logging.getLogger(__name__)


class ModelFilename(StrEnum):
    cpu = "model_file_cpu"
    gpu = "model_file"


def load_model(version: float = 1.1, device: str | torch.device = "cpu") -> torch.nn.Module:
    # TODO: think about non-torchscript checkpoint
    if device.lower() == "cpu":
        model_filename = ModelFilename.cpu
    elif "cuda" in device.casefold():
        model_filename = ModelFilename.gpu
    else:
        msg = "Invalid device: expected 'cpu' or 'cuda'/'cuda:x' (integer x)"
        raise ValueError(msg)
    repo_name = f"{HF_ACCOUNT_NAME}/FastNet-v{version}"
    logger.info(
        "Downloading FastNet v%s... (will use cached model if already downloaded)",
        version,
    )
    model_path = hf_hub_download(repo_id=repo_name, filename=model_filename)
    logger.info("Model downloaded!")
    logger.info("Loading model on device %s...", device)
    model = torch.jit.load(model_path, map_location=device)
    logger.info("Model loaded on %s successfully.", device)
    return model
