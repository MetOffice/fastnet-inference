import logging
from enum import StrEnum

import torch
from huggingface_hub import hf_hub_download

DEFAULT_MODEL_REPO_ID = "MetOffice/FastNet-global"
DEFAULT_MODEL_REPO_REVISION = "v1.1"
logger = logging.getLogger(__name__)


class ModelFilename(StrEnum):
    cpu = "model_file_cpu"
    gpu = "model_file"


def load_model(
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: str = DEFAULT_MODEL_REPO_REVISION,
    device: str | torch.device = "cpu"
) -> torch.nn.Module:
    # TODO: think about non-torchscript checkpoint
    if device.lower() == "cpu":
        model_filename = ModelFilename.cpu
    elif "cuda" in device.casefold():
        model_filename = ModelFilename.gpu
    else:
        msg = "Invalid device: expected 'cpu' or 'cuda'/'cuda:x' (integer x)"
        raise ValueError(msg)
    logger.info(
        "Downloading FastNet model from %s... (will use cached model if already downloaded)",
        repo_id,
    )
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        revision=revision,
    )
    logger.info("Model downloaded!")
    logger.info("Loading model on device %s...", device)
    model = torch.jit.load(model_path, map_location=device)
    logger.info("Model loaded on %s successfully.", device)
    return model
