import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from fastnet_inference.network import FastNet

DEFAULT_MODEL_REPO_ID = "MetOffice/FastNet-global"
CHECKPOINT_FILENAME = "model.safetensors"
logger = logging.getLogger(__name__)


def load_model(
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
) -> torch.nn.Module:
    """Build the FastNet model and load its weights.

    Weights are stored as safetensors, which holds tensors only — loading executes no
    code. A single checkpoint file serves every device; pass ``device="cpu"``,
    ``"cuda"``/``"cuda:N"``, etc.

    Args:
        repo_id: Hugging Face repo to download ``model.safetensors`` from.
        checkpoint_path: load this local file instead of downloading.
    """
    if checkpoint_path is None:
        logger.info(
            "Downloading FastNet model from %s... (will use cached model if already downloaded)",
            repo_id,
        )
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=CHECKPOINT_FILENAME)
        logger.info("Model downloaded!")
    logger.info("Loading model on device %s...", device)
    model = FastNet()
    model.load_state_dict(load_file(checkpoint_path))
    model.to(device)
    model.eval().requires_grad_(False)
    logger.info("Model loaded on %s successfully.", device)
    return model
