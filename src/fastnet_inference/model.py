import torch
from enum import StrEnum
from huggingface_hub import hf_hub_download


HF_ACCOUNT_NAME = "phinate"


class ModelFilename(StrEnum):
    cpu = "model_file_cpu"
    gpu = "model_file"


def load_model(version: float = 1.1, device: str = "cpu") -> torch.nn.Module:
    # TODO: think about non-torchscript checkpoint
    if device.lower() == "cpu":
        model_filename = ModelFilename.cpu
    elif "cuda" in device.casefold():
        model_filename = ModelFilename.gpu
    else:
        msg = "Invalid device: expected 'cpu' or 'cuda'/'cuda:x' (integer x)"
        raise ValueError(msg)
    repo_name = f"{HF_ACCOUNT_NAME}/FastNet-v{version}"
    model_path = hf_hub_download(repo_id=repo_name, filename=model_filename)
    return torch.jit.load(model_path, map_location=device)
