import torch
from huggingface_hub import hf_hub_download


REPO_NAME = "phinate/FastNet-v1.1"
MODEL_FILENAME = "model_file_cpu" if torch.device == "cpu" else "model_file"


def load_model(device: str) -> torch.nn.Module:
    # TODO: think about non-torchscript checkpoint
    model_path = hf_hub_download(repo_id=REPO_NAME, filename=MODEL_FILENAME)
    return torch.jit.load(model_path, map_location=device)
