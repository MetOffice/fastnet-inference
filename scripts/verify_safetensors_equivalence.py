"""Verify the safetensors + FastNet nn.Module path reproduces the TorchScript model.

Loads BOTH the legacy TorchScript checkpoint (via ``torch.jit.load``; the only remaining
sanctioned use, in a trusted environment) and the converted safetensors checkpoint (via
``fastnet_inference.model.load_model``), then compares:

1. every weight/buffer tensor between the two models
2. a single forward pass on identical random input (batch size 2)
3. an autoregressive rollout (default 3 steps, batch size 1), mirroring
   ``fastnet_inference.eval._rollout_loop``

and reports max absolute / relative output differences. Both models use the same ops,
dtypes and kernels, so on CPU the expected difference is exactly 0.0. On CUDA the
scatter-add aggregation uses non-deterministic atomics, so small differences are expected
even between two runs of the same model; the script prints the TorchScript model's
self-consistency as a noise floor to compare against (use e.g. ``--device cuda --atol 1e-4``).

Usage:
    uv run python scripts/verify_safetensors_equivalence.py [--checkpoint model.safetensors]
        [--device cuda]
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from fastnet_inference.model import load_model

DEFAULT_REPO_ID = "MetOffice/FastNet-global"
TORCHSCRIPT_FILENAME = "model_file_cpu"
N_GRID = 40320  # O96 reduced Gaussian grid
N_FORECAST, N_NONFORECAST = 72, 12


def compare_weights(ts_model: torch.nn.Module, new_model: torch.nn.Module) -> float:
    ts_state, new_state = ts_model.state_dict(), new_model.state_dict()
    assert set(ts_state) == set(new_state), (
        f"state_dict key mismatch: only-torchscript={sorted(set(ts_state) - set(new_state))}, "
        f"only-new={sorted(set(new_state) - set(ts_state))}"
    )
    worst = 0.0
    for key, ts_tensor in ts_state.items():
        new_tensor = new_state[key]
        assert ts_tensor.dtype == new_tensor.dtype, f"{key}: dtype differs"
        assert ts_tensor.shape == new_tensor.shape, f"{key}: shape differs"
        if ts_tensor.dtype.is_floating_point:
            worst = max(worst, (ts_tensor - new_tensor).abs().max().item())
        else:
            assert torch.equal(ts_tensor, new_tensor), f"{key}: integer buffer differs"
    return worst


def report_diff(label: str, reference: torch.Tensor, candidate: torch.Tensor) -> float:
    abs_diff = (reference - candidate).abs()
    max_abs = abs_diff.max().item()
    denom = reference.abs().clamp_min(1e-12)
    max_rel = (abs_diff / denom).max().item()
    bitwise = " (bitwise identical)" if torch.equal(reference, candidate) else ""
    print(f"  {label}: max |diff| = {max_abs:.6e}, max rel diff = {max_rel:.6e}{bitwise}")
    return max_abs


@torch.inference_mode()
def rollout(
    model: torch.nn.Module, non_forecast_features: torch.Tensor, initial_state: torch.Tensor
) -> torch.Tensor:
    """Autoregressive rollout, mirroring fastnet_inference.eval._rollout_loop."""
    steps = non_forecast_features.shape[1]
    current_state = initial_state
    states = []
    for t in range(steps):
        current_state = model(non_forecast_features[:, t : t + 1], current_state).squeeze(1)
        states.append(current_state)
    return torch.stack(states, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent.parent / "model.safetensors",
        help="locally converted safetensors checkpoint",
    )
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--device", default="cpu", help="device to run the comparison on")
    parser.add_argument("--atol", type=float, default=1e-6, help="pass/fail threshold")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"Running on {torch.cuda.get_device_name(device)}")

    print(f"Loading legacy TorchScript model ({TORCHSCRIPT_FILENAME}) from {args.repo_id}...")
    ts_path = hf_hub_download(repo_id=args.repo_id, filename=TORCHSCRIPT_FILENAME)
    ts_model = torch.jit.load(ts_path, map_location=device)
    ts_model.eval()
    print(f"Loading FastNet nn.Module from {args.checkpoint}...")
    new_model = load_model(checkpoint_path=args.checkpoint, device=device)

    print("1) Comparing all weights and buffers...")
    worst = compare_weights(ts_model, new_model)
    print(f"  OK: keys/shapes/dtypes identical; max |weight diff| = {worst:.6e}")

    torch.manual_seed(670)
    failures = 0.0

    print("2) Single forward pass (batch=2)...")
    non_forecast = torch.randn(2, 1, N_GRID, N_NONFORECAST).to(device)
    state = torch.randn(2, N_GRID, N_FORECAST).to(device)
    with torch.inference_mode():
        ts_out = ts_model(non_forecast, state)
        ts_out_again = ts_model(non_forecast, state)
        new_out = new_model(non_forecast, state)
    assert ts_out.shape == new_out.shape, f"shape mismatch: {ts_out.shape} vs {new_out.shape}"
    report_diff("torchscript self-consistency (noise floor)", ts_out, ts_out_again)
    failures = max(failures, report_diff("forward", ts_out, new_out))

    print(f"3) Autoregressive rollout ({args.rollout_steps} steps, batch=1)...")
    non_forecast = torch.randn(1, args.rollout_steps, N_GRID, N_NONFORECAST).to(device)
    state = torch.randn(1, N_GRID, N_FORECAST).to(device)
    ts_traj = rollout(ts_model, non_forecast, state)
    new_traj = rollout(new_model, non_forecast, state)
    for t in range(args.rollout_steps):
        failures = max(failures, report_diff(f"step {t + 1}", ts_traj[:, t], new_traj[:, t]))

    if failures > args.atol:
        msg = f"FAIL: max output difference {failures:.6e} exceeds atol={args.atol:.1e}"
        raise SystemExit(msg)
    print(
        f"PASS: models are numerically equivalent (max output diff {failures:.6e} "
        f"<= atol={args.atol:.1e})"
    )


if __name__ == "__main__":
    main()
