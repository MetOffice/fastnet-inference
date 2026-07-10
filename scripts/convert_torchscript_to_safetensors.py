"""One-off conversion of the FastNet TorchScript checkpoints to a single safetensors file.

Why: the HF-hosted checkpoints (``model_file``, ``model_file_cpu``) are TorchScript
archives, which bundle pickled executable code with the weights (flagged as Protect AI
PAIT-TCHST-301). safetensors stores raw tensors only — no code execution on load.

This script is NOT part of the fastnet-inference package. It is the single sanctioned
place where ``torch.jit.load`` (and therefore pickle execution) still happens; run it in
a trusted environment against the official MetOffice/FastNet-global files.

What it does:
1. downloads and loads BOTH checkpoints via ``torch.jit.load``
2. extracts ``state_dict()`` plus any non-persistent ``named_buffers()`` from each
3. verifies the two checkpoints carry identical tensors (they are expected to be
   variants of the same weights) — prints a full diff and aborts if not
4. writes a single ``model.safetensors`` (contiguous tensors, provenance metadata)

Usage:
    uv run python scripts/convert_torchscript_to_safetensors.py [--output model.safetensors]
"""

import argparse
import hashlib
import platform
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

DEFAULT_REPO_ID = "MetOffice/FastNet-global"
GPU_FILENAME = "model_file"
CPU_FILENAME = "model_file_cpu"


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def extract_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a TorchScript archive and pull out every parameter and buffer.

    ``state_dict()`` covers parameters + persistent buffers; the ``named_buffers()``
    union additionally catches non-persistent buffers (which a plain state_dict would
    silently drop — e.g. graph edge indices, if they had been registered that way).
    """
    module = torch.jit.load(path, map_location="cpu")
    tensors = {name: t.detach() for name, t in module.state_dict().items()}
    extra = 0
    for name, buf in module.named_buffers():
        if name not in tensors:
            tensors[name] = buf.detach()
            extra += 1
    n_params = sum(1 for _ in module.named_parameters())
    print(
        f"  extracted {len(tensors)} tensors "
        f"({n_params} parameters, {len(tensors) - n_params - extra} persistent buffers, "
        f"{extra} non-persistent buffers)"
    )
    return tensors


def diff_tensor_dicts(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], a_name: str, b_name: str
) -> list[str]:
    """Compare two tensor dicts; return a list of human-readable mismatch descriptions."""
    problems: list[str] = []
    for key in sorted(set(a) - set(b)):
        problems.append(f"only in {a_name}: {key} {tuple(a[key].shape)} {a[key].dtype}")
    for key in sorted(set(b) - set(a)):
        problems.append(f"only in {b_name}: {key} {tuple(b[key].shape)} {b[key].dtype}")
    for key in sorted(set(a) & set(b)):
        ta, tb = a[key], b[key]
        if ta.shape != tb.shape:
            problems.append(f"shape mismatch: {key}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        if ta.dtype != tb.dtype:
            # precision variants: still report the value discrepancy after upcasting
            max_diff = (ta.double() - tb.double()).abs().max().item()
            problems.append(
                f"dtype mismatch: {key}: {ta.dtype} vs {tb.dtype} "
                f"(max |diff| after upcast: {max_diff:.3e})"
            )
            continue
        if not torch.equal(ta, tb):
            diff = (ta.double() - tb.double()).abs()
            problems.append(
                f"value mismatch: {key}: max |diff| = {diff.max().item():.6e}, "
                f"mean |diff| = {diff.mean().item():.6e}"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "model.safetensors",
        help="where to write the converted checkpoint (default: repo root)",
    )
    args = parser.parse_args()

    print(f"Downloading TorchScript checkpoints from {args.repo_id}...")
    gpu_path = hf_hub_download(repo_id=args.repo_id, filename=GPU_FILENAME)
    cpu_path = hf_hub_download(repo_id=args.repo_id, filename=CPU_FILENAME)
    gpu_sha, cpu_sha = sha256_of(gpu_path), sha256_of(cpu_path)
    print(f"  {GPU_FILENAME}:     sha256={gpu_sha}")
    print(f"  {CPU_FILENAME}: sha256={cpu_sha}")
    if gpu_sha == cpu_sha:
        print("  note: the two checkpoint files are byte-identical")

    print(f"Loading {GPU_FILENAME} (torch.jit.load)...")
    gpu_tensors = extract_tensors(gpu_path)
    print(f"Loading {CPU_FILENAME} (torch.jit.load)...")
    cpu_tensors = extract_tensors(cpu_path)

    print("Comparing GPU vs CPU tensors...")
    problems = diff_tensor_dicts(gpu_tensors, cpu_tensors, GPU_FILENAME, CPU_FILENAME)
    if problems:
        print(
            f"MISMATCH: the two checkpoints do NOT contain identical weights "
            f"({len(problems)} differences):"
        )
        for p in problems:
            print(f"  - {p}")
        msg = "GPU and CPU checkpoints differ; refusing to convert. See diff above."
        raise SystemExit(msg)
    print(f"  OK: all {len(cpu_tensors)} tensors are identical (keys, shapes, dtypes, values)")

    # Save the CPU-extracted copy (== GPU copy, just asserted) as safetensors.
    # clone() breaks any storage sharing/views; contiguous() is required by safetensors.
    out_tensors = {name: t.clone().contiguous() for name, t in cpu_tensors.items()}
    n_params = sum(t.numel() for t in out_tensors.values())
    metadata = {
        "format": "pt",
        "source_repo": args.repo_id,
        "source_files": f"{GPU_FILENAME},{CPU_FILENAME}",
        "source_sha256": cpu_sha,
        "converted_by": Path(__file__).name,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "description": (
            "FastNet-global v1.1 weights extracted from the TorchScript checkpoint. "
            "Load with fastnet_inference.model.load_model (safetensors + FastNet nn.Module)."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, args.output, metadata=metadata)

    size_mb = args.output.stat().st_size / 1e6
    print(
        f"Wrote {args.output} ({size_mb:.1f} MB, {len(out_tensors)} tensors, "
        f"{n_params:,} total elements)"
    )
    print(f"  sha256={sha256_of(args.output)}")


if __name__ == "__main__":
    main()
