import importlib
import json

import torch

from fastnet_inference.data import (
    GRAVITATIONAL_ACCELERATION,
    UNSTANDARDIZED_FEATURES,
    AnemoiERA5Dataset,
    get_fastnet_var_order,
)


def test_stats_var_order():
    base_path = importlib.resources.files("fastnet_inference")
    with (base_path / "stats.json").open() as f:
        a = json.load(f)
    check = list(a["mean"].keys())
    f, nf = get_fastnet_var_order()
    assert check == [*f, *nf], "Mismatch between stats vars and dataset vars!"


def test_orography_scaling_and_unstandardized_features(synthetic_era5):
    forecast_vars, nonforecast_vars = get_fastnet_var_order()
    ordered = [*forecast_vars, *nonforecast_vars]
    ds = AnemoiERA5Dataset(
        dataset_path=str(synthetic_era5),
        forecast_vars=forecast_vars,
        nonforecast_vars=nonforecast_vars,
        rollout_steps=0,
    )

    raw = ds.ds[0:1]
    raw = torch.from_numpy(raw).squeeze(2).permute(0, 2, 1)
    normalized, _ = ds[0]

    orography_idx = ordered.index("orography")
    expected_orog = (
        raw[..., orography_idx] * GRAVITATIONAL_ACCELERATION - ds.mean[orography_idx]
    ) / ds.std[orography_idx]
    assert torch.allclose(normalized[..., orography_idx], expected_orog.float())

    for var in UNSTANDARDIZED_FEATURES:
        idx = ordered.index(var)
        assert torch.allclose(normalized[..., idx], raw[..., idx].float())
