import pytest

from fastnet_inference import InferenceConfig, run_inference


@pytest.mark.requires_hf
def test_run_inference(synthetic_era5, tmp_path):
    config = InferenceConfig(
        dataset_path=str(synthetic_era5),
        batch_size=2,
        num_workers=0,
        rollout_steps=2,
        output_path=tmp_path / "forecasts.zarr",
        freq_hours=6,
    )
    run_inference(config)


@pytest.mark.requires_hf
def test_run_inference_multi_worker(synthetic_era5, tmp_path):
    config = InferenceConfig(
        dataset_path=str(synthetic_era5),
        batch_size=2,
        num_workers=8,
        rollout_steps=2,
        output_path=tmp_path / "forecasts.zarr",
        freq_hours=6,
    )
    run_inference(config)
