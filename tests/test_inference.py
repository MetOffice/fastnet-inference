from pathlib import Path
from fastnet_inference import run_inference, InferenceConfig


def test_run_inference(synthetic_era5):
    config = InferenceConfig(
        dataset_path=synthetic_era5,
        inference_model_path="model_file_cpu",
        batch_size=2,
        num_workers=0,
        rollout_steps=2,
        output_path=Path("forecasts.zarr"),
        freq_hours=6,
    )
    run_inference(config)
