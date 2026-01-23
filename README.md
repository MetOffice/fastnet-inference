# FastNet Inference

Inference routines for the FastNet AI model for global weather prediction.

> [!IMPORTANT]
> This repo is designed to perform inference on O96 gridded data. Finer grids are not easily accessible; we may release a checkpoint to run FastNet on N320 data in future.

## Installation

```bash
uv sync
```

## Usage

### 1. Prepare input data

Generate an [anemoi-datasets](https://anemoi-datasets.readthedocs.io/) compatible O96 ERA5 dataset using the provided recipe:

```bash
anemoi-datasets create era5-subset.yaml era5-subset.zarr
```

The recipe (`era5-subset.yaml`) pulls from ECMWF's public ERA5 dataset and subsets to your desired date range:

```yaml
name: era5-subset
dates:
  start: '2022-01-01T00:00:00'
  end: '2022-12-31T23:00:00'
  frequency: 6h

input:
  pipe:
    - anemoi_dataset:
        dataset: https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr
    - z_to_orog:
        orography: orography
        geopotential: z
```

> [!NOTE]
> The produced dataset will have summary statistics, which are typically used when pre-processing inputs to ML models. Here, we actually package our own stats file for this to reflect what FastNet saw during training (1980-2020), so we ignore the statistics within the created dataset.

### 2. Configure inference

Create a config file (`inference-config.yaml`):

```yaml
start: "2022-01-01"
end: "2022-01-31"
batch_size: 32
num_workers: 8
rollout_steps: 40
output_path: "outputs/forecasts.zarr"
freq_hours: 6
zip: True
```

| Parameter | Description |
|-----------|-------------|
| `start` / `end` | Date range for inference |
| `batch_size` | Number of init times per batch |
| `num_workers` | DataLoader workers |
| `rollout_steps` | Forecast lead times (in multiples of `freq_hours`) |
| `output_path` | Output zarr store path |
| `freq_hours` | Time step frequency of rollout in hours (must match input data frequency) |
| `zip` | Compress output to zip |

### 3. Run inference

```bash
uv run fastnet-inference run-pipeline inference-config.yaml --dataset-path era5-subset.zarr
```

### Distributed (multi-GPU)

Use `torchrun` for multi-GPU inference:

```bash
uv run torchrun --nproc_per_node=8 -m fastnet_inference.cli run-pipeline inference-config.yaml --dataset-path era5-subset.zarr
```

Each GPU processes a contiguous chunk of init times and writes to non-overlapping regions of the output zarr store. The `--dataset-path` argument is provided for convenience for running on e.g. AzureML, where you may want to pass mounted paths to data from blob storage in a dynamic way.
