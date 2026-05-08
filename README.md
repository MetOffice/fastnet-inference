# FastNet Inference

This repository provides the end-to-end inference pipeline for the FastNet AI weather model as used in the corresponding research paper [FastNet: Improving the physical consistency of machine-learning weather prediction models through loss function design](doi).

- Downloads and loads a TorchScript checkpoint from the public [MetOffice/FastNet-global](https://huggingface.co/MetOffice/FastNet-global) repo on the Hugging Face Hub (cached locally after first download)
- Fetches data using anemoi
- Preprocesses data, including normalisation statistics computed over the full training period (1980-2020)
- Runs autoregressive rollout forecasts
- Handles output postprocessing and management
- Optional distributed execution support

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
uv run anemoi-datasets create era5-subset.yaml era5-subset.zarr
```

The recipe (`era5-subset.yaml`) pulls from ECMWF's public ERA5 dataset and subsets to your desired date range (currently limited up until the end of 2023):

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
model_repo: "MetOffice/FastNet-global"
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
| `model_repo` | Hugging Face repository ID to download model weights from |
| `zip` | Compress output to zip |

> [!NOTE]
> Two model files are available on Hugging Face: one optimised for GPU inference and one for CPU. The appropriate file is downloaded and loaded automatically based on the hardware available at runtime.


### 3. Run inference

```bash
uv run fastnet-inference --dataset-path era5-subset.zarr inference-config.yaml
```

### Output format

Writes **`output_path`** as Zarr: **`forecast`** `(init_time, lead_time, variable, grid)` plus **`latitude`/`longitude`** on **`grid`** (same ordering as the input dataset). Values are de-normalised with the loader **`mean`/`std`**.

For **FastNet-global** with the default forecast channel list, the **`variable`** axis has fixed size **72**. For **O96** input data, the **`grid`** axis has size **40320** (other grids follow `len(latitude)` on the input Zarr).

### Distributed (multi-GPU)

Use `torchrun` for multi-GPU inference:

```bash
uv run torchrun --nproc_per_node=8 -m fastnet_inference.cli run-pipeline inference-config.yaml --dataset-path era5-subset.zarr
```

Each GPU processes a contiguous chunk of init times and writes to non-overlapping regions of the output zarr store. The `--dataset-path` argument is provided for convenience for running on e.g. AzureML, where you may want to pass mounted paths to data from blob storage in a dynamic way.

## Model Weights License
The pretrained model weights associated with this repository are not covered by the repository's software license.
They are released under the Open Government Licence (OGL) v3.0 and are subject to British Crown copyright 2025, the Met Office.
The inference code in this repository is licensed separately under the GNU Affero General Public License (AGPL) v3.0. See [LICENSE](./LICENSE) for details.

## Contributors
The following people contributed to the development of FastNet, including model design, training, and evaluation:

Tom Dunstan; [Oliver Strickson](https://github.com/ots22); Thusal Bennett; Jack Bowyer; [Matthew Burnand](https://github.com/mo-mburnand); [James Chappell](https://github.com/jameschappell); [Alejandro Coca-Castro](https://github.com/acocac); Kirstine Ida Dale; Eric Daub; Noushin Eftekhari; [Manvendra Janmaijaya](https://github.com/manvendra9099); Jonathan Lillis; David Salvador-Jasin; [Nathan Simpson](https://github.com/phinate); Ryan Sze-Yin Chan; Mohamad Elmasri; Lydia Allegranza France; Sam Madge; [Sophie Luise Arana](https://github.com/aranas); Levan Bokeria; Hannah Brown; Evangeline Corcoran; Tom Dodds; Anna-Louise Ellis; Tomas Lazauskas; [David Llewellyn-Jones](https://github.com/llewelld); Theo McCaie; Sophia Moreton; Tom Potter; James Robinson; Adam Scaife; Iain Stenson; David Walters; Karina Bett-Williams; Louisa van Zeeland; Peter Yatsyshin; [J. Scott Hosking](https://github.com/scotthosking)
