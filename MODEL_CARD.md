# Model Card for FastNet-v1.1

**FastNet** is an data-driven medium range numerical weather prediction model developed jointly by the UK Met Office and the Alan Turing Institute. This release of FastNet v.1.1 marks the first publicly shared experimental release of the FastNet project.

FastNet produces highly skilled forecasts that overcome commonly known limitations of AI models, resulting in more physically realistic forecasts as demonstrated in the corresponding publication xxx.

## Model Details

### Model Description

FastNet has an encode-process-decode structure with a series of graph neural networks and auto-regressive rollout. The encoder is a directional bipartite graph linking the current atmospheric state defined on input grid cells to a lower resolution latent space defined on mesh nodes. The processor then advances the mesh state in time by six hour increments. The processor operates on a multi-scale icosahedral mesh, starting from the 12-node icosahedron and subdividing six times, enabling the model to capture both localised and long-range interactions. Finally, the decoder maps the latent mesh representation back to the output domain, and the prediction is fed back as input for subsequent steps during rollout. FastNet uses a residual formulation, where the decoder output represents the increment to be added to the input state via skip-level connections, rather than predicting the full field from scratch. Notably, FastNet was trained with loss-function adaptations designed to improve physical realism compared to similar models. 

- **Developed by:** Met Office & The Alan Turing Institute
- **Model type:** Encoder-processor-decoder model
- **License:** These model weights are published under a Creative Commons Attribution 4.0 International (CC BY 4.0). To view a copy of this licence, visit [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/). The corresponding inference scripts are released under the under the GNU Affero General Public License (AGPL) v3.0.

### Model Sources

- **Inference-only repository:** https://github.com/MetOffice/fastnet-inference
- **Paper:** _add-doi-here_

## Uses

### Direct Use
This model is intended for research and exploratory inference on historical or real-time atmospheric reanalysis inputs to produce global weather pattern predictions over a time horizon of up to ~2 days.
It is released for inference only: weights are provided for forward prediction, but training and fine-tuning are not supported in this release.
    
Typical direct uses include: benchmarking against baselines, sensitivity experiments (e.g., perturbing input fields), case-study analysis of notable events

### Out of scope
This release is not intended for operational forecasting, safety-critical decision making or issuing public warnings

## Known Limitations

## Training Details

### Training Data
FastNet is trained on the Copernicus ERA5 reanalysis dataset produced by ECMWF. This dataset is also available via [anemoi](https://anemoi.readthedocs.io/projects/training/en/latest/user-guide/download-era5-o96.html)
We use data re-gridded to an O96 (~ grid spacing of 104km) reduced Gaussian grid from 13 pressure levels taken at 6 hour time snapshots. 

The full list of input and output fields is shown below:

|Variable|Level|Input/Output|
|---|---|---|
|Geopotential|Pressure levels (hPa): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000|Both|
|Horizontal wind (zonal & meridional components)|Pressure levels (hPa): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000|Both|
|Specific humidity|Pressure levels (hPa): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000|Both|
|Temperature|Pressure levels (hPa): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000|Both|
|Surface pressure|Surface|Both|
|Mean sea-level pressure|Surface|Both|
|Skin temperature|Surface|Both|
|2m temperature|Surface|Both|
|2m dewpoint temperature|Surface|Both|
|10m horizontal wind (zonal & meridional components)|Surface|Both|
|Land-sea mask|Surface|Input-only|
|Orography|Surface|Input-only|
|Standard deviation of sub-grid orography|Surface|Input-only|
|Slope of sub-scale orography|Surface|Input-only|
|Top-of-atmosphere solar radiation|Surface|Input-only|
|Solar hour angle (cos & sin)|Surface|Input-only|
|Time of year (cos & sin)|Surface|Input-only|
|Latitude (cos & sin)|Surface|Input-only|
|Longitude (cos & sin)|Surface|Input-only|
### Training Procedure

Below we summarize the three-stage training procedure, including the number of update steps and the loss/objective used at each stage.

| Stage         | # rollout   | Loss / Objective |
| ------------- | ----------- | ---------------- |
| Pre-training  | n = 1 (6h)  | weighted MSE     |
| Fine-tuning 1 | n = 7 (42h) | MSE              |
| Fine-tuning 2 | n = 7 (42h) | spectral MSE     |
Weighting was applied per-variable and was proportional to pressure level.
#### Preprocessing
Training data from each pressure level (including surface-level variables) are separately standardised to have zero mean and unit standard deviation, over latitude, longitude and time. Orography is rescaled to the unit interval, and land-sea mask, solar hour angle, time of year, latitude, and longitude require no change as they are already suitably normalised.

### Speed, Sizes, Times

## Evaluation

FastNet is evaluated against ERA5 data using the WeatherBench 2 software package analysis for 2022. We compute the full rollout for all forecast valid times in 2022, re-gridding the output to a 1.5 degree latitude-longitude grid using a conservative re-gridding scheme. 

We also evaluate against the Met Office Global Model operational 

## Technical Specifications

### Hardware

Experiments were run on Microsoft Azure using `<GPU/CPU info>`

### Software

Code was implemented in Python using `<framework>` and executed on Azure; dependencies include `<key libs>` (see `requirements.txt` / `environment.yml`).

