from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset

from fastnet_inference.variables import FORECAST_VARS_ORDER_FASTNET, NONFORECAST_VARS_ORDER_FASTNET, LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND


def load_anemoi_era5(dataset_name: str, select: list[str] | None = None) -> Dataset:
    ordered_vars: list[str] = []
    for var, level in FORECAST_VARS_ORDER_FASTNET:
        if select is not None and var not in select: continue
        short_name = LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND[var]
        if level != 0:
            short_name += f"_{level}"
        ordered_vars.append(short_name)

    for var, level in NONFORECAST_VARS_ORDER_FASTNET:
        if select is not None and var not in select: continue
        short_name = LONGHAND_VARIABLE_TO_ANEMOI_ERA5_SHORTHAND[var]
        if level != 0:
            short_name += f"_{level}"
        ordered_vars.append(short_name)

    return open_dataset(dataset_name, select=ordered_vars)
