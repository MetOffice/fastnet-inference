import json
from fastnet_inference.data import get_fastnet_var_order
import importlib


def test_stats_var_order():
    base_path = importlib.resources.files("fastnet_inference")
    with (base_path / 'stats.json').open() as f:
        a = json.load(f)
    check = list(a["mean"].keys())
    f, nf = get_fastnet_var_order()
    assert check == [*f, *nf], "Mismatch between stats vars and dataset vars!"
