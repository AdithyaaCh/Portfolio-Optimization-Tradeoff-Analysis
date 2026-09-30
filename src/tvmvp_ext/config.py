from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"project", "estimation", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    config["_config_path"] = str(path)
    config["_extension_dir"] = str(path.parent)
    return config


def with_overrides(config: dict, **overrides) -> dict:
    """Return a deep copy with dotted-key overrides."""
    result = copy.deepcopy(config)
    for dotted, value in overrides.items():
        target = result
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return result
