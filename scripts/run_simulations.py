#!/usr/bin/env python3
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.simulation import run_simulations


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    _, summary, _ = run_simulations(config, int(config["project"]["workers"]))
    print(summary.to_string(index=False))

