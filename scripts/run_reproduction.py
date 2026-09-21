#!/usr/bin/env python3
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.reproduction import run_reproduction


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_reproduction(load_config(root / "config.yaml"))
    print(result.to_string(index=False))

