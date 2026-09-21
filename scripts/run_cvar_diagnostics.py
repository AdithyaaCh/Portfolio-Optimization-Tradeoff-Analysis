#!/usr/bin/env python3
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.cvar_diagnostics import analyse_cvar_failure
from tvmvp_ext.empirical import run_empirical


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    result, _, _, _ = run_empirical(config)
    outputs = analyse_cvar_failure(result, config)
    print(outputs["cvar_tail_calibration"].to_string(index=False))
