#!/usr/bin/env python3
"""Recompute descriptive study from frozen data and existing experiment outputs."""
from pathlib import Path

from tvmvp_ext.analytical_study import run_analytical_study
from tvmvp_ext.config import load_config

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = run_analytical_study(load_config(root / "config.yaml"))
    print(f"Wrote {len(results)} analytical tables, 3 figures and a hashed manifest to results/analysis")
