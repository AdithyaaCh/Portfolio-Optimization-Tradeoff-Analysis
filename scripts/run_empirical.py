#!/usr/bin/env python3
import argparse
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.empirical import run_empirical, run_sensitivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--with-sensitivity", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    result, outputs, returns, _ = run_empirical(config, args.force_download)
    if args.with_sensitivity:
        run_sensitivity(result, returns, config)
    print(outputs["empirical_performance"].to_string())

