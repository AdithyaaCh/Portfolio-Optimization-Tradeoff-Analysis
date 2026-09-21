#!/usr/bin/env python3
import argparse
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.cvar_diagnostics import analyse_cvar_failure
from tvmvp_ext.empirical import run_empirical, run_sensitivity
from tvmvp_ext.fair_comparison import run_fair_constraint_comparison
from tvmvp_ext.improved_cvar import run_improved_cvar_analysis
from tvmvp_ext.manifest import write_run_manifest
from tvmvp_ext.reproduction import run_reproduction
from tvmvp_ext.simulation import run_simulations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    print("Running author-example reproduction...")
    run_reproduction(config)
    print("Running NIFTY empirical study...")
    result, _, returns, _ = run_empirical(config, args.force_download)
    print("Diagnosing post-COVID CVaR performance...")
    analyse_cvar_failure(result, config)
    print("Running matched-constraint MVP-versus-CVaR comparison...")
    run_fair_constraint_comparison(returns, config)
    print("Running pre-specified filtered/stressed CVaR ablation...")
    run_improved_cvar_analysis(returns, index_returns, config)
    if not args.skip_sensitivity:
        print("Running robustness sweeps...")
        run_sensitivity(result, returns, config)
    print("Running Monte Carlo stress tests...")
    run_simulations(config, int(config["project"]["workers"]))
    print(f"Run manifest: {write_run_manifest(config)}")
