#!/usr/bin/env python3
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.data import adjusted_close_returns, download_or_load_nifty
from tvmvp_ext.improved_cvar import run_improved_cvar_analysis


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    prices, index_prices, _ = download_or_load_nifty(config)
    returns, index_returns = adjusted_close_returns(prices, index_prices)
    _, outputs = run_improved_cvar_analysis(returns, index_returns, config)
    print(outputs["improved_cvar_post_covid_performance"].to_string(index=False))
