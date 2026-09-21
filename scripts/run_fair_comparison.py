#!/usr/bin/env python3
from pathlib import Path

from tvmvp_ext.config import load_config
from tvmvp_ext.data import adjusted_close_returns, download_or_load_nifty
from tvmvp_ext.fair_comparison import run_fair_constraint_comparison


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    prices, index_prices, _ = download_or_load_nifty(config)
    returns, _ = adjusted_close_returns(prices, index_prices)
    _, outputs = run_fair_constraint_comparison(returns, config)
    print(outputs["matched_post_covid_performance"].to_string(index=False))
