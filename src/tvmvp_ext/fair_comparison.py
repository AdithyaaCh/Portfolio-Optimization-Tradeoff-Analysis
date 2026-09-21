from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .backtest import BacktestResult, run_backtest
from .inference import paired_strategy_inference
from .metrics import aggregate_monthly, expected_shortfall, performance_table


def _yearly_table(net: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows = []
    for year, frame in net.groupby(net.index.year):
        for model in frame:
            x = frame[model].dropna()
            rows.append({
                "Year": year,
                "Model": model,
                "Calendar return": float(np.prod(1 + x) - 1),
                "Annualised volatility": float(x.std(ddof=1) * np.sqrt(252)),
                "Sharpe ratio": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
                "Annualised ES 95%": expected_shortfall(-x, confidence) * np.sqrt(252),
            })
    return pd.DataFrame(rows)


def run_fair_constraint_comparison(returns: pd.DataFrame, config: dict) -> tuple[BacktestResult, dict[str, pd.DataFrame]]:
    """Compare MVP and CVaR under the same long-only 10%-cap feasible set."""
    root = Path(config["_extension_dir"]); tables = root / "results" / "tables"; plots = root / "results" / "plots"
    result = run_backtest(returns, config, comparison_mode="matched")
    annualisation = int(config["project"]["annualisation_days"])
    confidence = float(config["evaluation"]["realised_es_confidence"])
    full = performance_table(result.gross_returns, result.net_returns, result.turnover, annualisation, confidence).reset_index()
    start = pd.Timestamp(config["evaluation"]["post_covid_start"])
    post_gross = result.gross_returns.loc[result.gross_returns.index >= start]
    post_net = result.net_returns.loc[result.net_returns.index >= start]
    post_turnover = result.turnover.loc[result.turnover.index >= start]
    post = performance_table(post_gross, post_net, post_turnover, annualisation, confidence).reset_index()
    monthly = aggregate_monthly(post_net)
    primary = config["inference"]["matched_primary_comparisons"]
    inference = paired_strategy_inference(
        monthly,
        int(config["inference"]["bootstrap_draws"]),
        int(config["inference"]["monthly_block_length"]),
        float(config["inference"]["confidence"]),
        primary,
        int(config["project"]["seed"]) + 1201,
    )
    tail = result.tail_forecasts.copy()
    tail["Holding start"] = pd.to_datetime(tail["Holding start"])
    tail = tail[tail["Holding start"] >= start]
    calibration_rows = []
    for model, frame in tail.groupby("Model"):
        rho, _ = spearmanr(frame["Scenario CVaR 95%"], frame["Realised CVaR 95%"])
        calibration_rows.append({
            "Model": model,
            "Scenario CVaR 95%": frame["Scenario CVaR 95%"].mean(),
            "Realised next-month CVaR 95%": frame["Realised CVaR 95%"].mean(),
            "Realised-minus-scenario CVaR": (frame["Realised CVaR 95%"] - frame["Scenario CVaR 95%"] ).mean(),
            "VaR exceedance rate": frame["VaR exceedance rate"].mean(),
            "CVaR rank correlation": rho,
            "Effective assets": frame["Effective number of assets"].mean(),
            "Weights at cap": frame["Weights at 10% cap"].mean(),
            "Zero weights": frame["Zero weights"].mean(),
        })
    calibration = pd.DataFrame(calibration_rows)
    yearly = _yearly_table(post_net, confidence)
    outputs = {
        "matched_full_performance": full,
        "matched_post_covid_performance": post,
        "matched_post_covid_inference": inference,
        "matched_tail_calibration": calibration,
        "matched_yearly_performance": yearly,
        "matched_weights": result.weights,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    result.net_returns.to_csv(tables / "matched_daily_net_returns.csv", index_label="Date")
    result.turnover.to_csv(tables / "matched_daily_turnover.csv", index_label="Date")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    (1 + result.net_returns).cumprod().plot(ax=axes[0])
    (1 + post_net).cumprod().plot(ax=axes[1])
    axes[0].set(title="Matched constraints: full sample", ylabel="Net wealth", xlabel="")
    axes[1].set(title="Matched constraints: 2021 onward", ylabel="Net wealth", xlabel="")
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plots / "matched_constraint_returns.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return result, outputs
