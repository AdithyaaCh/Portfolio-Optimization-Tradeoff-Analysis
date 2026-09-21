from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .backtest import BacktestResult, ex_ante_volatility_regime, run_backtest
from .inference import pair_regime_interaction, paired_strategy_inference
from .metrics import aggregate_monthly, expected_shortfall, performance_table, regime_performance


def _calibration_table(tail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in tail.groupby("Model"):
        correlation = spearmanr(frame["Scenario CVaR 95%"], frame["Realised CVaR 95%"]).statistic
        rows.append({
            "Model": model,
            "Scenario method": frame["Scenario method"].iloc[0],
            "Mean scenario CVaR 95%": frame["Scenario CVaR 95%"].mean(),
            "Mean realised next-month CVaR 95%": frame["Realised CVaR 95%"].mean(),
            "Realised-minus-scenario CVaR": (frame["Realised CVaR 95%"] - frame["Scenario CVaR 95%"] ).mean(),
            "VaR exceedance rate": frame["VaR exceedance rate"].mean(),
            "CVaR rank correlation": correlation,
            "Effective assets": frame["Effective number of assets"].mean(),
            "Weights at cap": frame["Weights at 10% cap"].mean(),
            "Zero weights": frame["Zero weights"].mean(),
        })
    return pd.DataFrame(rows)


def _yearly_table(net: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows = []
    for year, frame in net.groupby(net.index.year):
        for model in frame:
            values = frame[model].dropna()
            rows.append({
                "Year": year,
                "Model": model,
                "Calendar return": float(np.prod(1 + values) - 1),
                "Annualised volatility": float(values.std(ddof=1) * np.sqrt(252)),
                "Annualised ES 95%": expected_shortfall(-values, confidence) * np.sqrt(252),
            })
    return pd.DataFrame(rows)


def _plot_diagnostics(result: BacktestResult, start: pd.Timestamp, window: int, destination: Path) -> None:
    short_names = {
        "Equal weight": "Equal weight",
        "Original covariance MVP (matched)": r"MVP: constant $\Sigma_F$",
        "Dynamic covariance MVP (matched)": r"MVP: EWMA $\Sigma_{F,t}$",
        "Empirical CVaR (matched)": "CVaR: empirical",
        "Filtered-stressed CVaR (matched)": "CVaR: filtered + stress",
        "Regularized filtered CVaR (matched)": "CVaR: filtered + regularised",
    }
    palette = {
        "Equal weight": "#4C78A8",
        r"MVP: constant $\Sigma_F$": "#F58518",
        r"MVP: EWMA $\Sigma_{F,t}$": "#54A24B",
        "CVaR: empirical": "#E45756",
        "CVaR: filtered + stress": "#B279A2",
        "CVaR: filtered + regularised": "#7F574C",
    }
    net = result.net_returns.loc[result.net_returns.index >= start].rename(columns=short_names)
    tail = result.tail_forecasts.copy()
    tail["Holding start"] = pd.to_datetime(tail["Holding start"])
    tail = tail[tail["Holding start"] >= start]
    tail["Model"] = tail["Model"].replace(short_names)
    effective = tail.pivot(index="Holding start", columns="Model", values="Effective number of assets")
    gap = tail.assign(gap=tail["Realised CVaR 95%"] - tail["Scenario CVaR 95%"])
    gap = gap.pivot(index="Holding start", columns="Model", values="gap")
    colors = [palette[column] for column in net.columns]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.9804))
    (1 + net).cumprod().plot(ax=axes[0, 0], legend=False, color=colors, linewidth=1.8)
    (net.rolling(window).std(ddof=1) * np.sqrt(252)).plot(ax=axes[0, 1], legend=False, color=colors, linewidth=1.6)
    effective = effective.reindex(columns=net.columns)
    gap = gap.reindex(columns=net.columns)
    effective.plot(ax=axes[1, 0], legend=False, color=colors, linewidth=1.6)
    gap.plot(ax=axes[1, 1], legend=False, color=colors, linewidth=1.6)
    axes[0, 0].set(title="Post-2021 net wealth", ylabel="Wealth", xlabel="")
    axes[0, 1].set(title=f"{window}-day realised volatility", ylabel="Annualised", xlabel="")
    axes[1, 0].set(title="Effective number of assets", ylabel="1 / sum(w²)", xlabel="")
    axes[1, 1].set(title="Tail calibration gap", ylabel="Realised minus scenario CVaR", xlabel="")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=10,
        frameon=True,
        columnspacing=1.8,
        handlelength=2.8,
    )
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.0, w_pad=1.5)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_improved_cvar_analysis(
    returns: pd.DataFrame,
    index_returns: pd.Series,
    config: dict,
) -> tuple[BacktestResult, dict[str, pd.DataFrame]]:
    """Run the pre-specified filtered/stressed and regularized CVaR ablation."""
    root = Path(config["_extension_dir"])
    tables = root / "results" / "tables"; plots = root / "results" / "plots"
    tables.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    result = run_backtest(returns, config, comparison_mode="improved")
    start = pd.Timestamp(config["evaluation"]["post_covid_start"])
    confidence = float(config["evaluation"]["realised_es_confidence"])
    annualisation = int(config["project"]["annualisation_days"])
    post_gross = result.gross_returns.loc[result.gross_returns.index >= start]
    post_net = result.net_returns.loc[result.net_returns.index >= start]
    post_turnover = result.turnover.loc[result.turnover.index >= start]
    performance = performance_table(post_gross, post_net, post_turnover, annualisation, confidence).reset_index()
    monthly = aggregate_monthly(post_net)
    inference = paired_strategy_inference(
        monthly,
        int(config["inference"]["bootstrap_draws"]),
        int(config["inference"]["monthly_block_length"]),
        float(config["inference"]["confidence"]),
        config["inference"]["improved_cvar_primary_comparisons"],
        int(config["project"]["seed"]) + 1601,
    )
    tail = result.tail_forecasts.copy()
    tail["Holding start"] = pd.to_datetime(tail["Holding start"])
    post_tail = tail[tail["Holding start"] >= start]
    calibration = _calibration_table(post_tail)
    regime = ex_ante_volatility_regime(index_returns, config).reindex(post_net.index).fillna(False)
    regime_metrics = regime_performance(post_net, regime, annualisation, confidence)
    regime_test = pair_regime_interaction(
        post_net,
        regime,
        config["inference"]["improved_cvar_regime_comparison"],
        int(config["inference"]["bootstrap_draws"]),
        int(config["inference"]["regime_daily_block_length"]),
        float(config["inference"]["confidence"]),
        int(config["project"]["seed"]) + 1661,
    )
    weights = result.weights.copy()
    weights["Rebalance date"] = pd.to_datetime(weights["Rebalance date"])
    post_weights = weights[weights["Rebalance date"] >= start]
    concentration = post_weights.groupby(["Rebalance date", "Model"])["Weight"].agg(
        Effective_assets=lambda x: 1 / np.sum(np.asarray(x) ** 2),
        Maximum_weight="max",
        Zero_weights=lambda x: np.sum(np.isclose(x, 0, atol=1e-7)),
    ).reset_index()
    outputs = {
        "improved_cvar_post_covid_performance": performance,
        "improved_cvar_post_covid_inference": inference,
        "improved_cvar_tail_calibration": calibration,
        "improved_cvar_regime_performance": regime_metrics,
        "improved_cvar_regime_interaction": regime_test,
        "improved_cvar_yearly_performance": _yearly_table(post_net, confidence),
        "improved_cvar_concentration": concentration,
        "improved_cvar_tail_by_rebalance": post_tail,
        "improved_cvar_weights": post_weights,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    post_net.to_csv(tables / "improved_cvar_daily_net_returns.csv", index_label="Date")
    post_turnover.to_csv(tables / "improved_cvar_daily_turnover.csv", index_label="Date")
    _plot_diagnostics(result, start, int(config["evaluation"]["rolling_days"]), plots / "improved_cvar_diagnostics.png")
    return result, outputs
