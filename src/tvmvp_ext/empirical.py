from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, ex_ante_volatility_regime, run_backtest
from .data import adjusted_close_returns, download_or_load_nifty
from .inference import paired_strategy_inference, regime_benefit_test
from .metrics import aggregate_monthly, performance_table, regime_performance
from .plotting import (
    plot_cumulative,
    plot_factor_diagnostics,
    plot_qq,
    plot_rolling_risk,
    plot_sensitivity_heatmap,
    plot_weights_turnover,
)


def _write_core_outputs(result: BacktestResult, returns: pd.DataFrame, index_returns: pd.Series, config: dict) -> dict[str, pd.DataFrame]:
    root = Path(config["_extension_dir"]); tables = root / "results" / "tables"; plots = root / "results" / "plots"
    tables.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    evaluation = config["evaluation"]
    performance = performance_table(result.gross_returns, result.net_returns, result.turnover, int(config["project"]["annualisation_days"]), float(evaluation["realised_es_confidence"]))
    monthly = aggregate_monthly(result.net_returns)
    inference = paired_strategy_inference(monthly, int(config["inference"]["bootstrap_draws"]), int(config["inference"]["monthly_block_length"]), float(config["inference"]["confidence"]), config["inference"]["primary_comparisons"], int(config["project"]["seed"]) + 41)
    regime = ex_ante_volatility_regime(index_returns, config).reindex(result.net_returns.index).fillna(False)
    regime_metrics = regime_performance(result.net_returns, regime, int(config["project"]["annualisation_days"]), float(evaluation["realised_es_confidence"]))
    regime_test = regime_benefit_test(result.net_returns, regime, int(config["inference"]["bootstrap_draws"]), int(config["inference"]["regime_daily_block_length"]), float(config["inference"]["confidence"]), int(config["project"]["seed"]) + 73)
    covariance_summary = result.covariance_losses.groupby("Model")[["Frobenius loss", "Portfolio-risk loss"]].agg(["mean", "std", "median"])
    covariance_summary.columns = [" ".join(column) for column in covariance_summary.columns]
    weight_summary = result.weights.groupby("Model")["Weight"].agg(["mean", "std", "min", "max", lambda x: x.abs().median()]).rename(columns={"<lambda_0>": "median_abs"})
    diagnostics_summary = result.factor_diagnostics.drop(columns="Rebalance date").agg(["mean", "std", "min", "max"]).T
    outputs = {
        "empirical_performance": performance,
        "monthly_net_returns": monthly,
        "pairwise_bootstrap_inference": inference,
        "regime_performance": regime_metrics,
        "regime_benefit_test": regime_test,
        "covariance_forecast_summary": covariance_summary,
        "covariance_forecast_by_rebalance": result.covariance_losses,
        "factor_diagnostics": result.factor_diagnostics,
        "factor_diagnostics_summary": diagnostics_summary,
        "weight_summary": weight_summary,
        "tail_forecast_by_rebalance": result.tail_forecasts,
    }
    for name, frame in outputs.items():
        destination = tables / f"{name}.csv"
        if name == "monthly_net_returns":
            frame.to_csv(destination, index_label="Month")
        elif name in {"empirical_performance", "covariance_forecast_summary", "factor_diagnostics_summary", "weight_summary"}:
            frame.reset_index().to_csv(destination, index=False)
        else:
            frame.to_csv(destination, index=False)
    result.gross_returns.to_csv(tables / "daily_gross_returns.csv", index_label="Date")
    result.net_returns.to_csv(tables / "daily_net_returns.csv", index_label="Date")
    result.turnover.to_csv(tables / "daily_turnover.csv", index_label="Date")
    result.weights.to_csv(tables / "weights_by_rebalance.csv", index=False)
    regime.rename("High volatility").to_csv(tables / "ex_ante_volatility_regime.csv", header=True, index_label="Date")
    plot_cumulative(result.net_returns, plots / "cumulative_net_returns.png")
    plot_rolling_risk(result.net_returns, int(evaluation["rolling_days"]), float(evaluation["realised_es_confidence"]), plots / "rolling_volatility_es.png")
    plot_weights_turnover(result.weights, result.turnover, plots / "weights_turnover_distribution.png")
    plot_factor_diagnostics(result.factor_diagnostics, plots / "factor_diagnostics.png")
    plot_qq(result.final_factor_fit.scores, result.final_factor_fit.residuals, plots / "factor_residual_qq.png")
    return outputs


def run_empirical(config: dict, force_download: bool = False) -> tuple[BacktestResult, dict[str, pd.DataFrame], pd.DataFrame, pd.Series]:
    prices, index_prices, _ = download_or_load_nifty(config, force_download)
    returns, index_returns = adjusted_close_returns(prices, index_prices)
    result = run_backtest(returns, config)
    outputs = _write_core_outputs(result, returns, index_returns, config)
    return result, outputs, returns, index_returns


def _net_at_cost(result: BacktestResult, bps: float) -> pd.DataFrame:
    net = result.gross_returns.copy()
    for model in net.columns:
        net[model] -= (bps / 10000) * 2 * result.turnover[model]
    return net


def run_sensitivity(base_result: BacktestResult, returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    root = Path(config["_extension_dir"]); tables = root / "results" / "tables"; plots = root / "results" / "plots"
    start = pd.Timestamp(config["sensitivity"]["evaluation_start"])
    cache: dict[tuple[str, str], BacktestResult] = {}
    rows = []

    def summarise(parameter: str, value: object, result: BacktestResult, bps: float | None = None) -> None:
        cost = float(config["evaluation"]["transaction_cost_bps"] if bps is None else bps)
        gross = result.gross_returns.loc[result.gross_returns.index >= start]
        turnover = result.turnover.loc[gross.index]
        net = _net_at_cost(result, cost).loc[gross.index]
        metrics = performance_table(gross, net, turnover, int(config["project"]["annualisation_days"]), float(config["evaluation"]["realised_es_confidence"]))
        original = metrics.loc["Original TV-MVP"]
        dynamic = metrics.loc["Dynamic SigmaF TV-MVP"]
        cvar = metrics.loc["Dynamic-factor CVaR"]
        rows.append({
            "Parameter": parameter,
            "Value": str(value),
            "Dynamic-minus-original Sharpe": dynamic["Sharpe ratio"] - original["Sharpe ratio"],
            "Dynamic-minus-original volatility": dynamic["Annualised volatility"] - original["Annualised volatility"],
            "Dynamic-minus-original net return": dynamic["Annualised net return"] - original["Annualised net return"],
            "CVaR-minus-original Sharpe": cvar["Sharpe ratio"] - original["Sharpe ratio"],
            "CVaR-minus-original ES": cvar["Realised ES 95% (annualised)"] - original["Realised ES 95% (annualised)"],
            "Observations": len(gross),
        })

    for parameter, config_key in (("EWMA decay", "ewma_decay"), ("Window days", "window_days"), ("Factor number", "factor_number"), ("CVaR confidence", "confidence")):
        section = "cvar" if parameter == "CVaR confidence" else "estimation"
        values = config["sensitivity"][{"EWMA decay": "ewma_decay", "Window days": "window_days", "Factor number": "factor_number", "CVaR confidence": "cvar_confidence"}[parameter]]
        for value in values:
            base_value = config[section][config_key]
            if value == base_value:
                result = base_result
            else:
                modified = copy.deepcopy(config); modified[section][config_key] = value
                key = (parameter, str(value))
                result = cache.setdefault(key, run_backtest(returns, modified))
            summarise(parameter, value, result)
    for bps in config["sensitivity"]["transaction_cost_bps"]:
        summarise("Transaction cost (bps)", bps, base_result, float(bps))
    table = pd.DataFrame(rows)
    table.to_csv(tables / "sensitivity_summary.csv", index=False)
    plot_sensitivity_heatmap(table, plots / "sensitivity_heatmap.png")
    return table
