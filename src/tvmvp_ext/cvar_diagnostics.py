from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .backtest import BacktestResult
from .inference import moving_block_indices
from .metrics import annualised_return, expected_shortfall, performance_table


def _yearly_metrics(returns: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows = []
    for year, frame in returns.groupby(returns.index.year):
        for model in frame:
            values = frame[model].dropna()
            rows.append({
                "Year": year,
                "Model": model,
                "Return": float(np.prod(1 + values) - 1),
                "Annualised volatility": float(values.std(ddof=1) * np.sqrt(252)),
                "Sharpe ratio": float(values.mean() / values.std(ddof=1) * np.sqrt(252)),
                "Annualised ES 95%": expected_shortfall(-values, confidence) * np.sqrt(252),
            })
    return pd.DataFrame(rows)


def _mechanism_bootstrap(tail: pd.DataFrame, config: dict) -> pd.DataFrame:
    pivot = tail.pivot(index="Rebalance date", columns="Model", values=["Scenario CVaR 95%", "Realised CVaR 95%", "Realised mean return"])
    comparisons = {
        "Scenario CVaR (D minus B)": pivot[("Scenario CVaR 95%", "Dynamic-factor CVaR")] - pivot[("Scenario CVaR 95%", "Original TV-MVP")],
        "Realised CVaR (D minus B)": pivot[("Realised CVaR 95%", "Dynamic-factor CVaR")] - pivot[("Realised CVaR 95%", "Original TV-MVP")],
        "Realised mean return (D minus B)": pivot[("Realised mean return", "Dynamic-factor CVaR")] - pivot[("Realised mean return", "Original TV-MVP")],
    }
    rng = np.random.default_rng(int(config["project"]["seed"]) + 911)
    draws = int(config["inference"]["bootstrap_draws"])
    block = int(config["inference"]["monthly_block_length"])
    confidence = float(config["inference"]["confidence"])
    rows = []
    for metric, series in comparisons.items():
        values = series.to_numpy()
        distribution = np.asarray([values[moving_block_indices(len(values), block, rng)].mean() for _ in range(draws)])
        lower, upper = np.quantile(distribution, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
        rows.append({
            "Exploratory mechanism": metric,
            "Mean difference": float(values.mean()),
            "CI lower": float(lower),
            "CI upper": float(upper),
            "Draws": draws,
            "Block length (months)": block,
        })
    return pd.DataFrame(rows)


def analyse_cvar_failure(result: BacktestResult, config: dict) -> dict[str, pd.DataFrame]:
    root = Path(config["_extension_dir"]); tables = root / "results" / "tables"; plots = root / "results" / "plots"
    start = pd.Timestamp(config["evaluation"]["post_covid_start"])
    gross = result.gross_returns.loc[result.gross_returns.index >= start]
    net = result.net_returns.loc[result.net_returns.index >= start]
    turnover = result.turnover.loc[result.turnover.index >= start]
    performance = performance_table(gross, net, turnover, int(config["project"]["annualisation_days"]), float(config["evaluation"]["realised_es_confidence"]))
    performance["Annualised cost drag"] = performance["Annualised return"] - performance["Annualised net return"]
    tail = result.tail_forecasts.copy()
    tail["Rebalance date"] = pd.to_datetime(tail["Rebalance date"])
    tail = tail[tail["Holding start"] >= start].copy()
    calibration_rows = []
    for model, frame in tail.groupby("Model"):
        rho, rho_p = spearmanr(frame["Scenario CVaR 95%"], frame["Realised CVaR 95%"])
        calibration_rows.append({
            "Model": model,
            "Mean scenario CVaR": frame["Scenario CVaR 95%"].mean(),
            "Mean realised next-month CVaR": frame["Realised CVaR 95%"].mean(),
            "Mean realised-minus-scenario CVaR": (frame["Realised CVaR 95%"] - frame["Scenario CVaR 95%"] ).mean(),
            "Mean VaR exceedance rate": frame["VaR exceedance rate"].mean(),
            "CVaR rank correlation": rho,
            "Rank-correlation p-value (descriptive)": rho_p,
        })
    calibration = pd.DataFrame(calibration_rows)
    concentration = tail.groupby("Model")[["Gross exposure", "Effective number of assets", "Weights at 10% cap", "Zero weights"]].agg(["mean", "std"])
    concentration.columns = [" ".join(c) for c in concentration.columns]
    concentration = concentration.reset_index()
    mechanisms = _mechanism_bootstrap(tail, config)
    yearly = _yearly_metrics(net, float(config["evaluation"]["realised_es_confidence"]))
    cvar_weights = result.weights[(result.weights["Model"] == "Dynamic-factor CVaR") & (result.weights["Rebalance date"] >= start)]
    allocation = cvar_weights.groupby("Asset")["Weight"].agg(["mean", "std", "max", lambda x: np.mean(np.isclose(x, 0.10, atol=1e-5)), lambda x: np.mean(np.isclose(x, 0.0, atol=1e-7))]).reset_index()
    allocation.columns = ["Asset", "Mean weight", "Weight std", "Maximum weight", "Cap frequency", "Zero frequency"]
    allocation = allocation.sort_values("Mean weight", ascending=False)
    factor_shift = result.factor_diagnostics.copy()
    factor_shift["Rebalance date"] = pd.to_datetime(factor_shift["Rebalance date"])
    factor_shift["Period"] = np.where(factor_shift["Rebalance date"] >= start, "Post-COVID", "Pre-2021")
    factor_shift = factor_shift.drop(columns="Rebalance date").groupby("Period").mean(numeric_only=True).reset_index()
    outputs = {
        "cvar_post_covid_performance": performance.reset_index(),
        "cvar_tail_calibration": calibration,
        "cvar_concentration": concentration,
        "cvar_mechanism_bootstrap": mechanisms,
        "cvar_yearly_performance": yearly,
        "cvar_asset_allocation": allocation,
        "cvar_factor_diagnostic_shift": factor_shift,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)

    d_tail = tail[tail["Model"] == "Dynamic-factor CVaR"].set_index("Rebalance date")
    relative_wealth = (1 + net["Dynamic-factor CVaR"]).cumprod() / (1 + net["Original TV-MVP"]).cumprod()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    relative_wealth.plot(ax=axes[0], color="#b22222")
    axes[0].axhline(1, color="black", linewidth=1, linestyle="--")
    axes[0].set(title="CVaR wealth relative to TV-MVP", ylabel="Relative wealth", xlabel="")
    axes[1].scatter(d_tail["Scenario CVaR 95%"], d_tail["Realised CVaR 95%"], alpha=0.65)
    limit = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1]); axes[1].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[1].set(title="CVaR forecast calibration", xlabel="Scenario CVaR", ylabel="Next-month realised CVaR")
    d_tail[["Effective number of assets", "Weights at 10% cap"]].plot(ax=axes[2])
    axes[2].set(title="CVaR concentration", ylabel="Count / effective count", xlabel="")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(plots / "cvar_post_covid_diagnostics.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return outputs
