from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .covariance import ewma_factor_covariance, sample_factor_covariance, sparse_residual_covariance
from .factor import fit_local_factor_model
from .metrics import annualised_return, expected_shortfall, maximum_drawdown
from .optimisation import cvar_weights, equal_weights, factor_bootstrap_scenarios, minimum_variance_weights


def _draw_multivariate(rng: np.random.Generator, covariance: np.ndarray, student_df: int | None) -> np.ndarray:
    z = np.linalg.cholesky(covariance) @ rng.normal(size=len(covariance))
    if student_df is None:
        return z
    return z / np.sqrt(rng.chisquare(student_df) / student_df) * np.sqrt((student_df - 2) / student_df)


def _simulate_one(args: tuple[str, int, dict]) -> list[dict]:
    setting, replication, sim = args
    rng = np.random.default_rng(int(sim["seed"]) + 10000 * ("gaussian", "student_t", "regime_shift").index(setting) + replication)
    p, k = int(sim["assets"]), int(sim["factor_number"])
    n, future_n = int(sim["estimation_days"]), int(sim["evaluation_days"])
    total = n + future_n
    base_loading = rng.normal(scale=0.35, size=(p, k))
    trend = rng.normal(scale=0.10, size=(p, k))
    returns = np.empty((total, p)); true_covariances = []
    for t in range(total):
        phase = t / max(total - 1, 1)
        loading = base_loading + trend * np.sin(2 * np.pi * phase)
        factor_scale = 0.00010 * (1.0 + 1.2 * phase)
        correlation = 0.20
        residual_scale = 0.00010
        if setting == "regime_shift" and t >= n - 20:
            factor_scale *= 4.0
            residual_scale *= 2.25
            correlation = 0.65
        factor_cov = factor_scale * np.array([[1.0, 0.25], [0.25, 0.8]])
        residual_cov = residual_scale * correlation ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
        df = int(sim["student_df"]) if setting == "student_t" else None
        factor = _draw_multivariate(rng, factor_cov, df)
        residual = _draw_multivariate(rng, residual_cov, df)
        returns[t] = loading @ factor + residual
        true_covariances.append(loading @ factor_cov @ loading.T + residual_cov)
    training, future = returns[:n], returns[n:]
    fit = fit_local_factor_model(training, k)
    residual_cov, _ = sparse_residual_covariance(fit.residuals, float(sim["residual_threshold"]), float(sim["eigenvalue_floor"]))
    constant_f, _ = sample_factor_covariance(fit.scores, float(sim["eigenvalue_floor"]))
    dynamic_f, _ = ewma_factor_covariance(fit.scores, float(sim["ewma_decay"]), float(sim["eigenvalue_floor"]))
    original_cov = fit.endpoint_loading @ constant_f @ fit.endpoint_loading.T + residual_cov
    dynamic_cov = fit.endpoint_loading @ dynamic_f @ fit.endpoint_loading.T + residual_cov
    scenarios = factor_bootstrap_scenarios(fit.endpoint_loading, fit.scores, fit.residuals, int(sim["scenarios"]), int(sim["scenario_block_days"]), rng)
    scenario_cov = np.cov(scenarios, rowvar=False)
    forecasts = {"Original TV-MVP": original_cov, "Dynamic SigmaF TV-MVP": dynamic_cov, "Dynamic-factor CVaR": scenario_cov}
    weights = {
        "Equal weight": equal_weights(p),
        "Original TV-MVP": minimum_variance_weights(original_cov),
        "Dynamic SigmaF TV-MVP": minimum_variance_weights(dynamic_cov),
        "Dynamic-factor CVaR": cvar_weights(scenarios, float(sim["cvar_confidence"]), float(sim["max_weight"])),
    }
    true_cov = np.mean(true_covariances[n:], axis=0)
    rows = []
    for model, weight in weights.items():
        portfolio = future @ weight
        net_portfolio = portfolio.copy()
        net_portfolio[0] -= float(sim["transaction_cost_bps"]) / 10000 * np.sum(np.abs(weight))
        volatility = float(np.std(portfolio, ddof=1) * np.sqrt(252))
        row = {
            "Setting": setting,
            "Replication": replication,
            "Model": model,
            "Realised variance": float(np.var(portfolio, ddof=1)),
            "Realised ES 95%": expected_shortfall(-portfolio, 0.95),
            "Mean return": float(np.mean(portfolio)),
            "Annualised volatility": volatility,
            "Annualised return": annualised_return(portfolio, 252),
            "Sharpe ratio": float(np.mean(portfolio) / np.std(portfolio, ddof=1) * np.sqrt(252)),
            "Maximum drawdown": maximum_drawdown(portfolio),
            "Annualised ES 95%": expected_shortfall(-portfolio, 0.95) * np.sqrt(252),
            "Initial traded notional": float(np.sum(np.abs(weight))),
            "Annualised net return": annualised_return(net_portfolio, 252),
            "True portfolio variance": float(weight @ true_cov @ weight),
            "True covariance Frobenius error": np.nan,
        }
        if model in forecasts:
            row["True covariance Frobenius error"] = float(np.linalg.norm(forecasts[model] - true_cov, ord="fro"))
        rows.append(row)
    return rows


def run_simulations(config: dict, workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(config["_extension_dir"]); table_dir = root / "results" / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    source = config["simulation"]
    sim = {
        **source,
        "seed": int(config["project"]["seed"]),
        "residual_threshold": float(config["estimation"]["residual_threshold"]),
        "eigenvalue_floor": float(config["estimation"]["eigenvalue_floor"]),
        "ewma_decay": float(config["estimation"]["ewma_decay"]),
        "scenario_block_days": int(config["cvar"]["scenario_block_days"]),
        "cvar_confidence": float(config["cvar"]["confidence"]),
        "max_weight": float(config["cvar"]["max_weight"]),
        "transaction_cost_bps": float(config["evaluation"]["transaction_cost_bps"]),
    }
    jobs = [(setting, rep, sim) for setting in ("gaussian", "student_t", "regime_shift") for rep in range(int(sim["replications"]))]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_simulate_one, jobs, chunksize=2))
    else:
        nested = [_simulate_one(job) for job in jobs]
    raw = pd.DataFrame([row for group in nested for row in group])
    raw.to_csv(table_dir / "simulation_replications.csv", index=False)
    metric_columns = [
        "Realised variance", "Realised ES 95%", "Mean return", "Annualised volatility",
        "Annualised return", "Sharpe ratio", "Maximum drawdown", "Annualised ES 95%",
        "Initial traded notional", "Annualised net return", "True portfolio variance",
        "True covariance Frobenius error",
    ]
    summary_rows = []
    for (setting, model), group in raw.groupby(["Setting", "Model"]):
        for metric in metric_columns:
            values = group[metric].dropna()
            if values.empty:
                continue
            se = values.std(ddof=1) / np.sqrt(len(values))
            summary_rows.append({"Setting": setting, "Model": model, "Metric": metric, "Mean": values.mean(), "Standard deviation": values.std(ddof=1), "MC CI lower": values.mean() - 1.96 * se, "MC CI upper": values.mean() + 1.96 * se, "Replications": len(values)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "simulation_summary.csv", index=False)
    probability_rows = []
    for setting, group in raw.groupby("Setting"):
        pivot = group.pivot(index="Replication", columns="Model", values=["Realised variance", "Realised ES 95%"])
        for model in ["Equal weight", "Dynamic SigmaF TV-MVP", "Dynamic-factor CVaR"]:
            for metric in ["Realised variance", "Realised ES 95%"]:
                probability_rows.append({"Setting": setting, "Model": model, "Metric": metric, "Probability beats original TV-MVP": float(np.mean(pivot[(metric, model)] < pivot[(metric, "Original TV-MVP")]))})
    probabilities = pd.DataFrame(probability_rows)
    probabilities.to_csv(table_dir / "simulation_beat_probabilities.csv", index=False)
    plot_dir = root / "results" / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    figure_data = raw.groupby(["Setting", "Model"])[["Realised variance", "Realised ES 95%"]].mean()
    for setting in figure_data.index.get_level_values("Setting").unique():
        baseline = figure_data.loc[(setting, "Original TV-MVP")]
        figure_data.loc[setting, :] = figure_data.loc[setting, :].to_numpy() / baseline.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, ax in zip(("Realised variance", "Realised ES 95%"), axes):
        figure_data[metric].unstack("Model").plot(kind="bar", ax=ax)
        ax.axhline(1, color="black", linewidth=1, linestyle="--")
        ax.set(title=f"{metric}: ratio to original", ylabel="Ratio", xlabel="")
        ax.tick_params(axis="x", rotation=15)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(plot_dir / "simulation_stress_summary.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    return raw, summary, probabilities
