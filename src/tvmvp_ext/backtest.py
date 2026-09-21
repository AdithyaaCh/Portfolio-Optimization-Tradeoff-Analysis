from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from . import MODELS
from .covariance import ewma_factor_covariance, nearest_positive_definite, sample_factor_covariance, sparse_residual_covariance
from .data import month_end_positions
from .factor import FactorFit, fit_local_factor_model, loading_space_angle_degrees
from .metrics import expected_shortfall
from .optimisation import (
    cvar_weights,
    equal_weights,
    factor_bootstrap_scenarios,
    filtered_factor_bootstrap_scenarios,
    minimum_variance_weights,
    regularized_cvar_weights,
)


@dataclass
class BacktestResult:
    gross_returns: pd.DataFrame
    net_returns: pd.DataFrame
    turnover: pd.DataFrame
    weights: pd.DataFrame
    covariance_losses: pd.DataFrame
    factor_diagnostics: pd.DataFrame
    tail_forecasts: pd.DataFrame
    final_factor_fit: FactorFit


def run_backtest(returns: pd.DataFrame, config: dict, comparison_mode: str = "core") -> BacktestResult:
    """Run the common rolling engine; matched modes use long-only capped weights."""
    est = config["estimation"]
    window = int(est["window_days"])
    positions = month_end_positions(returns.index, window)
    if len(positions) < 2:
        raise ValueError("Insufficient observations for at least two monthly holding periods")
    p = returns.shape[1]
    if comparison_mode == "core":
        models = list(MODELS)
    elif comparison_mode == "matched":
        models = [
            "Equal weight",
            "Original covariance MVP (matched)",
            "Dynamic covariance MVP (matched)",
            "Dynamic-factor CVaR (matched)",
        ]
    elif comparison_mode == "improved":
        models = [
            "Equal weight",
            "Original covariance MVP (matched)",
            "Dynamic covariance MVP (matched)",
            "Empirical CVaR (matched)",
            "Filtered-stressed CVaR (matched)",
            "Regularized filtered CVaR (matched)",
        ]
    else:
        raise ValueError("comparison_mode must be 'core', 'matched', or 'improved'")
    gross = pd.DataFrame(index=returns.index[positions[0] + 1 : positions[-1] + 1], columns=models, dtype=float)
    net = gross.copy()
    turnover = pd.DataFrame(0.0, index=gross.index, columns=models)
    weight_rows: list[dict] = []
    covariance_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    tail_rows: list[dict] = []
    pretrade: dict[str, np.ndarray | None] = {name: None for name in models}
    prior_loading = None
    base_seed = int(config["project"]["seed"])
    final_fit: FactorFit | None = None

    for rebalance_number, position in enumerate(positions[:-1]):
        next_position = positions[rebalance_number + 1]
        estimation = returns.iloc[position - window + 1 : position + 1]
        future = returns.iloc[position + 1 : next_position + 1]
        fit = fit_local_factor_model(estimation.to_numpy(), int(est["factor_number"]), est["bandwidth"])
        final_fit = fit
        residual_cov, residual_info = sparse_residual_covariance(
            fit.residuals, float(est["residual_threshold"]), float(est["eigenvalue_floor"])
        )
        factor_constant, constant_info = sample_factor_covariance(fit.scores, float(est["eigenvalue_floor"]))
        factor_dynamic, dynamic_info = ewma_factor_covariance(
            fit.scores, float(est["ewma_decay"]), float(est["eigenvalue_floor"])
        )
        original_cov = fit.endpoint_loading @ factor_constant @ fit.endpoint_loading.T + residual_cov
        dynamic_cov = fit.endpoint_loading @ factor_dynamic @ fit.endpoint_loading.T + residual_cov
        rng = np.random.default_rng(base_seed + rebalance_number)
        scenarios = factor_bootstrap_scenarios(
            fit.endpoint_loading,
            fit.scores,
            fit.residuals,
            int(config["cvar"]["scenarios"]),
            int(config["cvar"]["scenario_block_days"]),
            rng,
        )
        filtered_scenarios = None
        if comparison_mode == "improved":
            improved = config["cvar_improved"]
            residual_dynamic, _ = ewma_factor_covariance(
                fit.residuals, float(improved["residual_ewma_decay"]), float(est["eigenvalue_floor"])
            )
            residual_weight = float(improved["residual_dynamic_weight"])
            current_residual, _ = nearest_positive_definite(
                residual_weight * residual_dynamic + (1 - residual_weight) * residual_cov,
                float(est["eigenvalue_floor"]),
            )
            filtered_scenarios = filtered_factor_bootstrap_scenarios(
                fit.endpoint_loading,
                fit.scores,
                fit.residuals,
                factor_dynamic,
                current_residual,
                int(improved["scenarios"]),
                int(config["cvar"]["scenario_block_days"]),
                float(improved["recent_block_decay"]),
                float(improved["stress_fraction"]),
                float(improved["factor_stress_multiplier"]),
                float(improved["residual_stress_multiplier"]),
                np.random.default_rng(base_seed + 50000 + rebalance_number),
            )
        cvar_target = cvar_weights(
            scenarios,
            float(config["cvar"]["confidence"]),
            config["cvar"]["max_weight"] if config["cvar"]["long_only"] else None,
        )
        if comparison_mode == "core":
            targets = {
                "Equal weight": equal_weights(p),
                "Original TV-MVP": minimum_variance_weights(original_cov),
                "Dynamic SigmaF TV-MVP": minimum_variance_weights(dynamic_cov),
                "Dynamic-factor CVaR": cvar_target,
            }
            forecasts = {
                "Original TV-MVP": original_cov,
                "Dynamic SigmaF TV-MVP": dynamic_cov,
                "Dynamic-factor CVaR": np.cov(scenarios, rowvar=False, ddof=1),
            }
        elif comparison_mode == "matched":
            cap = float(config["cvar"]["max_weight"])
            targets = {
                "Equal weight": equal_weights(p),
                "Original covariance MVP (matched)": minimum_variance_weights(original_cov, long_only=True, max_weight=cap),
                "Dynamic covariance MVP (matched)": minimum_variance_weights(dynamic_cov, long_only=True, max_weight=cap),
                "Dynamic-factor CVaR (matched)": cvar_target,
            }
            forecasts = {
                "Original covariance MVP (matched)": original_cov,
                "Dynamic covariance MVP (matched)": dynamic_cov,
                "Dynamic-factor CVaR (matched)": np.cov(scenarios, rowvar=False, ddof=1),
            }
        else:
            if filtered_scenarios is None:
                raise RuntimeError("Improved scenario generation was not initialized")
            cap = float(config["cvar"]["max_weight"])
            filtered_name = "Filtered-stressed CVaR (matched)"
            regularized_name = "Regularized filtered CVaR (matched)"
            filtered_target = cvar_weights(filtered_scenarios, float(config["cvar"]["confidence"]), cap)
            previous_regularized = pretrade[regularized_name]
            if previous_regularized is None:
                previous_regularized = equal_weights(p)
            regularized_target = regularized_cvar_weights(
                filtered_scenarios,
                float(config["cvar"]["confidence"]),
                cap,
                previous_regularized,
                float(config["cvar_improved"]["turnover_penalty"]),
                float(config["cvar_improved"]["diversification_penalty"]),
            )
            targets = {
                "Equal weight": equal_weights(p),
                "Original covariance MVP (matched)": minimum_variance_weights(original_cov, long_only=True, max_weight=cap),
                "Dynamic covariance MVP (matched)": minimum_variance_weights(dynamic_cov, long_only=True, max_weight=cap),
                "Empirical CVaR (matched)": cvar_target,
                filtered_name: filtered_target,
                regularized_name: regularized_target,
            }
            filtered_covariance = np.cov(filtered_scenarios, rowvar=False, ddof=1)
            forecasts = {
                "Original covariance MVP (matched)": original_cov,
                "Dynamic covariance MVP (matched)": dynamic_cov,
                "Empirical CVaR (matched)": np.cov(scenarios, rowvar=False, ddof=1),
                filtered_name: filtered_covariance,
                regularized_name: filtered_covariance,
            }
        scenario_sets = {model: scenarios for model in targets}
        if comparison_mode == "improved" and filtered_scenarios is not None:
            # Evaluate all matched estimators on the same filtered/stressed distribution;
            # the empirical baseline keeps its own unfiltered scenario diagnostic.
            scenario_sets.update({model: filtered_scenarios for model in targets})
            scenario_sets["Empirical CVaR (matched)"] = scenarios
        tail_confidence = float(config["evaluation"].get("realised_es_confidence", 0.95))
        for model, weights in targets.items():
            scenario_returns = scenario_sets[model] @ weights
            realised_returns = future.to_numpy() @ weights
            scenario_losses = -scenario_returns
            realised_losses = -realised_returns
            scenario_var = float(np.quantile(scenario_losses, tail_confidence))
            tail_rows.append({
                "Rebalance date": returns.index[position],
                "Holding start": future.index[0],
                "Holding end": future.index[-1],
                "Model": model,
                "Scenario mean return": float(np.mean(scenario_returns)),
                "Scenario method": (
                    "paired empirical blocks"
                    if comparison_mode != "improved" or model == "Empirical CVaR (matched)"
                    else "filtered recent-weighted blocks with stress mixture"
                ),
                "Scenario VaR 95%": scenario_var,
                "Scenario CVaR 95%": expected_shortfall(scenario_losses, tail_confidence),
                "Realised mean return": float(np.mean(realised_returns)),
                "Realised VaR 95%": float(np.quantile(realised_losses, tail_confidence)),
                "Realised CVaR 95%": expected_shortfall(realised_losses, tail_confidence),
                "VaR exceedance rate": float(np.mean(realised_losses > scenario_var)),
                "Gross exposure": float(np.sum(np.abs(weights))),
                "Effective number of assets": float(1.0 / np.sum(weights ** 2)),
                "Weights at 10% cap": int(np.sum(np.isclose(weights, 0.10, atol=1e-5))),
                "Zero weights": int(np.sum(np.isclose(weights, 0.0, atol=1e-7))),
            })
        realised_cov = np.cov(future.to_numpy(), rowvar=False, ddof=1)
        for model, forecast in forecasts.items():
            weights = targets[model]
            covariance_rows.append({
                "Rebalance date": returns.index[position],
                "Model": model,
                "Frobenius loss": float(np.linalg.norm(realised_cov - forecast, ord="fro")),
                "Portfolio-risk loss": float(abs(weights @ realised_cov @ weights - weights @ forecast @ weights)),
            })
        score_kurtosis = kurtosis(fit.scores, axis=0, fisher=True, bias=False, nan_policy="omit")
        residual_kurtosis = kurtosis(fit.residuals, axis=0, fisher=True, bias=False, nan_policy="omit")
        diagnostic_rows.append({
            "Rebalance date": returns.index[position],
            "Variance explained": fit.explained_variance,
            "Maximum principal angle (degrees)": loading_space_angle_degrees(prior_loading, fit.endpoint_loading),
            "Residual-correlation sparsity": residual_info["off_diagonal_sparsity"],
            "Mean factor excess kurtosis": float(np.nanmean(score_kurtosis)),
            "Median residual excess kurtosis": float(np.nanmedian(residual_kurtosis)),
            "Residual covariance eigenvalues clipped": residual_info["eigenvalues_clipped"],
            "Factor covariance eigenvalues clipped (original)": constant_info["eigenvalues_clipped"],
            "Factor covariance eigenvalues clipped (dynamic)": dynamic_info["eigenvalues_clipped"],
            "EWMA effective observations": dynamic_info["effective_observations"],
        })
        prior_loading = fit.endpoint_loading

        for model, target in targets.items():
            turn = 0.0 if pretrade[model] is None else 0.5 * float(np.sum(np.abs(target - pretrade[model])))
            first_day = future.index[0]
            turnover.loc[first_day, model] = turn
            for asset, weight in zip(returns.columns, target):
                weight_rows.append({"Rebalance date": returns.index[position], "Model": model, "Asset": asset, "Weight": weight})
            held = target.copy()
            daily = future.to_numpy() @ target
            # Drift holdings each day; this is the pre-trade state at the next rebalance.
            for day_number, (day, asset_return) in enumerate(zip(future.index, future.to_numpy())):
                portfolio_return = float(held @ asset_return)
                gross.loc[day, model] = portfolio_return
                trading_cost = 0.0
                if day_number == 0:
                    trading_cost = float(config["evaluation"]["transaction_cost_bps"]) / 10000 * (2 * turn)
                net.loc[day, model] = portfolio_return - trading_cost
                denominator = 1 + portfolio_return
                held = held * (1 + asset_return) / denominator if denominator > 1e-12 else target.copy()
            pretrade[model] = held

    if final_fit is None:
        raise RuntimeError("Backtest produced no fits")
    return BacktestResult(
        gross.dropna(how="all"),
        net.dropna(how="all"),
        turnover.loc[gross.dropna(how="all").index],
        pd.DataFrame(weight_rows),
        pd.DataFrame(covariance_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(tail_rows),
        final_fit,
    )


def ex_ante_volatility_regime(index_returns: pd.Series, config: dict) -> pd.Series:
    evaluation = config["evaluation"]
    trailing = index_returns.rolling(int(evaluation["index_volatility_days"])).std(ddof=1) * np.sqrt(252)
    # Both the volatility signal and expanding threshold are shifted to prevent classification look-ahead.
    known_vol = trailing.shift(1)
    threshold = known_vol.expanding(int(evaluation["high_volatility_min_history"])).quantile(
        float(evaluation["high_volatility_quantile"])
    ).shift(1)
    return (known_vol > threshold).fillna(False).rename("High volatility")
