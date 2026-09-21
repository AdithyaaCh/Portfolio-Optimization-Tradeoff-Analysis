import numpy as np
import pandas as pd

from tvmvp_ext.backtest import ex_ante_volatility_regime, run_backtest
from tvmvp_ext.covariance import ewma_factor_covariance, nearest_positive_definite
from tvmvp_ext.inference import moving_block_indices
from tvmvp_ext.optimisation import (
    cvar_weights,
    filtered_factor_bootstrap_scenarios,
    minimum_variance_weights,
    regularized_cvar_weights,
)


def test_positive_definite_repair():
    fixed, info = nearest_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]), 1e-6)
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert info["eigenvalues_clipped"] == 1


def test_ewma_emphasises_recent_variance():
    scores = np.r_[np.ones((90, 1)), np.arange(10, dtype=float).reshape(-1, 1)]
    ewma, _ = ewma_factor_covariance(scores, 0.8, 1e-9)
    assert ewma[0, 0] > np.var(scores, ddof=1)


def test_portfolio_constraints():
    covariance = np.array([[0.02, 0.005], [0.005, 0.01]])
    weights = minimum_variance_weights(covariance)
    assert np.isclose(weights.sum(), 1)
    scenarios = np.random.default_rng(2).normal(size=(200, 5))
    cvar = cvar_weights(scenarios, 0.95, 0.30)
    assert np.isclose(cvar.sum(), 1)
    assert cvar.min() >= -1e-9 and cvar.max() <= 0.30 + 1e-9


def test_regularized_cvar_constraints_and_reproducible_filtered_scenarios():
    rng = np.random.default_rng(13)
    scores = rng.normal(size=(120, 2))
    residuals = rng.normal(scale=0.5, size=(120, 6))
    loading = rng.normal(scale=0.2, size=(6, 2))
    kwargs = dict(
        loading=loading,
        scores=scores,
        residuals=residuals,
        current_factor_covariance=np.diag([2.0, 0.5]),
        current_residual_covariance=np.eye(6) * 0.3,
        draws=200,
        block_length=5,
        recent_decay=0.99,
        stress_fraction=0.15,
        factor_stress_multiplier=1.5,
        residual_stress_multiplier=1.1,
    )
    scenarios_a = filtered_factor_bootstrap_scenarios(**kwargs, rng=np.random.default_rng(8))
    scenarios_b = filtered_factor_bootstrap_scenarios(**kwargs, rng=np.random.default_rng(8))
    assert np.array_equal(scenarios_a, scenarios_b)
    weights = regularized_cvar_weights(scenarios_a, 0.95, 0.25, np.full(6, 1 / 6), 0.001, 0.00025)
    assert np.isclose(weights.sum(), 1)
    assert weights.min() >= -1e-9 and weights.max() <= 0.25 + 1e-9


def test_moving_block_reproducible_and_contiguous():
    a = moving_block_indices(20, 4, np.random.default_rng(7))
    b = moving_block_indices(20, 4, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert all((a[i + 1] - a[i]) % 20 == 1 for i in range(20) if i % 4 != 3 and i < 19)


def test_regime_uses_lagged_information():
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    values = pd.Series(np.r_[np.zeros(199), 1.0], index=dates)
    config = {"evaluation": {"index_volatility_days": 5, "high_volatility_min_history": 20, "high_volatility_quantile": 0.75}}
    regime = ex_ante_volatility_regime(values, config)
    assert not regime.iloc[-1]


def test_future_returns_do_not_change_earlier_weights():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    returns = pd.DataFrame(rng.normal(scale=0.01, size=(120, 5)), index=dates)
    altered = returns.copy()
    altered.iloc[-10:] += 0.25
    config = {
        "project": {"seed": 3},
        "estimation": {"window_days": 40, "factor_number": 2, "bandwidth": "rule_of_thumb", "residual_threshold": 0.15, "eigenvalue_floor": 1e-7, "ewma_decay": 0.94},
        "cvar": {"scenarios": 40, "scenario_block_days": 4, "confidence": 0.95, "long_only": True, "max_weight": 0.40},
        "evaluation": {"transaction_cost_bps": 10},
    }
    original_weights = run_backtest(returns, config).weights
    altered_weights = run_backtest(altered, config).weights
    cutoff = altered.index[-10]
    left = original_weights[original_weights["Rebalance date"] < cutoff].reset_index(drop=True)
    right = altered_weights[altered_weights["Rebalance date"] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    matched = run_backtest(returns, config, comparison_mode="matched").weights
    optimised = matched[matched["Model"] != "Equal weight"]
    assert optimised["Weight"].min() >= -1e-8
    assert optimised["Weight"].max() <= 0.40 + 1e-8
    sums = optimised.groupby(["Rebalance date", "Model"])["Weight"].sum()
    assert np.allclose(sums, 1.0)
