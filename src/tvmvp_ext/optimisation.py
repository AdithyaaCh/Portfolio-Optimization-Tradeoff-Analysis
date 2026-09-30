from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize


def equal_weights(assets: int) -> np.ndarray:
    return np.full(assets, 1.0 / assets)


def minimum_variance_weights(covariance: np.ndarray, long_only: bool = False, max_weight: float | None = None) -> np.ndarray:
    p = len(covariance)
    if not long_only and max_weight is None:
        raw = np.linalg.solve(covariance, np.ones(p))
        return raw / raw.sum()
    upper = 1.0 if max_weight is None else float(max_weight)
    result = minimize(
        lambda w: float(w @ covariance @ w),
        np.full(p, 1 / p),
        jac=lambda w: 2 * covariance @ w,
        bounds=[(0.0, upper)] * p,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(p)},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"MVP optimisation failed: {result.message}")
    return result.x


def turnover_regularized_minimum_variance_weights(
    covariance: np.ndarray,
    previous_weights: np.ndarray,
    penalty: float,
    max_weight: float,
) -> np.ndarray:
    """Long-only MVP with an L1 penalty on trades from pre-rebalance weights."""
    covariance = np.asarray(covariance, float)
    previous = np.asarray(previous_weights, float)
    p = len(previous)
    if penalty < 0:
        raise ValueError("Turnover penalty must be non-negative")
    if penalty == 0:
        return minimum_variance_weights(covariance, long_only=True, max_weight=max_weight)

    # Variables are portfolio weights followed by absolute-trade auxiliaries.
    initial_weights = np.clip(previous, 0, max_weight)
    initial_weights /= initial_weights.sum()
    initial = np.r_[initial_weights, np.abs(initial_weights - previous)]

    def objective(x: np.ndarray) -> float:
        weights, trades = x[:p], x[p:]
        return float(weights @ covariance @ weights + penalty * trades.sum())

    def gradient(x: np.ndarray) -> np.ndarray:
        return np.r_[2 * covariance @ x[:p], np.full(p, penalty)]

    # u >= w-w0 and u >= -(w-w0).
    constraints = [
        {"type": "eq", "fun": lambda x: x[:p].sum() - 1, "jac": lambda x: np.r_[np.ones(p), np.zeros(p)]},
        {"type": "ineq", "fun": lambda x: x[p:] - x[:p] + previous,
         "jac": lambda x: np.c_[-np.eye(p), np.eye(p)]},
        {"type": "ineq", "fun": lambda x: x[p:] + x[:p] - previous,
         "jac": lambda x: np.c_[np.eye(p), np.eye(p)]},
    ]
    result = minimize(
        objective,
        initial,
        jac=gradient,
        bounds=[(0.0, max_weight)] * p + [(0.0, None)] * p,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Turnover-regularized MVP optimisation failed: {result.message}")
    return result.x[:p]


def cvar_weights(scenarios: np.ndarray, confidence: float, max_weight: float | None) -> np.ndarray:
    """Rockafellar-Uryasev empirical loss-CVaR linear program."""
    scenarios = np.asarray(scenarios, float)
    draws, p = scenarios.shape
    objective = np.r_[np.zeros(p), 1.0, np.full(draws, 1.0 / ((1 - confidence) * draws))]
    # u_s >= -r_s'w - alpha is equivalent to -r_s'w - alpha - u_s <= 0.
    a_ub = np.c_[-scenarios, -np.ones(draws), -np.eye(draws)]
    b_ub = np.zeros(draws)
    a_eq = np.zeros((1, p + 1 + draws))
    a_eq[0, :p] = 1
    upper = 1.0 if max_weight is None else float(max_weight)
    bounds = [(0.0, upper)] * p + [(None, None)] + [(0.0, None)] * draws
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"CVaR optimisation failed: {result.message}")
    return result.x[:p]


def regularized_cvar_weights(
    scenarios: np.ndarray,
    confidence: float,
    max_weight: float,
    previous_weights: np.ndarray,
    turnover_penalty: float,
    diversification_penalty: float,
) -> np.ndarray:
    """CVaR LP with L1 turnover and distance-to-equal-weight regularisation."""
    scenarios = np.asarray(scenarios, float)
    previous = np.asarray(previous_weights, float)
    draws, p = scenarios.shape
    # Variables: weights, VaR threshold, tail slacks, turnover slacks, diversification slacks.
    objective = np.r_[
        np.zeros(p),
        1.0,
        np.full(draws, 1.0 / ((1 - confidence) * draws)),
        np.full(p, turnover_penalty),
        np.full(p, diversification_penalty),
    ]
    total = p + 1 + draws + p + p
    tail = np.zeros((draws, total)); tail[:, :p] = -scenarios; tail[:, p] = -1
    tail[:, p + 1 : p + 1 + draws] = -np.eye(draws)
    turn_positive = np.zeros((p, total)); turn_positive[:, :p] = np.eye(p)
    turn_positive[:, p + 1 + draws : p + 1 + draws + p] = -np.eye(p)
    turn_negative = np.zeros((p, total)); turn_negative[:, :p] = -np.eye(p)
    turn_negative[:, p + 1 + draws : p + 1 + draws + p] = -np.eye(p)
    equal = np.full(p, 1 / p)
    div_positive = np.zeros((p, total)); div_positive[:, :p] = np.eye(p)
    div_positive[:, -p:] = -np.eye(p)
    div_negative = np.zeros((p, total)); div_negative[:, :p] = -np.eye(p)
    div_negative[:, -p:] = -np.eye(p)
    a_ub = np.vstack([tail, turn_positive, turn_negative, div_positive, div_negative])
    b_ub = np.r_[np.zeros(draws), previous, -previous, equal, -equal]
    a_eq = np.zeros((1, total)); a_eq[0, :p] = 1
    bounds = [(0.0, max_weight)] * p + [(None, None)] + [(0.0, None)] * (draws + 2 * p)
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Regularized CVaR optimisation failed: {result.message}")
    return result.x[:p]


def factor_bootstrap_scenarios(
    loading: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    draws: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block draws of paired factor scores and residual vectors."""
    n = len(scores)
    chosen: list[int] = []
    while len(chosen) < draws:
        start = int(rng.integers(0, n))
        chosen.extend(((start + np.arange(block_length)) % n).tolist())
    indices = np.asarray(chosen[:draws])
    return scores[indices] @ loading.T + residuals[indices]


def _symmetric_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values = np.maximum(values, 1e-12)
    power = -0.5 if inverse else 0.5
    return (vectors * values ** power) @ vectors.T


def filtered_factor_bootstrap_scenarios(
    loading: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    current_factor_covariance: np.ndarray,
    current_residual_covariance: np.ndarray,
    draws: int,
    block_length: int,
    recent_decay: float,
    stress_fraction: float,
    factor_stress_multiplier: float,
    residual_stress_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Filtered historical simulation with current-state scaling and explicit stress draws."""
    n = len(scores)
    score_centered = scores - scores.mean(axis=0)
    residual_centered = residuals - residuals.mean(axis=0)
    score_cov = np.cov(score_centered, rowvar=False, ddof=1)
    residual_cov = np.cov(residual_centered, rowvar=False, ddof=1)
    score_innovations = score_centered @ _symmetric_sqrt(np.atleast_2d(score_cov), inverse=True)
    residual_innovations = residual_centered @ _symmetric_sqrt(np.atleast_2d(residual_cov), inverse=True)
    factor_root = _symmetric_sqrt(current_factor_covariance)
    residual_root = _symmetric_sqrt(current_residual_covariance)
    start_probabilities = recent_decay ** np.arange(n - 1, -1, -1, dtype=float)
    start_probabilities /= start_probabilities.sum()
    indices: list[int] = []
    while len(indices) < draws:
        start = int(rng.choice(n, p=start_probabilities))
        indices.extend(((start + np.arange(block_length)) % n).tolist())
    selected = np.asarray(indices[:draws])
    factor_draws = score_innovations[selected] @ factor_root
    residual_draws = residual_innovations[selected] @ residual_root
    stressed = rng.random(draws) < stress_fraction
    factor_draws[stressed] *= factor_stress_multiplier
    residual_draws[stressed] *= residual_stress_multiplier
    return factor_draws @ loading.T + residual_draws
