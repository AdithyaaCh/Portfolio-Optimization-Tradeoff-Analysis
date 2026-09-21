from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles


@dataclass
class FactorFit:
    endpoint_loading: np.ndarray
    scores: np.ndarray
    residuals: np.ndarray
    explained_variance: float
    loading_path: np.ndarray


def rule_of_thumb_bandwidth(n: int, p: int) -> float:
    return float((2.35 / np.sqrt(12.0)) * n ** (-0.2) * p ** (-0.1))


def _epanechnikov_weights(n: int, target: int, bandwidth: float) -> np.ndarray:
    grid = np.arange(n, dtype=float)
    u = (grid - target) / (n * bandwidth)
    kernel = 0.75 * (1.0 - u * u) * (np.abs(u) <= 1.0)
    # Discrete normalisation is the stable finite-sample analogue of the manual's boundary integral.
    total = kernel.sum()
    if total <= 0:
        raise ValueError("Kernel has no support; increase bandwidth")
    return kernel / total


def fit_local_factor_model(returns: np.ndarray, factors: int, bandwidth: float | str = "rule_of_thumb") -> FactorFit:
    """Local PCA at every date, followed by contemporaneous factor-score regressions."""
    x = np.asarray(returns, float)
    n, p = x.shape
    if factors < 1 or factors >= min(n, p):
        raise ValueError("Invalid number of factors")
    h = rule_of_thumb_bandwidth(n, p) if bandwidth == "rule_of_thumb" else float(bandwidth)
    loadings = np.empty((n, p, factors))
    explained = np.empty(n)
    previous = None
    for target in range(n):
        weights = _epanechnikov_weights(n, target, h)
        weighted = np.sqrt(weights)[:, None] * x
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        loading = vt[:factors].T * singular[:factors]
        if previous is not None:
            # PCA signs are arbitrary. Align signs only; the loading space is unchanged.
            signs = np.sign(np.sum(previous * loading, axis=0))
            signs[signs == 0] = 1
            loading *= signs
        loadings[target] = loading
        previous = loading
        explained[target] = float(np.sum(singular[:factors] ** 2) / max(np.sum(singular ** 2), 1e-16))
    scores = np.empty((n, factors))
    residuals = np.empty_like(x)
    for t in range(n):
        scores[t] = np.linalg.lstsq(loadings[t], x[t], rcond=None)[0]
        residuals[t] = x[t] - loadings[t] @ scores[t]
    return FactorFit(loadings[-1], scores, residuals, float(explained[-1]), loadings)


def loading_space_angle_degrees(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return np.nan
    return float(np.degrees(np.max(subspace_angles(previous, current))))


def select_factor_number(returns: np.ndarray, maximum: int, bandwidth: float | str = "rule_of_thumb") -> tuple[int, np.ndarray]:
    """Manual B.3-style information criterion using the portable local-PCA residual loss."""
    x = np.asarray(returns, float)
    n, p = x.shape
    h = rule_of_thumb_bandwidth(n, p) if bandwidth == "rule_of_thumb" else float(bandwidth)
    penalty = (p + n * h) / (p * n * h) * np.log((p * n * h) / (p + n * h))
    criteria = []
    for factors in range(1, maximum + 1):
        fit = fit_local_factor_model(x, factors, h)
        loss = max(float(np.mean(fit.residuals ** 2)), 1e-16)
        criteria.append(np.log(loss) + penalty * factors)
    values = np.asarray(criteria)
    return int(np.argmin(values) + 1), values
