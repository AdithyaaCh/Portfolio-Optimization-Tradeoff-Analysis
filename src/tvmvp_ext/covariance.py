from __future__ import annotations

import numpy as np


def nearest_positive_definite(matrix: np.ndarray, floor: float = 1e-7) -> tuple[np.ndarray, dict]:
    """Symmetrise and clip small eigenvalues; return an auditable repair record."""
    sym = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2
    values, vectors = np.linalg.eigh(sym)
    scale = max(float(np.max(np.abs(values))), 1.0)
    absolute_floor = floor * scale
    clipped = np.maximum(values, absolute_floor)
    fixed = (vectors * clipped) @ vectors.T
    fixed = (fixed + fixed.T) / 2
    return fixed, {
        "min_eigenvalue_before": float(values.min()),
        "min_eigenvalue_after": float(clipped.min()),
        "eigenvalues_clipped": int(np.sum(values < absolute_floor)),
    }


def sparse_residual_covariance(residuals: np.ndarray, threshold_scale: float, floor: float) -> tuple[np.ndarray, dict]:
    """Soft-threshold residual correlations while preserving marginal variances."""
    sample = np.cov(residuals, rowvar=False, ddof=1)
    sd = np.sqrt(np.maximum(np.diag(sample), floor))
    corr = sample / np.outer(sd, sd)
    p, n = residuals.shape[1], residuals.shape[0]
    cutoff = threshold_scale * np.sqrt(np.log(max(p, 2)) / max(n, 2))
    off = corr - np.eye(p)
    sparse_off = np.sign(off) * np.maximum(np.abs(off) - cutoff, 0.0)
    sparse = (sparse_off + np.eye(p)) * np.outer(sd, sd)
    fixed, repair = nearest_positive_definite(sparse, floor)
    off_mask = ~np.eye(p, dtype=bool)
    repair.update({
        "correlation_cutoff": float(cutoff),
        "off_diagonal_sparsity": float(np.mean(sparse_off[off_mask] == 0)),
    })
    return fixed, repair


def sample_factor_covariance(scores: np.ndarray, floor: float) -> tuple[np.ndarray, dict]:
    cov = np.cov(scores, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    return nearest_positive_definite(cov, floor)


def ewma_factor_covariance(scores: np.ndarray, decay: float, floor: float) -> tuple[np.ndarray, dict]:
    """Normalised EWMA covariance; recent estimated scores receive the largest weight."""
    if not 0 < decay < 1:
        raise ValueError("EWMA decay must be in (0, 1)")
    n = len(scores)
    weights = decay ** np.arange(n - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    mean = weights @ scores
    centered = scores - mean
    # The effective-degrees correction avoids the downward bias of normalised weights.
    correction = max(1.0 - float(weights @ weights), 1e-12)
    cov = (centered * weights[:, None]).T @ centered / correction
    fixed, repair = nearest_positive_definite(cov, floor)
    repair["effective_observations"] = float(1.0 / (weights @ weights))
    return fixed, repair

