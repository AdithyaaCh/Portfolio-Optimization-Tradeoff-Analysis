from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .covariance import sample_factor_covariance, sparse_residual_covariance
from .data import load_author_example
from .factor import fit_local_factor_model, select_factor_number
from .optimisation import equal_weights, minimum_variance_weights


def author_dgp(p: int, n: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy port of `data_generate.m`; RNG streams necessarily differ from MATLAB."""
    rng = np.random.default_rng(seed)
    scaled = np.arange(1, n + 1) / n
    loading_1 = np.empty((n, p))
    loading_2 = np.empty((n, p))
    for asset in range(p):
        z = rng.normal(size=3)
        v = rng.normal(size=4)
        loading_1[:, asset] = rho * ((3 + z[0]) * scaled + z[1] * np.sin(4 * np.pi * scaled)) + z[2]
        loading_2[:, asset] = rho * ((3 + v[0]) * scaled + v[1] * np.sin(4 * np.pi * scaled) + (v[2] * scaled) ** 2) + v[3]
    residual_cov = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    factor = np.empty((n + 200, 2))
    state = rng.normal(size=2)
    for t in range(n + 200):
        state = np.array([0.6 * state[0] + rng.normal(scale=0.8), 0.3 * state[1] + rng.normal(scale=np.sqrt(0.91))])
        factor[t] = state
    factor = factor[200:]
    residual = rng.multivariate_normal(np.zeros(p), residual_cov, size=n)
    returns = loading_1 * factor[:, [0]] + loading_2 * factor[:, [1]] + residual
    endpoint_loading = np.c_[loading_1[-1], loading_2[-1]]
    return returns, endpoint_loading @ endpoint_loading.T + residual_cov, residual_cov


def _weekly_author_example(data: pd.DataFrame, seed: int) -> dict:
    gross_b, gross_a = [], []
    for number, position in enumerate(range(250, len(data), 5)):
        future = data.iloc[position : min(position + 5, len(data))].fillna(0).to_numpy()
        training = data.iloc[position - 250 : position].fillna(0).to_numpy()
        fit = fit_local_factor_model(training, 2)
        residual_cov, _ = sparse_residual_covariance(fit.residuals, 0.15, 1e-7)
        factor_cov, _ = sample_factor_covariance(fit.scores, 1e-7)
        covariance = fit.endpoint_loading @ factor_cov @ fit.endpoint_loading.T + residual_cov
        gross_b.extend(future @ minimum_variance_weights(covariance))
        gross_a.extend(future @ equal_weights(data.shape[1]))
    b, a = np.asarray(gross_b), np.asarray(gross_a)
    return {
        "Port weekly TV-MVP Sharpe (daily)": float(b.mean() / b.std(ddof=1)),
        "Port weekly equal-weight Sharpe (daily)": float(a.mean() / a.std(ddof=1)),
        "Port weekly TV-MVP mean": float(b.mean()),
        "Port weekly equal-weight mean": float(a.mean()),
        "Port weekly TV-MVP stdev": float(b.std(ddof=1)),
        "Port weekly equal-weight stdev": float(a.std(ddof=1)),
    }


def run_reproduction(config: dict) -> pd.DataFrame:
    root = Path(config["_extension_dir"])
    table_dir, plot_dir = root / "results" / "tables", root / "results" / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["seed"])
    returns, oracle_cov, residual_true = author_dgp(50, 200, 1.0, seed)
    selected, criterion = select_factor_number(returns, int(config["estimation"]["max_factor_number_reproduction"]))
    fit = fit_local_factor_model(returns, 2)
    residual_est, residual_info = sparse_residual_covariance(fit.residuals, 0.15, 1e-7)
    factor_cov, _ = sample_factor_covariance(fit.scores, 1e-7)
    tv_cov = fit.endpoint_loading @ factor_cov @ fit.endpoint_loading.T + residual_est
    sample_cov = np.cov(returns, rowvar=False)
    oracle_w = minimum_variance_weights(oracle_cov)
    tv_w = minimum_variance_weights(tv_cov)
    sample_w = minimum_variance_weights(sample_cov)
    author_example = load_author_example(root)
    weekly = _weekly_author_example(author_example, seed)
    values = {
        "Selected factor number": selected,
        "Oracle portfolio variance": float(oracle_w @ oracle_cov @ oracle_w),
        "TV-MVP forecast variance": float(tv_w @ tv_cov @ tv_w),
        "Sample-covariance forecast variance": float(sample_w @ sample_cov @ sample_w),
        "TV-MVP weight mean absolute error": float(np.mean(np.abs(tv_w - oracle_w))),
        "Sample weight mean absolute error": float(np.mean(np.abs(sample_w - oracle_w))),
        "Residual covariance relative Frobenius error": float(np.linalg.norm(residual_est - residual_true) / np.linalg.norm(residual_true)),
        "Residual sparsity": residual_info["off_diagonal_sparsity"],
        **weekly,
    }
    manual_targets = {
        "Selected factor number": 2,
        "Oracle portfolio variance": 0.1373,
        "TV-MVP forecast variance": 0.1000,
        "Sample-covariance forecast variance": 0.2117,
        "Port weekly TV-MVP Sharpe (daily)": 0.0865,
        "Port weekly equal-weight Sharpe (daily)": 0.0488,
        "Port weekly TV-MVP mean": 0.0006,
        "Port weekly equal-weight mean": 0.0005,
        "Port weekly TV-MVP stdev": 0.0075,
        "Port weekly equal-weight stdev": 0.0105,
    }
    rows = []
    for metric, value in values.items():
        rows.append({"Metric": metric, "Portable result": value, "Manual target": manual_targets.get(metric, np.nan), "Difference": value - manual_targets.get(metric, np.nan)})
    table = pd.DataFrame(rows)
    table.to_csv(table_dir / "reproduction_results.csv", index=False)
    pd.DataFrame({"Factors": np.arange(1, len(criterion) + 1), "Information criterion": criterion}).to_csv(
        table_dir / "reproduction_factor_selection.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(fit.loading_path[:, :, 0])
    ax.set(title="Reproduction: estimated first-factor loadings", xlabel="Observation", ylabel="Loading")
    fig.tight_layout(); fig.savefig(plot_dir / "reproduction_factor_loadings.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(residual_true, aspect="auto", cmap="viridis"); axes[0].set_title("True residual covariance")
    axes[1].imshow(residual_est, aspect="auto", cmap="viridis"); axes[1].set_title("Portable sparse estimate")
    fig.tight_layout(); fig.savefig(plot_dir / "reproduction_residual_covariance.png", dpi=180); plt.close(fig)
    return table
