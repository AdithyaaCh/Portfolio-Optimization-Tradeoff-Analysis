from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .metrics import rolling_expected_shortfall


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative(net_returns: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    (1 + net_returns).cumprod().plot(ax=ax)
    ax.set(title="Net cumulative wealth", ylabel="Wealth (initial = 1)", xlabel="")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_rolling_risk(returns: pd.DataFrame, window: int, confidence: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    (returns.rolling(window).std() * np.sqrt(252)).plot(ax=axes[0])
    pd.DataFrame({c: rolling_expected_shortfall(returns[c], window, confidence) * np.sqrt(252) for c in returns}).plot(ax=axes[1])
    axes[0].set(ylabel="Annualised volatility", title=f"Rolling {window}-day risk")
    axes[1].set(ylabel="Annualised ES", xlabel="")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, path)


def plot_weights_turnover(weights: pd.DataFrame, turnover: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=weights, x="Model", y="Weight", ax=axes[0], showfliers=False)
    sns.boxplot(data=turnover.melt(var_name="Model", value_name="Turnover").query("Turnover > 0"), x="Model", y="Turnover", ax=axes[1], showfliers=False)
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].tick_params(axis="x", rotation=25)
    axes[0].set_title("Weight distribution")
    axes[1].set_title("One-way turnover distribution")
    _save(fig, path)


def plot_factor_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    data = diagnostics.set_index("Rebalance date")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    data["Variance explained"].plot(ax=axes[0])
    data["Maximum principal angle (degrees)"].plot(ax=axes[1])
    data["Residual-correlation sparsity"].plot(ax=axes[2])
    axes[0].set_ylabel("Share")
    axes[1].set_ylabel("Degrees")
    axes[2].set_ylabel("Zero fraction")
    axes[0].set_title("Factor-model diagnostics by rebalance")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, path)


def plot_qq(scores: np.ndarray, residuals: np.ndarray, path: Path) -> None:
    score_values = ((scores - scores.mean(0)) / scores.std(0, ddof=1)).ravel()
    residual_values = ((residuals - residuals.mean(0)) / residuals.std(0, ddof=1)).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    stats.probplot(score_values[np.isfinite(score_values)], dist="norm", plot=axes[0])
    stats.probplot(residual_values[np.isfinite(residual_values)], dist="norm", plot=axes[1])
    axes[0].set_title("Standardised factor scores")
    axes[1].set_title("Standardised residuals")
    _save(fig, path)


def plot_sensitivity_heatmap(table: pd.DataFrame, path: Path) -> None:
    pivot = table.pivot_table(index="Parameter", columns="Value", values="Dynamic-minus-original Sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", center=0, cmap="vlag", ax=ax)
    ax.set_title("Robustness only: change in Sharpe vs original TV-MVP")
    _save(fig, path)
