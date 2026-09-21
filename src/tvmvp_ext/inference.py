from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import annualised_return, expected_shortfall


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block resample with exactly n observations."""
    output: list[int] = []
    while len(output) < n:
        start = int(rng.integers(0, n))
        output.extend(((start + np.arange(block_length)) % n).tolist())
    return np.asarray(output[:n])


def _loss_statistics(monthly_returns: np.ndarray, confidence: float) -> np.ndarray:
    losses = -monthly_returns
    return np.array([
        np.std(losses, ddof=1) * np.sqrt(12),
        expected_shortfall(losses, confidence),
        annualised_return(monthly_returns, 12),
    ])


def paired_strategy_inference(
    monthly_net: pd.DataFrame,
    draws: int,
    block_length: int,
    confidence: float,
    primary: list[list[str]],
    seed: int,
) -> pd.DataFrame:
    """Intervals use paired block resamples; p-values appear only for registered comparisons."""
    rng = np.random.default_rng(seed)
    names = list(monthly_net.columns)
    primary_set = {tuple(pair) for pair in primary}
    labels = ["Annualised volatility difference", "Monthly ES difference", "Annualised net return difference"]
    rows = []
    values = monthly_net.to_numpy()
    for left, right in combinations(range(len(names)), 2):
        point = _loss_statistics(values[:, left], confidence) - _loss_statistics(values[:, right], confidence)
        boot = np.empty((draws, 3))
        for draw in range(draws):
            idx = moving_block_indices(len(values), block_length, rng)
            boot[draw] = _loss_statistics(values[idx, left], confidence) - _loss_statistics(values[idx, right], confidence)
        is_primary = (names[left], names[right]) in primary_set or (names[right], names[left]) in primary_set
        for metric, estimate, distribution in zip(labels, point, boot.T):
            lower, upper = np.quantile(distribution, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
            p_value = min(1.0, 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0))) if is_primary else np.nan
            rows.append({
                "Strategy 1": names[left],
                "Strategy 2": names[right],
                "Metric (1 minus 2)": metric,
                "Estimate": estimate,
                "CI lower": lower,
                "CI upper": upper,
                "Bootstrap p-value": p_value,
                "Primary comparison": is_primary,
                "Bootstrap draws": draws,
                "Block length (months)": block_length,
            })
    return pd.DataFrame(rows)


def regime_benefit_test(
    daily_returns: pd.DataFrame,
    high_regime: pd.Series,
    draws: int,
    block_length: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Test whether B-minus-C risk benefit is larger in high-volatility observations."""
    b = daily_returns["Original TV-MVP"].to_numpy()
    c = daily_returns["Dynamic SigmaF TV-MVP"].to_numpy()
    high = high_regime.reindex(daily_returns.index).fillna(False).to_numpy(bool)

    def statistic(indices: np.ndarray) -> tuple[float, float]:
        bh, ch = b[indices][high[indices]], c[indices][high[indices]]
        bn, cn = b[indices][~high[indices]], c[indices][~high[indices]]
        vol = (np.std(bh, ddof=1) - np.std(ch, ddof=1)) - (np.std(bn, ddof=1) - np.std(cn, ddof=1))
        es = (expected_shortfall(-bh, confidence) - expected_shortfall(-ch, confidence)) - (
            expected_shortfall(-bn, confidence) - expected_shortfall(-cn, confidence)
        )
        return float(vol * np.sqrt(252)), float(es * np.sqrt(252))

    all_idx = np.arange(len(b))
    point = np.asarray(statistic(all_idx))
    rng = np.random.default_rng(seed)
    boot = np.asarray([statistic(moving_block_indices(len(b), block_length, rng)) for _ in range(draws)])
    rows = []
    for label, estimate, distribution in zip(("Volatility benefit interaction", "ES benefit interaction"), point, boot.T):
        lower, upper = np.quantile(distribution, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
        p_value = min(1.0, 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0)))
        rows.append({"Metric": label, "Estimate": estimate, "CI lower": lower, "CI upper": upper, "Bootstrap p-value": p_value})
    return pd.DataFrame(rows)


def pair_regime_interaction(
    daily_returns: pd.DataFrame,
    high_regime: pd.Series,
    pair: list[str],
    draws: int,
    block_length: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Block-bootstrap whether strategy 2's advantage over strategy 1 changes in high volatility."""
    first, second = pair
    x = daily_returns[[first, second]].to_numpy()
    high = high_regime.reindex(daily_returns.index).fillna(False).to_numpy(bool)

    def statistics(indices: np.ndarray) -> np.ndarray:
        sampled = x[indices]
        sampled_high = high[indices]
        output = []
        for mask in (sampled_high, ~sampled_high):
            left, right = sampled[mask, 0], sampled[mask, 1]
            risk_benefit = np.array([
                (np.std(left, ddof=1) - np.std(right, ddof=1)) * np.sqrt(252),
                (expected_shortfall(-left, confidence) - expected_shortfall(-right, confidence)) * np.sqrt(252),
            ])
            return_benefit = (np.mean(right) - np.mean(left)) * 252
            output.append(np.r_[risk_benefit, return_benefit])
        return output[0] - output[1]

    all_indices = np.arange(len(x))
    point = statistics(all_indices)
    rng = np.random.default_rng(seed)
    boot = np.asarray([statistics(moving_block_indices(len(x), block_length, rng)) for _ in range(draws)])
    labels = (
        "Volatility benefit: high minus normal",
        "ES benefit: high minus normal",
        "Arithmetic net-return benefit: high minus normal",
    )
    rows = []
    for label, estimate, distribution in zip(labels, point, boot.T):
        lower, upper = np.quantile(distribution, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
        p_value = min(1.0, 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0)))
        rows.append({
            "Strategy 1": first,
            "Strategy 2": second,
            "Metric": label,
            "Estimate": estimate,
            "CI lower": lower,
            "CI upper": upper,
            "Bootstrap p-value": p_value,
            "Bootstrap draws": draws,
            "Block length (days)": block_length,
        })
    return pd.DataFrame(rows)
