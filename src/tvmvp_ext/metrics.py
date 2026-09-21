from __future__ import annotations

import numpy as np
import pandas as pd


def expected_shortfall(losses: np.ndarray | pd.Series, confidence: float = 0.95) -> float:
    values = np.asarray(losses, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    threshold = np.quantile(values, confidence)
    tail = values[values >= threshold]
    return float(tail.mean())


def maximum_drawdown(returns: np.ndarray | pd.Series) -> float:
    wealth = np.cumprod(1 + np.asarray(returns, float))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return float(np.min(wealth / peak - 1.0))


def annualised_return(returns: np.ndarray | pd.Series, periods: int) -> float:
    values = np.asarray(returns, float)
    if len(values) == 0 or np.any(values <= -1):
        return np.nan
    return float(np.prod(1 + values) ** (periods / len(values)) - 1)


def performance_table(
    gross: pd.DataFrame,
    net: pd.DataFrame,
    turnover: pd.DataFrame,
    annualisation_days: int,
    confidence: float,
) -> pd.DataFrame:
    rows = []
    years = max(len(gross) / annualisation_days, 1e-12)
    for model in gross.columns:
        values = gross[model].dropna()
        net_values = net[model].dropna()
        volatility = values.std(ddof=1) * np.sqrt(annualisation_days)
        rows.append({
            "Model": model,
            "Annualised return": annualised_return(values, annualisation_days),
            "Annualised volatility": float(volatility),
            "Sharpe ratio": float(values.mean() / values.std(ddof=1) * np.sqrt(annualisation_days)),
            "Maximum drawdown": maximum_drawdown(values),
            "Realised ES 95% (annualised)": expected_shortfall(-values, confidence) * np.sqrt(annualisation_days),
            "Annualised turnover": float(turnover[model].sum() / years),
            "Annualised net return": annualised_return(net_values, annualisation_days),
            "Terminal net wealth": float(np.prod(1 + net_values)),
        })
    return pd.DataFrame(rows).set_index("Model")


def aggregate_monthly(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns).groupby(returns.index.to_period("M")).prod() - 1


def rolling_expected_shortfall(series: pd.Series, window: int, confidence: float) -> pd.Series:
    return series.rolling(window).apply(lambda x: expected_shortfall(-x, confidence), raw=True)


def regime_performance(returns: pd.DataFrame, regime: pd.Series, annualisation_days: int, confidence: float) -> pd.DataFrame:
    rows = []
    for label, mask in (("High volatility", regime), ("Normal volatility", ~regime)):
        for model in returns.columns:
            values = returns.loc[mask.reindex(returns.index).fillna(False), model].dropna()
            rows.append({
                "Regime": label,
                "Model": model,
                "Observations": len(values),
                "Annualised return": annualised_return(values, annualisation_days),
                "Annualised volatility": float(values.std(ddof=1) * np.sqrt(annualisation_days)),
                "Sharpe ratio": float(values.mean() / values.std(ddof=1) * np.sqrt(annualisation_days)),
                "Realised ES 95% (annualised)": expected_shortfall(-values, confidence) * np.sqrt(annualisation_days),
            })
    return pd.DataFrame(rows)

