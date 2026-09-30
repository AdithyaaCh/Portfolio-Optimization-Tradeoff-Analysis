"""Descriptive synthesis of saved experiments; does not fit new trading strategies.

All outputs live in results/analysis, leaving historical experiment outputs intact.
Fractions (not percentages) are stored in tables. Sharpe uses zero cash return;
expected shortfall is a DAILY loss, with fractional empirical tail mass.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq

from .covariance import downside_second_moment, sample_factor_covariance, sparse_downside_residual_moment
from .factor import fit_local_factor_model
from .metrics import annualised_return, maximum_drawdown
from .optimisation import minimum_variance_weights


def empirical_es(losses: np.ndarray, confidence: float = .95) -> float:
    """Mean of the worst (1-alpha) probability mass, including a fractional atom."""
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 < confidence < 1:
        raise ValueError("ES requires finite nonempty one-dimensional data and 0 < alpha < 1")
    x = np.sort(x)[::-1]
    mass = (1 - confidence) * len(x)
    whole = int(np.floor(mass))
    fraction = mass - whole
    return float((x[:whole].sum() + fraction * x[whole]) / mass)


def summary(gross: pd.Series, net: pd.Series, turnover: pd.Series) -> dict:
    if not gross.index.equals(net.index) or not gross.index.equals(turnover.index):
        raise ValueError("Return and turnover dates must match exactly")
    g, n = gross.to_numpy(), net.to_numpy()
    if not np.isfinite(np.r_[g, n, turnover.to_numpy()]).all():
        raise ValueError("Missing or nonfinite observations")
    return {
        "Observations": len(n), "Start": str(net.index[0].date()), "End": str(net.index[-1].date()),
        "Net CAGR": annualised_return(n, 252), "Gross CAGR": annualised_return(g, 252),
        "Net Sharpe (rf=0)": float(n.mean() / n.std(ddof=1) * np.sqrt(252)),
        "Gross volatility": float(g.std(ddof=1) * np.sqrt(252)),
        "Gross daily ES95": empirical_es(-g), "Net daily ES95": empirical_es(-n),
        "Net max drawdown": maximum_drawdown(n),
        "Annual one-way turnover": float(turnover.sum() * 252 / len(n)),
        "Annual arithmetic cost drag": float((g - n).mean() * 252),
    }


def lagged_high_volatility(index_returns: pd.Series, config: dict) -> pd.Series:
    """Expanding, lagged volatility classification available before each return."""
    evaluation = config["evaluation"]
    trailing = index_returns.rolling(int(evaluation["index_volatility_days"])).std(ddof=1) * np.sqrt(252)
    known = trailing.shift(1)
    threshold = known.expanding(int(evaluation["high_volatility_min_history"])).quantile(
        float(evaluation["high_volatility_quantile"])
    ).shift(1)
    return (known > threshold).fillna(False).rename("High volatility")


def adjusted_close_returns(prices: pd.DataFrame, index_prices: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = prices.index.intersection(index_prices.index)
    prices = prices.reindex(common).dropna(axis=0, how="any")
    index_prices = index_prices.reindex(prices.index).ffill()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    index_returns = index_prices.pct_change(fill_method=None).reindex(returns.index)
    valid = index_returns.notna()
    return returns.loc[valid], index_returns.loc[valid]


def run_analytical_study(config: dict) -> dict[str, pd.DataFrame]:
    root = Path(config["_extension_dir"])
    source = root / "results" / "tables"
    output = root / "results" / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    inputs: set[Path] = set()
    cost_rate = float(config["evaluation"]["transaction_cost_bps"]) / 10000

    def read(path: Path, dates: bool = False) -> pd.DataFrame:
        inputs.add(path)
        frame = pd.read_csv(path, index_col=0 if dates else None, parse_dates=dates)
        if dates and (not frame.index.is_unique or not frame.index.is_monotonic_increasing):
            raise ValueError(f"Invalid time index in {path}")
        return frame

    def cohort(net_name: str, turn_name: str, gross_name: str | None = None):
        net = read(source / net_name, True)
        turn = read(source / turn_name, True)
        if not net.index.equals(turn.index) or not net.columns.equals(turn.columns):
            raise ValueError(f"Mismatched series: {net_name}")
        gross = read(source / gross_name, True) if gross_name else net + cost_rate * 2 * turn
        if not gross.index.equals(net.index) or not gross.columns.equals(net.columns):
            raise ValueError(f"Mismatched gross series: {net_name}")
        if not np.allclose(gross - net, cost_rate * 2 * turn, atol=1e-12, rtol=0):
            raise ValueError(f"Saved costs do not reconcile: {net_name}")
        return gross, net, turn

    cohorts = {
        "Matched": cohort("matched_daily_net_returns.csv", "matched_daily_turnover.csv"),
        "CVaR repair": cohort("improved_cvar_daily_net_returns.csv", "improved_cvar_daily_turnover.csv"),
        "Fixed turnover": cohort("turnover_smoke_daily_net_returns.csv", "turnover_smoke_daily_turnover.csv", "turnover_smoke_daily_gross_returns.csv"),
    }
    prices = read(root / "data/raw/nifty50_adjusted_close.csv", True)
    index_prices = read(root / "data/raw/nifty50_index_adjusted_close.csv", True).iloc[:, 0]
    assets, index_returns = adjusted_close_returns(prices, index_prices)
    full_dates = cohorts["Matched"][1].index
    aligned_assets = assets.loc[full_dates]
    high_regime = lagged_high_volatility(index_returns, config).loc[full_dates]

    outputs: dict[str, pd.DataFrame] = {}
    rows, years, pairs = [], [], []
    for name, (gross, net, turn) in cohorts.items():
        periods = {"All available": np.ones(len(net), bool), "Through 2020": net.index.year <= 2020,
                   "2021 onward": net.index.year >= 2021}
        for period, mask in periods.items():
            if not mask.any():
                continue
            for model in net:
                rows.append({"Cohort": name, "Period": period, "Model": model,
                             **summary(gross.loc[mask, model], net.loc[mask, model], turn.loc[mask, model])})
        for year, frame in net.groupby(net.index.year):
            for model in net:
                x = frame[model]
                years.append({"Cohort": name, "Year": year, "Model": model,
                              "Partial calendar year": year in (2017, 2026),
                              "Net period return": float((1 + x).prod() - 1),
                              **summary(gross.loc[frame.index, model], x, turn.loc[frame.index, model])})
        if name == "Matched":
            # Static vs dynamic similarity, computed within its own experiment.
            static, dynamic = list(net.columns)[1:3]
            diff = gross[dynamic] - gross[static]
            pairs.append({"Cohort": name, "Static": static, "Dynamic": dynamic,
                          "Daily return correlation": gross[static].corr(gross[dynamic]),
                          "Annual tracking error": diff.std(ddof=1) * np.sqrt(252)})
    outputs["performance"] = pd.DataFrame(rows)
    outputs["yearly"] = pd.DataFrame(years)
    outputs["dynamic_similarity"] = pd.DataFrame(pairs)

    market_rows = []
    for year, frame in assets.groupby(assets.index.year):
        corr = frame.corr().to_numpy()
        covariance = frame.cov().to_numpy()
        market = index_returns.reindex(frame.index)
        market_rows.append({"Year": year, "Days": len(frame), "Partial calendar year": year == 2026,
            "Mean pair correlation": corr[~np.eye(len(corr), dtype=bool)].mean(),
            "Index volatility": market.std(ddof=1) * np.sqrt(252),
            "Index period return (price index)": (1 + market).prod() - 1,
            "First covariance PC share": np.linalg.eigvalsh(covariance)[-1] / np.trace(covariance),
            "Median asset excess kurtosis": frame.kurt().median()})
    outputs["market_years"] = pd.DataFrame(market_rows)
    regime_rows = []
    for regime, mask in (("Lagged high volatility", high_regime), ("Other days", ~high_regime)):
        corr = aligned_assets.loc[mask].corr().to_numpy()
        regime_rows.append({"Regime": regime, "Days": int(mask.sum()),
            "Mean pair correlation": corr[~np.eye(len(corr), dtype=bool)].mean(),
            "Index volatility": index_returns.loc[full_dates][mask].std(ddof=1) * np.sqrt(252)})
    outputs["market_regimes"] = pd.DataFrame(regime_rows)

    # Common-state comparisons: same market days for every model; no traded signal.
    matched_gross, matched_net, matched_turn = cohorts["Matched"]
    ew = matched_gross["Equal weight"]
    i = index_returns.loc[full_dates]
    states = {"Index up days": i > 0, "Index down days": i < 0,
              "Worst 5% index days (ex post)": i <= i.quantile(.05),
              "Lagged high volatility": high_regime, "Other days": ~high_regime}
    conditional = []
    for state, mask in states.items():
        for model in matched_net:
            x = matched_gross.loc[mask, model]
            conditional.append({"State": state, "Model": model, "Days": len(x),
                "Mean gross daily return": x.mean(),
                "Mean difference vs equal weight": (x - ew.loc[mask]).mean(),
                "Gross volatility": x.std(ddof=1) * np.sqrt(252), "Gross daily ES95": empirical_es(-x.to_numpy())})
    outputs["conditional_returns"] = pd.DataFrame(conditional)
    exposure = []
    for model in matched_gross:
        x = matched_gross[model]
        exposure.append({"Model": model, "Beta to equal weight": x.cov(ew) / ew.var(),
                         "Correlation to equal weight": x.corr(ew),
                         "Annual tracking error vs equal weight": (x - ew).std() * np.sqrt(252)})
    outputs["exposure"] = pd.DataFrame(exposure)

    # Risk reduction can result from concentrated defensive positions.
    weight_rows = []
    allocation_rows = []
    for name, filename in (("Matched", "matched_weights.csv"), ("CVaR repair", "improved_cvar_weights.csv")):
        weights = read(source / filename)
        weights["Rebalance date"] = pd.to_datetime(weights["Rebalance date"])
        for (date, model), group in weights.groupby(["Rebalance date", "Model"]):
            w = group.Weight.to_numpy()
            if not np.isclose(w.sum(), 1, atol=1e-7):
                raise ValueError("Weights do not sum to one")
            if w.min() < -1e-7 or w.max() > .1000001:
                raise ValueError("Matched weights violate constraints")
            weight_rows.append({"Cohort": name, "Date": date, "Model": model,
                "Effective positions (1/sum(w^2))": 1 / (w @ w), "Gross exposure": np.abs(w).sum(),
                "Top five signed weights": np.sort(w)[-5:].sum(),
                "Positions at 10% cap": np.isclose(w, .1, atol=1e-5).sum(),
                "Nonzero positions": (np.abs(w) > 1e-7).sum()})
        if name == "Matched":
            for (model, asset), group in weights.groupby(["Model", "Asset"]):
                allocation_rows.append({"Model": model, "Asset": asset, "Mean rebalance weight": group.Weight.mean(),
                                        "Fraction of rebalances held": (group.Weight > 1e-7).mean()})
    outputs["concentration_path"] = pd.DataFrame(weight_rows)
    outputs["concentration"] = outputs["concentration_path"].groupby(["Cohort", "Model"]).mean(numeric_only=True).reset_index()
    outputs["average_allocations"] = pd.DataFrame(allocation_rows)

    # Exact arithmetic attribution from saved target weights and drifting holdings.
    # It explains the observed return gap, not why the optimizer selected an asset.
    weights = read(source / "matched_weights.csv")
    weights["Rebalance date"] = pd.to_datetime(weights["Rebalance date"])
    contributions = {}
    mean_weights = {}
    for model in ("Equal weight", "Original covariance MVP (matched)"):
        targets = weights[weights.Model == model].pivot(index="Rebalance date", columns="Asset", values="Weight").reindex(columns=assets.columns)
        contrib = pd.DataFrame(index=full_dates, columns=assets.columns, dtype=float)
        for number, (date, target) in enumerate(targets.iterrows()):
            end = targets.index[number + 1] if number + 1 < len(targets) else full_dates[-1]
            future = assets.loc[(assets.index > date) & (assets.index <= end)]
            held = target.to_numpy().copy()
            for day, row in future.iterrows():
                terms = held * row.to_numpy()
                contrib.loc[day] = terms
                held = held * (1 + row.to_numpy()) / (1 + terms.sum())
        if not np.allclose(contrib.sum(axis=1), matched_gross[model], atol=1e-11, rtol=0):
            raise ValueError("Asset attribution does not reproduce saved portfolio returns")
        contributions[model] = contrib
        mean_weights[model] = targets.mean()
    base_name = "Original covariance MVP (matched)"
    attribution_rows = []
    for asset in assets:
        attribution_rows.append({"Asset": asset,
            "Static mean rebalance weight": mean_weights[base_name][asset],
            "Equal mean rebalance weight": mean_weights["Equal weight"][asset],
            "Asset CAGR over OOS dates": annualised_return(assets.loc[full_dates, asset], 252),
            "Static annual arithmetic return contribution": contributions[base_name][asset].mean() * 252,
            "Equal annual arithmetic return contribution": contributions["Equal weight"][asset].mean() * 252,
            "Equal-minus-static contribution": (contributions["Equal weight"][asset] - contributions[base_name][asset]).mean() * 252})
    outputs["return_attribution"] = pd.DataFrame(attribution_rows).sort_values("Equal-minus-static contribution", ascending=False)

    # Exact cost-accounting decomposition for the original selected penalty.
    cost_rows = []
    for period, begin, end in (("Through 2020", "2017", "2020-12-31"), ("2021 onward", "2021", "2026-12-31")):
        g, n, t = [x.loc[begin:end] for x in cohorts["Fixed turnover"]]
        base, target = "TV-MVP baseline", "TV-MVP L1=3e-05"
        gross_gap = float((g[target] - g[base]).mean() * 252)
        trading_saved = float((t[base] - t[target]).mean() * 2 * 252)
        savings = trading_saved * cost_rate

        def sharpe_gap(bps):
            adjusted = g - bps / 10000 * 2 * t
            s = adjusted.mean() / adjusted.std(ddof=1) * np.sqrt(252)
            return float(s[target] - s[base])

        root_cost = (brentq(sharpe_gap, 0, 200) if sharpe_gap(0) * sharpe_gap(200) < 0 else np.nan)
        cost_rows.append({"Period": period, "Gross arithmetic return difference": gross_gap,
            "Annual cost saving at 10 bps": savings, "Net arithmetic return difference": gross_gap + savings,
            "Annual absolute notional saved": trading_saved,
            "Arithmetic-return break-even bps": -gross_gap / trading_saved * 10000,
            "Net-Sharpe break-even bps in [0,200]": root_cost})
    outputs["cost_decomposition"] = pd.DataFrame(cost_rows)

    # Annual risk rankings and tail concentration are descriptive stability checks.
    tail_years = []
    for model in matched_net:
        losses = -matched_net[model]
        worst = losses[losses >= losses.quantile(.95)]
        for year, frame in worst.groupby(worst.index.year):
            tail_years.append({"Model": model, "Year": year, "Tail days": len(frame),
                               "Share of worst-5% loss sum": frame.sum() / worst.sum()})
    outputs["tail_year_contributions"] = pd.DataFrame(tail_years)

    # Joint lower-partial factor moments are NOT invariant to arbitrary PCA signs.
    # Use the final actual rebalance window; exhaust all sign conventions, not model tuning.
    last_weights = read(source / "matched_weights.csv")
    last_date = pd.Timestamp(last_weights["Rebalance date"].max())
    est = config["estimation"]
    fit = fit_local_factor_model(assets.loc[:last_date].iloc[-int(est["window_days"]):].to_numpy(),
                                int(est["factor_number"]), est["bandwidth"], align_to_endpoint=True)
    b, f = fit.endpoint_loading, fit.scores
    floor = float(est["eigenvalue_floor"])
    residual, _ = sparse_downside_residual_moment(fit.residuals, float(est["residual_threshold"]), floor)
    ds, _ = downside_second_moment(f, floor)
    cs, _ = sample_factor_covariance(f, floor)
    original = b @ ds @ b.T + residual
    w0 = minimum_variance_weights(original, True, .1)
    sign_rows = []
    for signs in itertools.product((1, -1), repeat=f.shape[1]):
        z = np.array(signs)
        bz, fz = b * z, f * z
        d, _ = downside_second_moment(fz, floor)
        c, _ = sample_factor_covariance(fz, floor)
        changed = bz @ d @ bz.T + residual
        w = minimum_variance_weights(changed, True, .1)
        sign_rows.append({"Rebalance date": str(last_date.date()), "Factor signs": str(signs),
            "Return reconstruction max change": np.max(np.abs(f @ b.T - fz @ bz.T)),
            "Standard covariance relative change": np.linalg.norm(b @ cs @ b.T - bz @ c @ bz.T) / np.linalg.norm(b @ cs @ b.T),
            "Downside matrix relative change": np.linalg.norm(original - changed) / np.linalg.norm(original),
            "Downside weight L1 change": np.abs(w0 - w).sum()})
    outputs["downside_sign_audit"] = pd.DataFrame(sign_rows)

    diagnostics = read(source / "factor_diagnostics.csv")
    outputs["factor_diagnostics_summary"] = diagnostics.drop(columns="Rebalance date").describe().T.reset_index(names="Metric")
    outputs["cvar_calibration_context"] = read(source / "improved_cvar_tail_calibration.csv")
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    plot_study(outputs, output)
    # Hash every actual input plus source; the saved summary is not a backtest rerun.
    inputs.add(Path(config["_config_path"]))
    inputs.update((root / "src/tvmvp_ext").glob("*.py"))
    manifest = {
        "purpose": "Exploratory descriptive synthesis; no new portfolio optimization experiment",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "cost_bps": float(config["evaluation"]["transaction_cost_bps"]), "risk_free_rate": 0,
        "es_units": "daily decimal loss, exact fractional 5% tail",
        "data_start": str(prices.index.min().date()), "data_end": str(prices.index.max().date()),
        "assets": prices.shape[1], "missing_frozen_prices": int(prices.isna().sum().sum()),
        "oos_start": str(full_dates[0].date()), "oos_end": str(full_dates[-1].date()),
        "comparability": "Compare within cohorts; initial trading cost and alignment differ across experiments.",
        "inputs_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def plot_study(tables: dict[str, pd.DataFrame], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    market = tables["market_years"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax[0].bar(market.Year, 100 * market["Index volatility"], color="#315b7c")
    ax[0].set(title="Volatility varied sharply across years", ylabel="NIFTY price-index volatility (%)", xticks=market.Year)
    ax[1].plot(market.Year, market["Mean pair correlation"], marker="o", color="#b95d36", label="Mean stock correlation")
    ax[1].plot(market.Year, market["First covariance PC share"], marker="s", color="#315b7c", label="First PC variance share")
    ax[1].set(title="Common movement rose in 2020", ylabel="Correlation / variance share", xticks=market.Year)
    ax[1].legend(fontsize=9)
    for a in ax:
        a.tick_params(axis="x", rotation=45)
    fig.suptitle("Market context • 39 frozen constituent histories • 2026 is partial", fontsize=13)
    fig.savefig(output / "market_context.png", dpi=180); plt.close(fig)

    perf = tables["performance"]
    p = perf[(perf.Cohort == "Matched") & (perf.Period == "All available")].copy()
    labels = ["Equal weight", "Static MVP", "Dynamic MVP", "Empirical CVaR"]
    colors = ["#315b7c", "#388576", "#9c7baf", "#b95d36"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    for a, col, title in zip(ax, ("Net CAGR", "Gross daily ES95", "Annual one-way turnover"),
                           ("Annual net return (%)", "Daily expected shortfall (%)", "Annual one-way turnover")):
        values = p[col].to_numpy() * (1 if col == "Annual one-way turnover" else 100)
        a.barh(labels, values, color=colors)
        for i, v in enumerate(values):
            a.text(v, i, f"  {v:.2f}", va="center", fontsize=9)
        a.set(xlabel=title, xlim=(0, max(values) * 1.24)); a.invert_yaxis()
    fig.suptitle("Defensive allocation reduced risk and return • matched constraints • 2017–2026", fontsize=13)
    fig.savefig(output / "risk_return_cost.png", dpi=180); plt.close(fig)

    fixed = perf[(perf.Cohort == "Fixed turnover") & perf.Period.isin(["Through 2020", "2021 onward"])]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for period, color in (("Through 2020", "#315b7c"), ("2021 onward", "#b95d36")):
        frame = fixed[fixed.Period == period]
        for a, col, multiplier in ((ax[0], "Net Sharpe (rf=0)", 1), (ax[1], "Gross volatility", 100)):
            a.plot(frame["Annual one-way turnover"], frame[col] * multiplier, "o-", color=color, label=period)
            target = frame[frame.Model == "TV-MVP L1=3e-05"].iloc[0]
            a.scatter([target["Annual one-way turnover"]], [target[col] * multiplier], marker="*", s=190, color=color, edgecolor="black", zorder=5)
        ax[0].annotate("Baseline", (frame.iloc[0]["Annual one-way turnover"], frame.iloc[0]["Net Sharpe (rf=0)"]), xytext=(-45, 9), textcoords="offset points", fontsize=8)
    ax[0].set(xlabel="Annual one-way turnover", ylabel="Net Sharpe (zero cash return)", title="The return tradeoff changed over time")
    ax[1].set(xlabel="Annual one-way turnover", ylabel="Annual gross volatility (%)", title="Very low turnover need not minimize risk")
    ax[0].legend(fontsize=9)
    fig.suptitle("Fixed penalty grid • stars mark the earlier-period selected penalty • exploratory", fontsize=13)
    fig.savefig(output / "turnover_tradeoff.png", dpi=180); plt.close(fig)
