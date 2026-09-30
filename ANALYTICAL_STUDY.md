# What TV-MVP changes: an empirical analysis of portfolio tradeoffs

This project studies what happens when portfolio construction shifts from broad
diversification to estimated risk minimization, tail-loss optimization, and
restrained trading. Its contribution is an auditable account of **which risks
fell, what return and trading costs accompanied them, and which apparent gains
failed to persist**. The study is descriptive: it does not claim a profitable
strategy or a newly confirmed trading rule.

## Scope and measurement

The frozen input contains adjusted prices for 39 long-history stocks selected
from a later NIFTY 50 membership list and a NIFTY price-index series. Backtests
start on 1 February 2017 after a 504-trading-day estimation window and end on
31 August 2026, giving 2,362 out-of-sample days. Monthly weights take effect
on the following trading day. Unless stated otherwise, portfolio comparisons
use the same fully invested, long-only, 10%-per-stock feasible set and a 10-bp
cost per unit of absolute traded notional.

Annual net return is compounded; Sharpe uses *net* daily returns and a zero
cash return. Volatility is annualized from *gross* daily returns to describe
exposure before trading costs. Expected shortfall (ES) is the average loss in
the worst 5% of **daily** returns, expressed as a daily percentage; it is not a
monthly risk measure. Drawdown is computed on net compounded wealth. These
units and the different strategy cohorts matter when reading the older reports,
some of which call a gross-return Sharpe simply “Sharpe.” The reproducible
calculation and source checksums are in
[`results/analysis/`](results/analysis/manifest.json); run
`python scripts/run_analytical_study.py` after installing the project.

## Five findings that survive the comparisons

### 1. Diversification weakened precisely when market risk rose

In 2020 the mean cross-stock daily correlation reached **0.432**, compared
with **0.137** in 2017. NIFTY price-index volatility rose from **9.0%** to
**31.3%** annualized, while the first covariance principal component's share
rose from **19.4%** to **48.4%**. On days flagged *in advance* by the project's
lagged high-volatility rule, the mean stock correlation was **0.397** versus
**0.211** on other days. This shows why a portfolio that appears diversified by
stock count can still have strong common-market exposure during stress. These
are correlations conditional on the observed sample, not evidence that the
volatility signal predicts future correlations.

![Market volatility and co-movement](results/analysis/market_context.png)

### 2. Minimum variance bought observable protection and surrendered upside

Under matched constraints, the following full-sample results capture the
central tradeoff. All numbers were recomputed from daily gross/net returns
and turnover, rather than mixed from tables with different conventions.

| Strategy | Annual net return | Annual gross volatility | Daily gross ES95 | Annual turnover |
|---|---:|---:|---:|---:|
| Equal weight | 19.74% | 15.94% | 2.33% | 0.31 |
| Static TV-MVP | 13.98% | 13.81% | 1.92% | 3.65 |
| Dynamic-factor TV-MVP | 14.13% | 13.89% | 1.94% | 3.86 |
| Empirical CVaR | 12.37% | 14.19% | 2.00% | 4.73 |

Static TV-MVP cut volatility by **2.13 percentage points** and daily ES by
**0.41 percentage points** versus equal weight, while giving up **5.76 points**
of annual compounded net return. Its net Sharpe was **1.02**, versus **1.21**
for equal weight. This risk reduction was not a one-year artifact: annual
gross volatility was lower for the static MVP in every observed calendar year
from 2017 through the partial 2026 sample. Equal weight had higher net calendar
return in nine of those ten periods. The 2026 comparison covers only part of a
year.

The allocation also had roughly **0.77 market beta** measured against the
equal-weight portfolio. On the 1,083 NIFTY down days, its gross return averaged
**−0.491%** versus **−0.633%** for equal weight. On the 1,277 NIFTY up days it
averaged **0.525%** versus **0.678%**. The same defensive tilt that softened
losses gave up gains; this is a conditional description using realized market
states, not a tradable signal. On the worst 5% of NIFTY days, its mean return
was **−1.67%**, versus **−2.23%** for equal weight.

![Matched portfolio tradeoffs](results/analysis/risk_return_cost.png)

### 3. The return sacrifice can be traced to holdings, not just costs

The stock-level daily attribution exactly reconstructs the saved static-MVP
and equal-weight gross returns. Equal weight exceeded static TV-MVP by **4.59
percentage points of annual arithmetic gross return**. About **2.47 points**
of that gap came from three stocks: Bajaj Finance, BEL and Adani Ports. Their
average static-MVP target weights were only **0.80%**, **0.80%** and **0.30%**,
versus **2.56%** each in equal weight; all three had strong returns in this
particular sample. This is an attribution identity, not proof that the
optimizer made a forecasting mistake: the optimizer minimizes estimated risk
and does not attempt to predict those gains.

Equal weight also carried all **39** stocks at each rebalance; static TV-MVP
had about **14.2 effective positions**, around **20** nonzero holdings, and an
average **46.5%** allocation in its five largest positions. A 10% individual
cap therefore did not guarantee broad diversification. Concentration itself
is not automatically bad: the static MVP was more concentrated than equal
weight and had lower realized volatility. What matters is the covariance and
return exposure of the positions, as well as the opportunity cost of excluding
future winners.

### 4. More complex risk estimates did not consistently improve decisions

The matched static and dynamic TV-MVP daily returns had **0.996 correlation**
and only **1.27% annualized tracking error**. Their full-sample volatilities
and daily ES values were almost identical, while the dynamic version traded
slightly more. The separate aligned-factor and dynamic-residual experiment
also failed its own primary variance test. The empirical lesson is about the
*incremental portfolio decision*: a changed risk estimate can leave actual
weights and returns nearly unchanged under caps and long-only constraints.

Empirical CVaR took about **12.8 effective positions** and hit the 10% cap
on **4.7 stocks** per rebalance, compared with **14.2** and **3.2** for static
TV-MVP. Despite optimizing estimated tail loss, it had higher realized daily
ES (**2.00%** versus **1.92%**) and higher annual turnover (**4.73** versus
**3.65**). Its net return was also lower. Scenario-tail calibration tables
show optimistic in-sample tail forecasts for the empirical CVaR, but those
monthly realized tail estimates are based on only about 20 trading days and
are very noisy. Filtered/stressed CVaR improved several 2021-onward numbers in
the earlier exploratory experiment; those results do not identify which design
choice caused the gain or establish that it will generalize.

### 5. Trading restraint is a real cost lever, but its value changed by period

A fixed turnover penalty of `3e-5` cut annual one-way turnover from **3.73**
to **0.97** through 2020, and from **3.69** to **0.70** from 2021 onward. In
the earlier period it improved both gross performance and cost-adjusted
performance. In the later period it sacrificed **2.27 percentage points** of
annual arithmetic gross return while saving only **0.60 points** of annual
trading cost at 10 bps. The net arithmetic effect was **−1.67 points**.

The later-period annual net Sharpe was **0.953** for the selected penalty and
**1.075** for baseline. Their net-Sharpe break-even trading cost, holding the
saved trading paths fixed, was about **34 bps** rather than the assumed 10 bps.
This is a cost sensitivity calculation, not a proposed fee or an estimate of
actual market impact. A penalty chosen from the earlier period's net Sharpe
did not transfer well to the later period. Lower turnover alone is therefore
not an investment result.

![Turnover and performance changed across periods](results/analysis/turnover_tradeoff.png)

## What the evidence does not support

Several extensions were inspected on this same dataset, so the analytical
tables are exploratory. They should not be presented as a sequence of
prospectively registered hypothesis tests. A crucial implementation audit also
invalidates the previously reported downside-aware TV-MVP result as evidence
of downside protection: PCA factors have arbitrary signs, yet reversing one
factor's sign while preserving all reconstructed asset returns changed its
proposed downside matrix by as much as **56%** and changed optimized weights by
as much as **0.31 in L1 distance** on the audited rebalance date. Ordinary
covariance was invariant to every sign flip. The earlier downside ES reduction
and p-value therefore depend on a representation choice with no economic
meaning. This experiment is retained in the historical record and excluded
from the conclusions above. The componentwise lower-partial moment does not
equal the lower-partial moment of the assembled asset portfolio, even with a
fixed sign convention.

Other limits also shape interpretation. The 39-stock universe was chosen from
later constituents with long histories, creating survivorship and membership
bias; its equal-weight performance must not be called the NIFTY 50 index
return. Price adjustment, missing-data filtering, the portable residual
covariance substitute, and the 10-bp trading-cost assumption affect every
backtest. The original author's empirical numbers were not exactly reproduced.
Market-regime and stock-attribution analyses describe outcomes after they
occurred. The figures are evidence about this dataset and implementation,
without a claim of future profitability.

## Reproduce and inspect

Run `python scripts/run_analytical_study.py` from the repository after
`python -m pip install -e .`. It reads only frozen prices and saved strategy
paths, checks return/cost identities, verifies matched constraints and stock
attribution, then writes 16 analysis tables, three figures and a SHA-256 input
manifest under [`results/analysis/`](results/analysis/manifest.json). The
script does not download new prices or refit strategy parameters. Start with
[`performance.csv`](results/analysis/performance.csv),
[`market_regimes.csv`](results/analysis/market_regimes.csv),
[`return_attribution.csv`](results/analysis/return_attribution.csv), and
[`cost_decomposition.csv`](results/analysis/cost_decomposition.csv).
