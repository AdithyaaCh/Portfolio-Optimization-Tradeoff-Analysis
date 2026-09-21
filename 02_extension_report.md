# 02 - Extension report

## Models and information timing

Every NIFTY backtest rebalance occurs at a trading month-end. Estimation uses the prior 504 daily returns including that close; weights apply only from the next trading day through the following month-end. The four models share dates and data:

- A: monthly rebalanced equal weight.
- B: author-style unconstrained TV-MVP, `Sigma_r,t = B_t Sigma_F B_t' + Sigma_e`, with full-window sample factor covariance.
- C: the same estimator and feasible set, replacing only `Sigma_F` with normalized EWMA covariance (`lambda = 0.94`; effective sample size 32.3).
- D: empirical 95% CVaR, long-only with a 10% cap.

Local PCA is fitted throughout each trailing window. Estimated scores solve a contemporaneous least-squares projection against each date's local loading. Residual correlations are soft-thresholded using a dimension/sample-size-scaled cutoff. Every covariance is symmetrized and eigenvalues below `1e-7` times spectral scale are clipped; the empirical run required no clips.

## CVaR scenarios

At each date, the code circular-moving-block bootstraps paired `(factor score, residual vector)` observations in 10-day blocks. Pairing retains contemporaneous factor-residual dependence; blocks retain short serial clusters. Six hundred scenarios are mapped through the current loading, `r^(s) = B_t f^(s) + e^(s)`. The Rockafellar-Uryasev linear program minimizes empirical loss CVaR. It does not estimate expected return and cannot generate tail states absent from the rolling history.

Models B/C are unconstrained because that is the author benchmark; D's long-only cap was explicitly requested. Thus C-vs-D differences combine risk objective and feasible-set effects. The sensitivity table varies CVaR confidence; the cap can be disabled or changed in `config.yaml`.

## Evaluation and inference

Daily gross returns are compounded. A 10 bp cost per one-way traded notional is charged at each rebalance; reported turnover is half the L1 weight change and accounts for weight drift. Metrics include annualized return/volatility, zero-rate Sharpe, maximum drawdown, annualized daily 95% ES, turnover, and net return.

The primary hypotheses were fixed before result inspection: B versus C, and C versus D. Daily net returns are compounded into non-overlapping monthly returns and losses are their negatives. Paired differences in annualized volatility, monthly 95% ES, and annualized net return use 1,999 circular moving-block draws with three-month blocks. Only primary comparisons receive p-values; other pairwise intervals are descriptive.

High volatility is defined without portfolio outcomes: trailing 21-day NIFTY volatility, lagged one day, above its lagged expanding 75th percentile after 126 observations. A separate 21-day-block bootstrap tests whether C's risk benefit over B is larger in high-volatility observations.

Covariance forecasts are compared with next-month realised sample covariance using Frobenius loss and absolute portfolio-risk loss. Diagnostics cover retained-factor variance share, maximum principal angle between consecutive loading spaces, sparse residual-correlation share, excess kurtosis, and QQ plots. Kurtosis/QQ evidence is descriptive; large-sample normality p-values are intentionally not emphasized.

Monte Carlo uses 100 independent replications in each of Gaussian, Student-t(5), and abrupt volatility/correlation-shift settings. Each reports means, standard deviations, 95% Monte Carlo intervals, true-covariance errors, and beat probabilities. Parameter grids are robustness evidence, not additional confirmatory hypotheses.

