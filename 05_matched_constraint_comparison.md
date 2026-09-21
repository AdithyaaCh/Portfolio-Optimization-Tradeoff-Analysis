# 05 - Matched-constraint MVP versus CVaR

## Design

This rerun removes the earlier feasible-set confound. Original-covariance MVP, dynamic-covariance MVP, and dynamic-factor CVaR are all long-only, sum to one, and capped at 10% per stock. They use the same 39 assets, 504-day windows, monthly dates, factor/residual estimates, 600 bootstrap scenarios, and 10 bp one-way trading cost. Equal weight remains descriptive. “Post-COVID” was fixed at 2021-01-01.

## Measure-based return and risk comparison

| Model, 2021 onward | Gross return | Net return | Volatility | Sharpe | Max drawdown | ES 95% | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal weight | 20.14% | 20.07% | 13.87% | 1.393 | -15.15% | 32.09% | 0.29 |
| Original-covariance MVP | 13.62% | 12.79% | 11.86% | 1.136 | -17.13% | 26.17% | 3.69 |
| Dynamic-covariance MVP | 13.52% | 12.62% | 11.92% | 1.123 | -16.63% | 26.25% | 3.99 |
| Dynamic-factor CVaR | 11.22% | 10.17% | 12.13% | 0.937 | -18.68% | 27.23% | 4.75 |

The ordering is no longer attributable to constraints. Relative to matched original MVP, CVaR has 2.41 percentage points lower gross annual return, 2.62 points lower net return, 26 bp higher daily-based annualized volatility, 1.06 points higher annualized ES, a 1.55-point deeper drawdown, and 29% higher turnover. Dynamic MVP is also better on every listed measure.

The full 2017-2026 sample agrees: net returns are 13.98% (original MVP), 14.13% (dynamic MVP), and 12.37% (CVaR); volatilities are 13.81%, 13.89%, and 14.19%.

## Paired dependent-data inference

Inference uses realised monthly losses, 1,999 paired circular moving-block draws, three-month blocks, and percentile 95% intervals. For original MVP minus CVaR, monthly-loss volatility is -0.00862 (CI -0.01522, -0.00219; p=0.010), monthly ES is -0.01061 (CI -0.01944, -0.00139; p=0.031), and annualized net return is +0.02565 (CI 0.00083, 0.05109; p=0.043). Thus matched original MVP is statistically better at 5% on all three registered measures in this sample.

For dynamic MVP minus CVaR, volatility is -0.00793 (CI -0.01518, -0.00095; p=0.028), ES is -0.01242 (CI -0.02120, -0.00145; p=0.036), and net return is +0.02397 (CI -0.00240, 0.05039; p=0.073). Risk differences are distinguishable; the net-return difference is not at 5%.

## Why CVaR loses despite minimizing CVaR

CVaR wins *inside its estimated scenarios*: average scenario CVaR is 1.106%, compared with 1.226% for matched original MVP. Out of sample the ordering reverses: next-month realised CVaR is 1.326% versus 1.295%. CVaR's VaR breach rate is 10.25%, versus 9.21% for original MVP, and its forecast/realised CVaR rank correlation is only 0.314.

This is tail-estimation error: each optimization is driven by roughly 30 nominal tail scenarios, resampled from a rolling historical distribution that does not adequately represent the next month. CVaR intensifies the sampling error into a more concentrated portfolio—12.5 effective assets, 22.5 zero weights, and 4.9 cap hits, versus 14.2, 19.0, and 3.2 for matched original MVP. Higher turnover adds cost, but gross results show it is not the primary cause.

Conclusion: under identical constraints, empirical-bootstrap CVaR genuinely underperforms the covariance MVP after 2021. The evidence supports scenario-tail overfit and calibration failure, not a solver or feasible-set explanation. It remains sample-specific and is not proof of future ordering. Report 06 tests a pre-specified filtered/stressed and regularised repair without altering this baseline.
