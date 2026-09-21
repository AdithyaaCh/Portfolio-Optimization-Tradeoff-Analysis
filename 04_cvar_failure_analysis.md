# 04 - Why CVaR lagged after COVID

> Follow-up: the constraint confound identified here has now been removed. The matched long-only, 10%-cap results in `05_matched_constraint_comparison.md` supersede cross-objective comparisons in this report and strengthen the tail-overfitting diagnosis.

“Post-COVID” is fixed as 2021-01-01 onward. The main finding is more precise than “CVaR failed”: it largely achieved its risk objective, but failed to convert the modest risk reduction into competitive return or Sharpe.

## What happened

| Model, 2021 onward | Gross return | Volatility | ES 95% | Max drawdown | Net return |
|---|---:|---:|---:|---:|---:|
| Equal weight | 20.14% | 13.87% | 32.09% | -15.15% | 20.07% |
| Original TV-MVP | 13.43% | 12.50% | 27.52% | -21.10% | 11.84% |
| Dynamic-factor CVaR | 11.22% | 12.13% | 27.23% | -18.68% | 10.17% |

Against original TV-MVP, CVaR reduced annualized volatility by 37 bp, ES by 29 bp, and drawdown by 2.42 percentage points, but lost 2.21 percentage points of gross annual return. It therefore did not fail as a tail-risk minimizer; it failed on total performance. The gap was concentrated in risk-on years: CVaR returned 3.4% versus 14.4% in 2022 and 10.2% versus 18.7% in 2024. It protected better in 2026 through August (-10.5% versus -17.2%).

## Diagnosed mechanisms

1. **The comparison has different feasible sets.** CVaR is long-only and capped at 10%; original TV-MVP is unconstrained with mean gross exposure 1.82. On the same bootstrap scenarios, D's CVaR was actually 0.105 percentage points *higher* than B's (exploratory three-month-block CI 0.052 to 0.163). D is optimal only inside its capped long-only set, so “CVaR versus B” is not a pure objective-function experiment.

2. **The historical bootstrap underpredicted subsequent tails.** D's mean scenario CVaR was 1.106% per day versus 1.326% realised over the next month. Losses exceeded its forecast 95% VaR on 10.25% of days, about twice the nominal 5%. Forecast-to-realised CVaR rank correlation was only 0.314. Residual excess kurtosis increased from 2.11 before 2021 to 2.77 afterward, while the bootstrap cannot invent shocks absent from its 504-day history.

3. **The optimizer produced corner portfolios.** The 39-stock portfolio held only 12.5 effective assets on average, set 22.5 weights to zero, and placed 4.9 names at the 10% cap each month. HINDUNILVR and NESTLEIND averaged the largest weights. This defensive concentration is consistent with lower risk and weaker participation in broad rallies; it is an interpretation, not causal proof.

4. **CVaR ignores expected return.** Scenario loss minimization selects assets that behaved safely in the historical tail. It has no term rewarding upside participation. The realised next-month mean-return difference from B was -0.0082% per day, although its exploratory block-bootstrap interval (-0.0241%, 0.0084%) includes zero.

5. **Trading costs are not the main explanation.** Annualized cost drag was 1.05% for CVaR versus 1.59% for B because CVaR turnover was lower. Costs actually narrowed, rather than created, the post-COVID return gap.

## Conclusion and next experiment

The initial evidence points to return sacrifice, corner solutions, and tail under-calibration—not an optimization implementation bug. The constraint-matched comparison is completed in Report 05. The pre-specified filtered/stressed-scenario and regularisation experiment is now completed in Report 06.

Supporting outputs: `results/tables/cvar_post_covid_performance.csv`, `cvar_tail_calibration.csv`, `cvar_concentration.csv`, `cvar_mechanism_bootstrap.csv`, `cvar_yearly_performance.csv`, `cvar_asset_allocation.csv`, and `results/plots/cvar_post_covid_diagnostics.png`.
