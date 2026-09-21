# 03 - Results report

## NIFTY study

The frozen adjusted-close sample retains 39 liquid, long-history stocks from a current NIFTY 50 list. Out-of-sample results cover 2,362 trading days (2017-02-01 to 2026-08-31), 115 rebalances, and a 10 bp one-way cost.

| Model | Return | Volatility | Sharpe | Max DD | ES 95% | Turnover | Net return |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal weight | 19.81% | 15.94% | 1.215 | -35.79% | 36.86% | 0.31 | 19.74% |
| Original TV-MVP | 15.88% | 14.37% | 1.098 | -30.04% | 31.81% | 6.63 | 14.35% |
| Dynamic SigmaF TV-MVP | 15.74% | 14.36% | 1.091 | -30.34% | 31.81% | 6.63 | 14.22% |
| Dynamic-factor CVaR | 13.44% | 14.19% | 0.960 | -34.37% | 31.65% | 4.73 | 12.37% |

Lower CVaR volatility and ES are small; equal weight has the highest return/Sharpe. No extension earns an improvement claim. Average next-month covariance Frobenius losses are 0.00562 (B), 0.00596 (C), and 0.00557 (D scenario covariance); portfolio-risk losses are 0.000057, 0.000058, and 0.000052.

High-volatility observations have much greater risk. B/C annualized volatility is 22.30%/22.30% and ES 53.16%/53.13%; D is 21.75% and 50.72%. In normal periods, B/C volatility is 10.79%/10.78%; D is 10.82%. These conditional differences are descriptive.

Diagnostics show three factors explain 48.9% on average. Consecutive loading spaces are unstable (mean maximum principal angle 54.5 degrees). Only 14.5% of off-diagonal residual correlations are thresholded to zero. Mean factor and median residual excess kurtosis are 2.43 and 2.49, consistent with heavy tails; QQ plots confirm tail departures without relying on large-sample normality-test p-values.

## Stress tests and robustness

Across 100 independent replications per DGP, C beats B on realised variance with probabilities 0.48 (Gaussian), 0.54 (Student-t), and 0.21 (regime shift); corresponding ES probabilities are 0.47, 0.60, and 0.19. D beats B on variance/ES with probabilities 0.27/0.31, 0.22/0.29, and 0.64/0.62. True-covariance Frobenius error for C is not lower than B in any setting. The deliberately abrupt shift exposes a sensitivity failure of `lambda=0.94`; the CVaR portfolio adapts more often there, but not reliably elsewhere.

Robustness from 2021 onward also fails to rescue a general claim. C-minus-B Sharpe ranges from -0.0187 to -0.0339 across EWMA decay, -0.0209 to -0.0461 across windows, and -0.0290 to -0.0422 across factor counts. D-minus-B Sharpe ranges from -0.133 at 95% CVaR to -0.0049 at 99%, while ES changes sign at 99%. Transaction costs monotonically worsen the high-turnover dynamic strategy's net-return gap.

## Statistical evidence and limitations

Paired monthly comparisons use 1,999 circular moving-block bootstrap draws, three-month blocks, percentile 95% confidence intervals, and two-sided bootstrap p-values for the two pre-specified pairs only. B-minus-C differences are 0.00038 volatility (CI -0.00198, 0.00238; p=0.755), 0.00198 monthly ES (CI -0.00580, 0.00901; p=0.434), and 0.00131 net return (CI -0.00766, 0.01089; p=0.781). C-minus-D differences are -0.00701 volatility (CI -0.02382, 0.01072; p=0.498), -0.01492 ES (CI -0.03600, 0.00183; p=0.119), and 0.01804 net return (CI -0.01883, 0.05471; p=0.358). None is statistically distinguishable from zero.

The 21-day-block high-minus-normal benefit test also finds no evidence that C helps more in high volatility: volatility interaction -0.00018 (CI -0.00404, 0.00396; p=0.906) and ES interaction 0.00103 (CI -0.01585, 0.01734; p=0.975).

Limitations are material: current constituents create survivorship bias; Yahoo adjusted-close data and one failed ticker limit provenance; no risk-free series is used; costs exclude impact, taxes, borrowing, and shorting costs; B/C allow shorts while D does not; realised monthly covariance is noisy; ES samples are small; overlapping estimation windows and researcher choices remain. Parameter sweeps are robustness checks, not proof. Backtest significance—even if present—would not prove future profitability because regimes, membership, liquidity, costs, and model fit can change.

A pre-specified follow-up repairs CVaR's scenario calibration and turnover rather than changing its constraint set. Post-2021, regularised filtered CVaR raises net return from 10.17% to 14.30%, reduces annual turnover from 4.75 to 3.50, and narrows mean scenario-versus-realised daily CVaR error from 0.00219 to 0.00005. Its differences from matched original MVP are nevertheless not statistically distinguishable at 5%. This extension is documented in `06_improved_cvar_analysis.md`; it does not overturn the limitations above.

Plots: `results/plots/cumulative_net_returns.png`, `rolling_volatility_es.png`, `weights_turnover_distribution.png`, `factor_diagnostics.png`, `factor_residual_qq.png`, `simulation_stress_summary.png`, `sensitivity_heatmap.png`, and `improved_cvar_diagnostics.png`.
