# 06 - Improved CVaR analysis

## What was wrong, and what changed

The matched-constraint experiment showed that the failure was not caused only by CVaR being long-only and capped. Baseline CVaR optimised a noisy 5% empirical tail: 600 scenarios leave only about 30 tail draws, with fewer effectively independent observations after block sampling. Optimisation then exploits sampling error. It selected about 4.9 cap-bound and 22.5 zero weights per rebalance, yet its fitted daily CVaR (1.106%) understated subsequent holding-period realised daily CVaR (1.326%). Its 10.25% VaR exceedance rate was roughly twice the nominal 5%. CVaR also controls loss, not expected return, and high turnover made estimation changes costly.

The repair was specified before viewing its results and leaves baseline CVaR untouched. Factor and residual shocks are whitened and rescaled to current EWMA conditions; circular block starts overweight recent observations; 15% of scenarios receive 1.50x factor and 1.15x residual shocks, increasing volatility and common-factor correlation. A second version adds L1 penalties for turnover and distance from equal weight. Every portfolio remains long-only, sums to one, and has the same 10% cap.

## Post-2021 return and risk comparison

All returns below are annualised; transaction costs are 10 bp per one-way traded notional.

| Model | Gross return | Net return | Volatility | Sharpe | Max drawdown | Daily ES 95% | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original covariance MVP, matched | 13.62% | 12.79% | 11.86% | 1.136 | -17.13% | 26.17% | 3.69 |
| Empirical CVaR, matched | 11.22% | 10.17% | 12.13% | 0.937 | -18.68% | 27.23% | 4.75 |
| Filtered/stressed CVaR, matched | 13.92% | 12.88% | 12.10% | 1.138 | -16.94% | 27.00% | 4.60 |
| Regularised filtered CVaR, matched | 15.10% | 14.30% | 11.95% | 1.237 | -15.67% | 26.56% | 3.50 |

Relative to baseline empirical CVaR, filtering/stressing adds 2.71 percentage points of net return; regularisation adds another 1.42 points and removes 1.10 annual turnover. Calibration also changes materially: regularised CVaR's mean fitted/realised daily CVaR is 1.282%/1.287%, its exceedance rate falls to 8.14%, and its effective asset count rises modestly from 12.52 to 12.56. The exceedance rate remains above 5%, so the repair is incomplete.

The measure-based conclusion depends on frequency. Daily metrics favour regularised CVaR over original MVP on return, Sharpe, drawdown, and turnover, but not volatility or ES. Required inference uses realised monthly loss series. Original-minus-regularised differences are -0.45 percentage point annualised monthly volatility (95% CI -1.18, 0.26; p=0.221), +0.26 point monthly ES (CI -0.51, 0.90; p=0.497), and -1.47 points annualised net return (CI -4.28, 1.15; p=0.275). None is statistically distinguishable from zero.

For the pre-specified empirical-versus-filtered comparison, empirical CVaR has +0.47 point annualised monthly volatility (CI -0.46, 1.43; p=0.315), +1.41 points monthly ES (CI -0.08, 2.55; p=0.071), and -2.65 points annualised net return (CI -5.92, 0.36; p=0.097). These are suggestive, not 5%-significant. Exploratory empirical-versus-regularised intervals exclude zero for monthly ES and net return, but receive no p-values because that pair was not pre-specified.

## Regimes and conclusion

In high-volatility observations, regularised CVaR has 16.13% volatility and 32.83% ES versus baseline CVaR's 16.05% and 33.69%; in normal observations it has 10.41% and 23.76% versus 10.71% and 24.91%. The 1,999-draw, 21-day-block interaction test does not show a reliably larger high-volatility benefit: volatility interaction -0.38 point (p=0.234), ES -0.29 point (p=0.740), and net-return interaction +1.55 points (p=0.665).

So there was nothing inherently “wrong” with the CVaR objective. The baseline failure came from tail-scenario miscalibration, small effective tail samples, unstable optimisation, and trading—not from the cap comparison. The repair improves this one backtest substantially, but does not beat matched MVP with statistical confidence and still trails equal weight. Survivorship-biased current constituents, Yahoo data, a single market/sample, simplified costs, noisy realised ES, and researcher choices remain. Backtest significance would not prove future profitability; here, even that threshold is not met.

Evidence: `results/tables/improved_cvar_post_covid_performance.csv`, `improved_cvar_post_covid_inference.csv`, `improved_cvar_tail_calibration.csv`, `improved_cvar_regime_performance.csv`, and `improved_cvar_regime_interaction.csv`. Plot: `results/plots/improved_cvar_diagnostics.png`.
