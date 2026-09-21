# 01 - Reproduction report

## Target and audit

The supplied research artifact is a 16-page user manual plus MATLAB/R code, not the journal paper's empirical dataset. The manual describes local Epanechnikov-kernel PCA, a constant sparse idiosyncratic covariance, an unconstrained minimum-variance portfolio, and a weekly 50-stock example. The full paper is Fan, Wu, Yang, and Zhong (2024), *Journal of Econometrics* 239(2), 105339, DOI `10.1016/j.jeconom.2022.08.007`.

All 26 supplied files were inspected before implementation. They are preserved byte-for-byte under `original_author_code/` (pre/post relocation SHA-256 hashes matched). Important execution dependencies are MATLAB CVX and Windows calls to R packages `spcov`, `PDSCE`, and `R.matlab`; paths are hard-coded in `Rspcov.bat` and `spcov_test.R`. Those dependencies were unavailable as a working original stack here. `example_data.mat` contains a 50 x 500 return matrix with 1.4% missing observations.

## What was reproduced

The Python reproduction ports the manual's simulated two-factor DGP, rule-of-thumb bandwidth, boundary-aware local PCA, endpoint covariance decomposition, and unconstrained MVP. A diagonal-preserving soft-threshold covariance followed by positive-definite projection is the portable substitute for the unavailable penalised quasi-likelihood R step. NumPy and MATLAB do not share random-number streams, so matching a seed does not reproduce identical draws.

| Quantity | Manual | Portable result |
|---|---:|---:|
| Selected factor count | 2 | 2 |
| Oracle portfolio variance | 0.1373 | 0.2270 |
| TV-MVP forecast variance | 0.1000 | 0.2426 |
| Sample-covariance forecast variance | 0.2117 | 0.3190 |
| Weekly TV-MVP daily Sharpe | 0.0865 | 0.0378 |
| Weekly equal-weight daily Sharpe | 0.0488 | 0.0495 |
| Weekly TV-MVP mean / stdev | 0.0006 / 0.0075 | 0.00024 / 0.00636 |

The factor count and equal-weight weekly statistics match closely. The covariance risks and TV-MVP weekly Sharpe do not. The portable TV-MVP weight mean absolute error from the simulated oracle is 0.0431, versus 0.0758 for the sample-covariance MVP, so the qualitative weight result in the manual holds for this draw; this is not an exact table replication.

## Reasons for differences

1. The original `spcov`/PDSCE estimator and its cross-validated penalty were replaced, changing the residual covariance materially (relative Frobenius error 0.482 in the portable draw).
2. Random streams, eigensolver conventions, and optimization differ between MATLAB and Python.
3. The manual's reported risk values are forecast variances under three different covariance matrices, not a common realised-risk comparison.
4. The author weekly routine changes the asset universe after observing missing returns during a future holding interval and does not renormalize remaining weights. The portable reproduction fills missing example returns with zero but never mutates the universe; it therefore avoids copying that timing issue.

Full values are in `results/tables/reproduction_results.csv`; diagnostic figures are `results/plots/reproduction_factor_loadings.png` and `results/plots/reproduction_residual_covariance.png`. This is a close methodological reproduction, not a claim of bitwise replication.

