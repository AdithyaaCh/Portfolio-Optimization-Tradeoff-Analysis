# Implementation plan

## Scope and invariants

- Preserve every author-supplied file byte-for-byte under `original_author_code/`. The repository root becomes the clean interface for this independent project.
- Use Python 3 for the extension because the supplied MATLAB benchmark depends on CVX, a Windows `.bat` file, and R packages (`spcov`, `PDSCE`, `R.matlab`) that are not portable in this environment. The port follows the equations and algorithm in `Manual.pdf`; all departures below are explicit.
- Fix all pseudo-random seeds in `config.yaml`. A run manifest will record configuration, data hashes, package versions, and timestamps.
- Use returns available strictly before each rebalance. Monthly weights are held through the next rebalance interval.

## Repository layout and files to create

- `original_author_code/*` - all 26 supplied author files, relocated without content changes.
- `README.md` - concise setup, provenance, and run instructions for the independent project.
- `config.yaml` - the single source of all experiment, data, model, inference, and plotting parameters.
- `requirements.txt` and `pyproject.toml` - Python dependencies and package/test configuration.
- `scripts/run_all.py`, `scripts/run_reproduction.py`, `scripts/run_empirical.py`, `scripts/run_cvar_diagnostics.py`, `scripts/run_fair_comparison.py`, `scripts/run_improved_cvar.py`, `scripts/run_simulations.py` - command-line entry points.
- `src/tvmvp_ext/__init__.py`
- `src/tvmvp_ext/config.py` - validated configuration loading.
- `src/tvmvp_ext/data.py` - author `.mat` loading, NIFTY download/cache, adjusted-close returns, and monthly rebalance calendar.
- `src/tvmvp_ext/reproduction.py` - portable author-DGP and supplied-example reproduction workflow.
- `src/tvmvp_ext/empirical.py` - empirical, inference, diagnostics, and sensitivity orchestration.
- `src/tvmvp_ext/cvar_diagnostics.py` - post-COVID tail calibration, concentration, and failure-channel analysis.
- `src/tvmvp_ext/fair_comparison.py` - matched-constraint return, risk, calibration, and bootstrap comparison.
- `src/tvmvp_ext/improved_cvar.py` - pre-specified filtered/stressed-scenario and regularisation ablation.
- `src/tvmvp_ext/factor.py` - endpoint and full-window local PCA, factor scores, residuals, explained variance, and loading-space principal angles.
- `src/tvmvp_ext/covariance.py` - benchmark constant factor covariance, EWMA dynamic factor covariance, sparse residual covariance, and positive-definite repair.
- `src/tvmvp_ext/optimisation.py` - equal-weight, minimum-variance, and Rockafellar-Uryasev empirical CVaR optimisers.
- `src/tvmvp_ext/backtest.py` - common-information-set rolling backtest and covariance forecast diagnostics.
- `src/tvmvp_ext/metrics.py` - performance, loss, turnover, costs, drawdown, rolling volatility/ES, and regime metrics.
- `src/tvmvp_ext/inference.py` - moving-block bootstrap confidence intervals, p-values, pairwise tables, and high-vs-low regime interaction test.
- `src/tvmvp_ext/simulation.py` - Gaussian, Student-t, and volatility/correlation-shift DGPs with known covariance.
- `src/tvmvp_ext/plotting.py` - all publication-ready plots and compact sensitivity heatmap.
- `src/tvmvp_ext/manifest.py` - environment, source-control, seed, and configuration audit record.
- `tests/test_core.py` - unit tests for timing, covariance repair, constraints, metrics, and bootstrap reproducibility.
- `data/raw/nifty50_adjusted_close.csv` - cached adjusted-close snapshot used by the empirical run.
- `data/raw/nifty50_index_adjusted_close.csv` and `data/data_manifest.json` - index snapshot, tickers, source, download date, hashes, missingness, and survivorship warning.
- `results/tables/*.csv` - reproduction, empirical, inference, diagnostics, simulation, and sensitivity tables.
- `results/plots/*.png` - cumulative/net performance, rolling risk, weights/turnover, diagnostics/QQ, simulation, and sensitivity plots.
- `results/run_manifest.json` - reproducibility metadata.
- `01_reproduction_report.md`, `02_extension_report.md`, `03_results_report.md`, `PROJECT_SUMMARY.md` - requested concise reports.
- `04_cvar_failure_analysis.md` - focused post-COVID tail-calibration and failure-mechanism investigation added in follow-up.
- `05_matched_constraint_comparison.md` - fair long-only, 10%-cap comparison that removes the MVP/CVaR feasible-set confound.
- `06_improved_cvar_analysis.md` - measured assessment of the CVaR repair, including uncertainty and regime results.

No author MATLAB/R source, data, helper output, or `Manual.pdf` will be edited; the original README will be retained as `original_author_code/README.md`.

## Model implementation

1. Reproduce the manual's simulated DGP and local Epanechnikov-kernel PCA. At a rebalance endpoint, estimate time-varying loadings throughout the trailing window, regress each contemporaneous return on its local loading, and retain the endpoint loading matrix.
2. Estimate a constant sparse residual covariance over the trailing window using diagonal-preserving soft thresholding, followed by eigenvalue clipping. This is the documented portable substitute for the unavailable `spcov`/PDSCE penalised quasi-likelihood implementation.
3. Model B uses the full-window sample covariance of estimated factor scores, matching the author formula `B_t Sigma_F B_t' + Sigma_e`.
4. Model C replaces only `Sigma_F` with a normalised EWMA covariance of demeaned factor scores, `lambda^(T-1-t)`, and leaves the loading/residual portions unchanged.
5. Every covariance is symmetrised and checked by Cholesky/eigenvalues. If needed, eigenvalues below the configured floor are clipped and the intervention is recorded.

## CVaR scenario generation

At each rebalance, use only the trailing estimation window. Form paired observations `(f_u, e_u)` from the estimated factor scores and residual vectors. Draw a configured number of scenarios with a circular moving-block bootstrap of configured block length; sampling paired blocks preserves serial dependence and contemporaneous factor-residual dependence. Map each draw through the current endpoint loading:

`r^(s) = B_t f^(s) + e^(s)`.

Optimise empirical 95% loss CVaR with the Rockafellar-Uryasev linear program, subject to weights summing to one, non-negative weights, and the configured optional per-asset cap (default 10%). This bootstrap is conditional on the fitted local factor representation; it does not extrapolate unseen tail events.

### Pre-specified CVaR repair added after the failure diagnosis

Before rerunning results, the follow-up experiment fixed three models: the unchanged empirical CVaR baseline; filtered/stressed CVaR; and the same filtered/stressed CVaR with regularisation. Historical factor and residual shocks are separately demeaned and whitened, then rescaled by current EWMA factor covariance and a 50/50 blend of EWMA and sparse residual covariance. Circular blocks retain serial and factor-residual dependence, while start probabilities decay by 0.995 per lag. A fixed 15% stress mixture multiplies factor shocks by 1.50 and residual shocks by 1.15, deliberately increasing common-factor risk and therefore conditional correlation. The final LP adds fixed L1 penalties of 0.0010 on distance from pre-trade weights and 0.00025 on distance from equal weight. It retains the 95% objective, long-only constraint, and 10% cap. These settings were recorded in `config.yaml` before the improved run and were not tuned after inspecting performance.

The two pre-specified improved-run comparisons are empirical versus filtered/stressed CVaR and matched original MVP versus regularised filtered CVaR. All other pairwise intervals and the intermediate ablation are exploratory. The high-minus-normal regime interaction was also fixed as empirical versus regularised filtered CVaR.

## Statistical analysis

- Aggregate daily realised gross/net returns to non-overlapping monthly returns. Define loss as minus monthly return.
- Use a circular moving-block bootstrap on paired monthly strategy vectors. The primary hypotheses are fixed before inspecting results: B vs C and C vs D. Report two-sided bootstrap p-values and percentile 95% intervals for differences in annualised volatility, 95% ES, and annualised net return. Other pairwise intervals are descriptive and carry no p-values.
- Define high-volatility days ex ante from trailing NIFTY-index realised volatility above its expanding 75th percentile, shifted one day. Report high/normal metrics and bootstrap the difference-in-benefit `(B-C)_high - (B-C)_normal`.
- Compare each covariance forecast with the next holding-period realised covariance using Frobenius loss and absolute portfolio-risk loss. In simulations, also compare with the known true covariance.
- Diagnose factor explained variance, consecutive loading-subspace principal angles, residual-correlation sparsity, excess kurtosis, and QQ plots. Normality-test p-values will not be used as substantive evidence.
- Run at least 100 independent replications for Gaussian, Student-t, and regime-shift DGPs. Report mean, standard deviation, Monte Carlo 95% intervals, and probabilities of beating Model B on realised variance and ES.
- Treat EWMA decay, factor count, window length, CVaR level, and transaction cost sweeps as robustness only and summarise them in compact tables/heatmaps.

## Departures from the paper/manual

- The full paper's proprietary S&P/CRSP-style empirical data are not supplied. Reproduction targets the public manual examples and supplied `original_author_code/example_data.mat`; the new empirical application uses a frozen Yahoo Finance adjusted-close snapshot for a selected current-constituent NIFTY universe and explicitly carries survivorship bias.
- Sparse residual covariance uses portable soft thresholding plus positive-definite projection instead of the Windows-only R `spcov`/PDSCE pipeline.
- The paper's unconstrained MVP is retained for Models B/C. Model D is long-only and capped by default as requested, so differences combine objective and feasible-set effects; a no-cap CVaR sensitivity is retained.
- Monthly rebalancing replaces the manual's weekly demonstration for the NIFTY study. Simulation sizes are chosen for reproducibility and computational feasibility and are not claimed to replicate every paper table.
