# TV-MVP reproduction and dynamic-risk extensions

This is an independent, reproducible research project built around Fan, Wu, Yang, and Zhong's time-varying minimum-variance portfolio (TV-MVP). The untouched author release is retained under `original_author_code/`. New code reproduces the documented local-PCA benchmark and compares it with dynamic EWMA factor covariance and an empirical-scenario CVaR portfolio.

The complete 34-page technical report is available as `output/pdf/TV_MVP_research_report.pdf`; editable LaTeX source is in `report/TV_MVP_research_report.tex`.

## Run

Use Python 3.10+ from this directory:

```bash
python3 -m pip install -e .
python3 scripts/run_all.py
```

Focused commands are `python3 scripts/run_reproduction.py`, `python3 scripts/run_empirical.py --with-sensitivity`, `python3 scripts/run_cvar_diagnostics.py`, `python3 scripts/run_fair_comparison.py`, `python3 scripts/run_improved_cvar.py`, and `python3 scripts/run_simulations.py`. The fair-comparison command gives MVP and CVaR identical long-only 10%-cap constraints. The improved-CVaR command adds a pre-specified filtered-historical/stress scenario model and an L1-regularised version without replacing the baseline. Add `--force-download` to refresh Yahoo Finance prices; otherwise the frozen files in `data/raw/` are used. All key choices—including seed, dates, assets, window, factor count, EWMA decay, CVaR settings, bootstrap draws, simulation replications, and costs—live in `config.yaml`.

Outputs are written to `results/tables/` and `results/plots/`. Start with `PROJECT_SUMMARY.md`, then see the numbered reports; `05_matched_constraint_comparison.md` removes the constraint confound and `06_improved_cvar_analysis.md` tests the proposed repair. `IMPLEMENTATION_PLAN.md` records methods and departures before implementation.

## Interpretation guardrails

Models B/C use the author's unconstrained MVP; Model D is long-only and capped at 10% by default. The portable benchmark replaces the author's unavailable Windows-only `spcov`/PDSCE step with documented soft thresholding plus positive-definite projection. The NIFTY universe is a long-history subset of current constituents, so it has survivorship bias. Results are research evidence, not investment advice or proof of future profitability.
