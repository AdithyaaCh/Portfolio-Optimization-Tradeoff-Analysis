# TV-MVP portfolio tradeoffs: an empirical analysis

This is an independent, reproducible empirical study of Fan, Wu, Yang, and Zhong's time-varying minimum-variance portfolio (TV-MVP). It examines how market co-movement, portfolio constraints, concentration, tail risk, and trading costs shaped realized outcomes across 39 NIFTY-stock histories. The main deliverable is **[ANALYTICAL_STUDY.md](ANALYTICAL_STUDY.md)**, with regenerable data tables and figures in `results/analysis/`. The repository retains only the frozen inputs and strategy paths required to reproduce that synthesis.

The complete 34-page technical report is available as `output/pdf/TV_MVP_research_report.pdf`; editable LaTeX source is in `report/TV_MVP_research_report.tex`.

## Run

Use Python 3.10+ from this directory. To reproduce the analytical synthesis from frozen prices and saved strategy paths:

```bash
python3 -m pip install -e .
python3 scripts/run_analytical_study.py
```

The synthesis reads the exact retained strategy paths under `results/tables/`, validates cost and weight identities, and writes its outputs to `results/analysis/`. It does not download new prices or search for new strategies.

## Interpretation guardrails

The main portfolio comparison uses identical long-only 10%-cap constraints. The 39-stock universe is a long-history subset of later NIFTY constituents and has survivorship bias. A sign-invariance audit invalidated a prior downside-factor experiment as evidence of improved tail protection; see the analytical report. Results are exploratory research evidence, not proof of future profitability.
