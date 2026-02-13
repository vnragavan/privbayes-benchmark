## Project structure (source of truth)

This repository is organized around a simple contract:

- **Runner** writes `comprehensive_results_*.json` (plus a CSV summary and synthetic CSVs).
- **Plotter** reads those JSONs and generates paper-ready figures.
- **Schemas** model “public side information” used by default/non-strict regimes.

### Key directories

- `src/pbbench/`
  - **Library code** (metrics + audit probes + implementation adapters)
  - `enhanced_metrics.py`: comprehensive utility metrics (TVD/EMD, MI, corr, coverage, downstream ML)
  - `privacy_audit.py`: minimum viable audit probes (NN memorization, rare/unique leakage, conditional disclosure, MIA)
  - `variants/`: adapters for Enhanced / SynthCity / DPMM

- `scripts/`
  - **Entry points** (run, validate, plot)
  - See `scripts/README.md` for a canonical list

- `schemas/`
  - Per-dataset public schemas (bounds + categorical domains + label domain)
  - See `schemas/README.md`

- `data/`
  - Input datasets (`adult.csv`, `breast_cancer.csv`, optional others)

- Output directories (not meant to be committed)
  - `results*/`, `medical_*/`, `paper_runs*/`, `runs*/`

### “One correct workflow”

1. Run experiments with `scripts/comprehensive_comparison.py` (or `scripts/run_paper_experiments.py`).
2. Validate JSON sanity (optional): `scripts/validate_metrics_json.py`.
3. Plot from JSON: `scripts/plot_utility_privacy_from_json.py` (optionally `--split`, `--uncertainty`, `--numeric-first`).

