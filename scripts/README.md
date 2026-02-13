## Scripts (canonical entry points)

This folder contains the **recommended commands** for running experiments and generating plots.

### Run benchmarks

- `comprehensive_comparison.py`
  - Main runner. Produces `comprehensive_results_*.json`, summary CSV, synthetic CSVs.
  - Supports regimes (`default` vs `strict`) and public schemas (`--schema`).
  - Supports audit probes (`--audit`, `--audit-mia`).

- `run_paper_experiments.py`
  - Convenience orchestrator: runs Adult + Breast Cancer (and optional lung dataset),
    runs both regimes (where applicable), then generates split figures.

### Plot results

- `plot_utility_privacy_from_json.py`
  - One-stop plotting from `comprehensive_results_*.json` (or a directory containing it).
  - Supports `--split`, `--uncertainty`, and `--numeric-first`.

### Validate / sanity check

- `verify_install.py`
  - Quick import + environment sanity checks.

- `validate_metrics_json.py`
  - Validates `comprehensive_results_*.json` schema and sanity of metric values.

### Schema tooling (public metadata simulation)

- `prepare_schema_from_csv.py`
  - Generates a candidate schema (bounds, categories, label domain) from a CSV.

- `validate_schema_json.py`
  - Validates a schema JSON and can cross-check against a dataset CSV.

### Metric backfills / maintenance

- `recompute_qi_linkage.py`
  - Recomputes QI-linkage rates for an existing results JSON using saved synthetic CSVs.

### Internal / legacy

- `internal/_smoke_dpmm_adapter.py`
  - Internal smoke test for the DPMM adapter (kept for debugging macOS multiprocessing issues).

- `internal/augment_results_json_with_downstream.py`
  - Legacy helper to backfill downstream ML metrics into an older JSON. Prefer re-running with the
    current runner when feasible.

