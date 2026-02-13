## Results artifacts and how to read them

Each run of `scripts/comprehensive_comparison.py` writes:

- **Results JSON**: `comprehensive_results_<timestamp>.json`
  - One JSON object per (implementation, regime, ε, seed)
  - Contains utility, privacy, audit probes, performance, and `synthetic_data_path`

- **Summary CSV**: `comprehensive_summary_<timestamp>.csv`
  - A compact table of averaged metrics per implementation (good for quick checks)

- **Synthetic CSVs**: `synthetic_<implementation>_eps<ε>_seed<seed>.csv`
  - Saved so you can recompute metrics or audit probes without re-synthesizing

### Plotting

Use the plotter on a JSON file or a directory containing it:

```bash
python scripts/plot_utility_privacy_from_json.py <results_dir_or_json> --split --uncertainty ci95
```

The plotter supports:
- `--split`: writes **utility/fidelity**, **privacy**, and **performance** as separate figures
- `--uncertainty {none,se,ci95}`: uncertainty across seeds per (implementation, ε)
- `--numeric-first`: for mostly-numeric datasets, swaps overlap-heavy panels for numeric-fidelity panels

### A note on “conditional disclosure (L1)”

This audit probe reports a **conditional distribution gap**:
average \(\|P(S\mid Q=q) - \hat{P}(S\mid Q=q)\|_1\) weighted by \(P_{\text{real}}(Q=q)\).

It is in \([0,2]\) and **lower is better**, but values near ~1 can happen when many real QI patterns
are absent in the synthetic (a support/coverage issue), especially with binned numeric QIs.

