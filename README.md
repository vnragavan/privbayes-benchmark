# PrivBayes Benchmark: Comprehensive Comparison

A comprehensive benchmarking framework for comparing different PrivBayes implementations with extensive utility and privacy metrics.

## Overview

This repository provides a clean, focused comparison framework for three PrivBayes implementations:
- **Enhanced PrivBayes**: Advanced implementation with QI-linkage reduction features
- **SynthCity PrivBayes**: Implementation from the SynthCity library
- **DPMM PrivBayes**: Implementation from the DPMM library

## Features

### Comprehensive Metrics

**Utility Metrics:**
- Basic: Jaccard similarity, Weighted Jaccard, Marginal error
- TVD (Total Variation Distance): 1D, 2D, 3D marginals
- Mutual Information (MI): Preservation ratios, NMI correlations
- Correlation: Pearson and Spearman correlation preservation
- Coverage: KL divergence, Jaccard coverage
- Downstream ML: Logistic Regression and Random Forest performance

**Privacy Metrics:**
- Exact Row Match Rate (ERMR)
- QI Linkage Rate
- Privacy budget allocation and transparency
- Audit probes (optional but enabled by default in the runner): NN memorization, unique/rare leakage, conditional disclosure
- Membership inference (MIA) distance attack (optional but enabled by default in the runner)

**Performance Metrics:**
- Training time
- Sampling time
- Memory usage

## Project Structure

```
privbayes-benchmark/
├── data/
│   ├── adult.csv              # Adult dataset for benchmarking
│   └── breast_cancer.csv      # (Optional) generated medical dataset for testing
├── schemas/                   # Per-dataset public schema JSONs (public knowledge simulation)
│   ├── adult_public_schema.json
│   ├── breast_cancer_public_schema.json
│   └── README.md
├── external/
│   └── privbayes_enhanced.py  # Enhanced PrivBayes implementation
├── src/
│   └── pbbench/
│       ├── enhanced_metrics.py      # Comprehensive metrics module
│       ├── privacy_audit.py         # Minimum viable audit probes
│       └── variants/
│           ├── pb_enhanced.py        # Enhanced PrivBayes adapter
│           ├── pb_synthcity.py       # SynthCity PrivBayes adapter
│           └── pb_datasynthesizer.py # DPMM PrivBayes adapter
├── scripts/
│   ├── comprehensive_comparison.py              # Run benchmark + write JSON/CSV (+ synthetic CSVs)
│   ├── run_paper_experiments.py                 # Orchestrate paper sweeps + plotting
│   ├── plot_utility_privacy_from_json.py        # One-stop plotting from comprehensive_results_*.json
│   ├── validate_metrics_json.py                 # Sanity-check results JSON structure/ranges
│   ├── verify_install.py                        # Verify environment + imports + datasets
│   ├── prepare_breast_cancer_dataset.py         # Generate data/breast_cancer.csv from sklearn
│   ├── prepare_schema_from_csv.py               # Generate candidate public-schema JSON
│   ├── validate_schema_json.py                  # Validate schema JSON (+ optional dataset cross-check)
│   ├── recompute_qi_linkage.py                  # Maintenance: recompute QI linkage for existing JSONs
│   └── internal/                                # Legacy/debug helpers (not needed for typical runs)
├── docs/                      # Project docs (structure, workflows, results layout)
│   ├── PROJECT_STRUCTURE.md
│   ├── WORKFLOWS.md
│   └── RESULTS_AND_PLOTS.md
├── requirements.txt
└── README.md
```

See:
- `docs/PROJECT_STRUCTURE.md`
- `docs/WORKFLOWS.md`
- `docs/RESULTS_AND_PLOTS.md`
- `scripts/README.md`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd privbayes-benchmark
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional (recommended for development): install the package in editable mode, so imports work
without `sys.path` manipulation:

```bash
python -m pip install -e .
```

3. Verify your installation (recommended):
```bash
python scripts/verify_install.py
```

**Notes:**
- The `pbbench` package uses a `src/` layout and is not installed as a wheel; scripts in this repo (including `verify_install.py`) add `src/` to `sys.path`.
- `dpmm` may not be available on Python 3.12 via PyPI (it declares `Requires-Python <3.12`). If you need the DPMM variant, use Python 3.11, or install it separately with `--ignore-requires-python` at your own risk.

### Dependency / platform notes (important)

- **Python 3.12 compatibility**: Some packages historically lag Python 3.12 wheels. If `pip install -r requirements.txt` fails on your machine, try **Python 3.11** first.
- **DPMM (`dpmm`) on Python 3.12**: PyPI metadata may block install (`Requires-Python <3.12`). Recommended: use **Python 3.11** if you want DPMM without workarounds.
- **SynthCity + PyTorch version constraints**: SynthCity may declare conservative torch constraints in its metadata. On some setups you may hit a runtime error like `torch.nn` missing `RMSNorm` (sometimes mis-remembered as an “RMSProp issue”). If that happens, upgrade PyTorch and reinstall SynthCity without deps (advanced users; see Linux notes below).
- **Optional SynthCity warnings**: You may see warnings like `No module named 'dgl'` from optional plugins. This does **not** affect PrivBayes runs (the PrivBayes plugin still works); it just disables those optional modules.

### Linux install recommendations

For Linux (especially if you want **DPMM**), we recommend **Python 3.11**.

**CPU-only Linux (simplest):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python scripts/verify_install.py
```

**CUDA Linux (recommended if you have an NVIDIA GPU):**

Install PyTorch first using the official PyTorch command for your CUDA version, then install the rest:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 1) Install torch/torchvision using the official PyTorch instructions for your CUDA version.
# 2) Then:
python -m pip install -r requirements.txt
python scripts/verify_install.py
```

**If SynthCity fails at runtime with an RMSNorm error**

If you see an error like:
- `AttributeError: module 'torch.nn' has no attribute 'RMSNorm'`

Use this workaround:

```bash
python -m pip install -U "torch>=2.4,<2.5" "torchvision>=0.19,<0.20"
python -m pip install --no-deps "synthcity==0.2.12"
```

## Usage

First, activate your virtual environment:
```bash
source .venv/bin/activate
```

### Basic Usage

Run a comparison with default settings:
```bash
python scripts/comprehensive_comparison.py \
    --data data/adult.csv \
    --eps 0.5 1.0 2.0 \
    --seeds 0 1 2 \
    --out-dir runs/example_adult \
    --implementations Enhanced SynthCity DPMM
```

Notes:
- `--audit` and `--audit-mia` are **enabled by default**. Disable for faster runs with `--no-audit` and/or `--no-audit-mia`.

### Recommended: Multi-ε trends with uncertainty (multiple seeds)

If you want to interpret trends across ε reliably, run **multiple seeds** per ε and plot mean + error bars.

**Run the benchmark (example: 5 ε values, 5 seeds):**

```bash
python scripts/comprehensive_comparison.py \
  --data data/breast_cancer.csv \
  --eps 0.1 0.5 1.0 3.0 5.0 \
  --seeds 0 1 2 3 4 \
  --out-dir medical_breast_cancer_trend_eps_seeds5 \
  --implementations Enhanced SynthCity DPMM \
  --target-col target \
  --n-bootstrap 0
```

Notes:
- `--n-bootstrap 0` disables internal bootstrap confidence intervals for speed. For trend plots, the most meaningful uncertainty typically comes from **run-to-run variation across seeds**.
- SynthCity can use very large peak memory on some datasets.

**Plot from the saved JSON (mean + uncertainty across seeds):**

```bash
python scripts/plot_utility_privacy_from_json.py medical_breast_cancer_trend_eps_seeds5 \
  --prefix trend_eps_seeds5 \
  --uncertainty ci95
```

This will write:
- `trend_eps_seeds5.png` and `trend_eps_seeds5.pdf` (9-panel utility/privacy)
- `trend_eps_seeds5_performance.png` and `trend_eps_seeds5_performance.pdf` (4-panel performance)
into the output directory (`medical_breast_cancer_trend_eps_seeds5/`).

### Command Line Arguments

- `--data`: Path to the dataset CSV file (required)
- `--eps`: List of epsilon (privacy budget) values to test (default: 0.5, 1.0, 2.0)
- `--seeds`: List of random seeds for reproducibility (default: 0, 1, 2)
- `--out-dir`: Output directory for results (default: results)
- `--implementations`: List of implementations to compare (default: Enhanced, SynthCity, DPMM)
- `--target-col`: Target column name for downstream ML metrics (optional, auto-detected if not provided)
  - Auto-detection searches for common names: `target`, `income`, `label`, `class`, `outcome`
  - Required for computing downstream ML metrics (Logistic Regression and Random Forest performance)
  - If not specified and no common name found, downstream ML metrics will be skipped
- `--n-samples`: Number of rows to generate in synthetic data (optional, default: same as training data size)
  - If not specified, generates the same number of rows as the training dataset
  - Useful for generating larger or smaller synthetic datasets
  - Example: `--n-samples 50000` to generate 50,000 rows regardless of training data size
- `--n-bootstrap`: Number of bootstrap resamples used when computing confidence intervals inside the comprehensive metrics (default: 30)
  - Set `--n-bootstrap 0` to disable CI computation for speed (recommended when you are running many seeds and will estimate uncertainty across seeds instead).

### Example: Quick Test

Run a quick test with a single epsilon and seed:
```bash
python scripts/comprehensive_comparison.py \
    --data data/adult.csv \
    --eps 1.0 \
    --seeds 0 \
    --out-dir quick_test \
    --implementations Enhanced SynthCity
```

### Example: With Target Column

Specify the target column explicitly for downstream ML metrics:
```bash
python scripts/comprehensive_comparison.py \
    --data data/adult.csv \
    --eps 1.0 \
    --seeds 0 \
    --out-dir results \
    --implementations Enhanced \
    --target-col income
```

### Example: Custom Number of Rows

Generate a specific number of rows in the synthetic data:
```bash
python scripts/comprehensive_comparison.py \
    --data data/adult.csv \
    --eps 1.0 \
    --seeds 0 \
    --out-dir results \
    --implementations Enhanced \
    --n-samples 50000
```

This will generate 50,000 rows in the synthetic dataset, regardless of the training data size (which might be 30,162 rows).

**Note:** The target column is automatically detected if not specified. The script searches for common column names (`target`, `income`, `label`, `class`, `outcome`) in your dataset. For the included `adult.csv` dataset, the `income` column is automatically detected.

## Output

The script generates:

1. **JSON Results**: `comprehensive_results_<timestamp>.json`
   - Complete results for all experiments with all metrics
   - Includes paths to saved synthetic datasets

2. **CSV Summary**: `comprehensive_summary_<timestamp>.csv`
   - Aggregated metrics in tabular format for easy analysis

3. **Synthetic Datasets**: `synthetic_<implementation>_eps<epsilon>_seed<seed>.csv`
   - One CSV file per configuration (implementation, epsilon, seed)
   - Contains the full synthetic dataset generated for that configuration
   - Saved in the output directory for further analysis or reuse

4. **Visualizations**: `utility_privacy_plots.png` and `.pdf`
   - 9-panel plot showing:
     - Utility vs Privacy Budget
     - Privacy Budget Efficiency
     - Privacy Risk vs Budget
     - Performance vs Budget
     - Utility-Privacy Tradeoff
     - MI Preservation
     - TVD Metrics
     - Correlation Metrics
     - Downstream ML Performance

## Plotting from JSON (no rerun required)

Use `scripts/plot_utility_privacy_from_json.py` to regenerate plots directly from any `comprehensive_results_*.json` (or a directory containing it).

Examples:

```bash
# Plot from the newest JSON in a directory
python scripts/plot_utility_privacy_from_json.py results_dir --prefix my_plots

# Plot with uncertainty across seeds (requires multiple seeds in the JSON)
python scripts/plot_utility_privacy_from_json.py results_dir --prefix my_plots_ci95 --uncertainty ci95
```

For mostly-numeric datasets (like Breast Cancer), you can use `--numeric-first` to replace overlap-heavy panels
with numeric-fidelity panels (e.g., normalized EMD/Wasserstein).

Note: the plotter will automatically enable numeric-first panels when the results JSON indicates the dataset is
mostly numeric. Use `--no-numeric-first` to override.

## Recommended “paper runs” workflow

Use the orchestrator which runs Adult + Breast Cancer across regimes and generates split figures:

```bash
python scripts/run_paper_experiments.py
```

Notes:
- `--audit` and `--audit-mia` are **enabled by default**.
- Use `--no-audit` and/or `--no-audit-mia` for faster iterations.
- `--split` is **enabled by default**. Use `--no-split` to disable.
- By default it runs **5 seeds** (`--seeds 0 1 2 3 4`). Override with `--seeds ...` if desired.

Examples:

```bash
# Explicit 5-seed run (same as default)
python scripts/run_paper_experiments.py --seeds 0 1 2 3 4

# Faster iteration: keep audit probes, disable MIA
python scripts/run_paper_experiments.py --no-audit-mia

# Change epsilon sweep
python scripts/run_paper_experiments.py --eps 0.1 1 3 5
```

Uncertainty modes:
- `--uncertainty none`: no error bars
- `--uncertainty se`: mean ± standard error across seeds/runs
- `--uncertainty ci95`: mean ± 1.96·SE across seeds/runs (normal approximation)

## Dataset Format

The dataset should be a CSV file with:
- Mixed data types (numeric and categorical)
- A target/label column (optional, for downstream ML metrics)
  - Common names: `target`, `income`, `label`, `class`, `outcome`

The included `adult.csv` dataset is preprocessed and ready to use.

## Per-dataset schema files (public knowledge)

Some configurations model **public side information** about a dataset (e.g., numeric ranges and
categorical domains). We represent this as a **per-dataset schema JSON** consumed by the runner:

```bash
python scripts/comprehensive_comparison.py --data data/adult.csv --schema schemas/adult_public_schema.json ...
```

Schema files live under `schemas/`. See `schemas/README.md` for the expected format and validation
commands.

## Metrics Explained

**Metric Direction Guide:**
- ↑ = Higher is better (more utility/preservation)
- ↓ = Lower is better (less error/distance/privacy risk)

### Basic Utility Metrics

- **Jaccard** ↑: Category coverage similarity (higher = more overlap)
- **Weighted Jaccard** ↑: Frequency-weighted category matching (higher = better preservation)
- **Marginal Error** ↓: L1 distance between marginal distributions (lower = less error)

### Comprehensive Utility Metrics

- **TVD (Total Variation Distance)** ↓: Measures distributional differences at 1D, 2D, and 3D levels (lower = better match)
  - `tvd_1d_mean`: Average 1D marginal TVD
  - `tvd_2d_mean`: Average 2D marginal TVD
  - `tvd_3d_mean`: Average 3D marginal TVD
- **EMD (Earth Mover's Distance)** ↓: Normalized Wasserstein distance for continuous variables (lower = better match)
- **MI (Mutual Information)** ↑: Information preservation between real and synthetic data
  - `mi_preservation_ratio_mean`: Average MI preservation ratio (higher = better)
  - `nmi_spearman`: Spearman correlation of NMI matrices (higher = better structure preservation)
- **Correlation** ↑: Preservation of correlation structure
  - `pearson_spearman`: Spearman correlation of Pearson correlation matrices (higher = better)
  - `spearman_spearman`: Spearman correlation of Spearman correlation matrices (higher = better)
- **Coverage**:
  - `kl_divergence_mean` ↓: Average KL divergence (lower = better distribution match)
  - `jaccard_coverage_mean` ↑: Average Jaccard coverage of value ranges (higher = better coverage)
- **Downstream ML** ↑: Performance of ML models trained on synthetic data
  - `syn2real_lr_auc`: Logistic Regression AUC (higher = better)
  - `syn2real_rf_auc`: Random Forest AUC (higher = better)
  - `symmetry_gap` ↓: Difference between syn→real and real→real performance (lower = better)

### Privacy Metrics

- **ERMR (Exact Row Match Rate)** ↓: Percentage of synthetic rows that exactly match real rows (lower = more privacy)
- **QI Linkage Rate** ↓: Linkage risk based on quasi-identifiers (lower = more privacy)

## Implementation Details

### Enhanced PrivBayes

- Temperature-based sampling for QI-linkage reduction
- Label column handling (no hashing, no UNK)
- Public category support
- Advanced epsilon allocation

### SynthCity PrivBayes

- Uses SynthCity library's PrivBayes plugin
- Automatic parameter tuning
- Memory-optimized for large datasets

### DPMM PrivBayes

- Uses DPMM library's PrivBayes pipeline
- Automatic discretization
- Bayesian network structure learning


