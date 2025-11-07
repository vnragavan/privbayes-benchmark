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

**Performance Metrics:**
- Training time
- Sampling time
- Memory usage

## Project Structure

```
privbayes-benchmark/
├── data/
│   └── adult.csv              # Adult dataset for benchmarking
├── external/
│   └── privbayes_enhanced.py  # Enhanced PrivBayes implementation
├── src/
│   └── pbbench/
│       ├── enhanced_metrics.py      # Comprehensive metrics module
│       └── variants/
│           ├── pb_enhanced.py        # Enhanced PrivBayes adapter
│           ├── pb_synthcity.py       # SynthCity PrivBayes adapter
│           └── pb_datasynthesizer.py # DPMM PrivBayes adapter
├── scripts/
│   └── comprehensive_comparison.py  # Main comparison script
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd privbayes-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run a comparison with default settings:
```bash
python scripts/comprehensive_comparison.py \
    --data data/adult.csv \
    --eps 0.5 1.0 2.0 \
    --seeds 0 1 2 \
    --out-dir results \
    --implementations Enhanced SynthCity DPMM
```

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

**Note:** The target column is automatically detected if not specified. The script searches for common column names (`target`, `income`, `label`, `class`, `outcome`) in your dataset. For the included `adult.csv` dataset, the `income` column is automatically detected.

## Output

The script generates:

1. **JSON Results**: `comprehensive_results_<timestamp>.json`
   - Complete results for all experiments with all metrics

2. **CSV Summary**: `comprehensive_summary_<timestamp>.csv`
   - Aggregated metrics in tabular format for easy analysis

3. **Visualizations**: `utility_privacy_plots.png` and `.pdf`
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

## Dataset Format

The dataset should be a CSV file with:
- Mixed data types (numeric and categorical)
- A target/label column (optional, for downstream ML metrics)
  - Common names: `target`, `income`, `label`, `class`, `outcome`

The included `adult.csv` dataset is preprocessed and ready to use.

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


