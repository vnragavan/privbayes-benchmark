## Workflows

### 1) Verify install

```bash
source .venv/bin/activate
python scripts/verify_install.py
```

### 2) Run a single configuration (quick sanity)

```bash
python scripts/comprehensive_comparison.py \
  --data data/breast_cancer.csv \
  --eps 1.0 \
  --seeds 0 \
  --out-dir runs/quick_smoke \
  --implementations Enhanced SynthCity DPMM \
  --schema schemas/breast_cancer_public_schema.json \
  --n-bootstrap 0
```

### 3) Multi-ε sweep (paper-style) + split plots

```bash
python scripts/run_paper_experiments.py \
  --eps 0.1 1 3 5 \
  --seeds 0
```

Outputs go under `--out-root` (default: `paper_runs/` unless changed).

Notes:
- `--audit` and `--audit-mia` are **enabled by default**. Use `--no-audit` and/or
  `--no-audit-mia` to disable for faster runs.
- `--split` is **enabled by default**. Use `--no-split` to disable.

### 4) Plot from an existing JSON (no re-run required)

```bash
python scripts/plot_utility_privacy_from_json.py runs/quick_smoke \
  --split \
  --uncertainty none \
  --prefix quick_smoke
```

### 5) Schema simulation (public metadata)

Generate a candidate schema from any CSV:

```bash
python scripts/prepare_schema_from_csv.py \
  --csv data/breast_cancer.csv \
  --out schemas/breast_cancer_public_schema.json \
  --target-col target
```

Validate a schema:

```bash
python scripts/validate_schema_json.py \
  --schema schemas/breast_cancer_public_schema.json \
  --csv data/breast_cancer.csv
```

