## Dataset schema files (public knowledge)

This folder contains **per-dataset schema artifacts** that represent **public side information**
available *before* running experiments.

These files are used by the benchmark runner via `--schema` and are intended to model practical
deployments where some schema metadata is public (documentation, data dictionaries, measurement
instrument ranges, clinically plausible ranges, etc.).

### Why we separate schemas from experiments

If numeric bounds or categorical domains are computed from private data (e.g., raw min/max or
`unique()` values), the pipeline is **not** end-to-end DP unless that metadata is released with a
DP mechanism and budgeted. Treating schema files as inputs makes it explicit what information is
assumed public.

### Schema JSON format

Minimal required keys:

- **`dataset`**: string dataset identifier (e.g., `"adult"`)
- **`target_col`**: string (or `null`) target/label column name
- **`label_domain`**: list of allowed label values (recommended: strings)
- **`public_bounds`**: object mapping `col -> [L, U]` for numeric columns
- **`public_categories`**: object mapping `col -> [cat1, cat2, ...]` for categorical columns

Optional keys:

- **`provenance`**: object with notes about how the schema was produced and why it is considered public

### Validation

Use the validator to check schemas (and optionally cross-check against a dataset CSV):

```bash
python scripts/validate_schema_json.py --schema schemas/adult_public_schema.json
python scripts/validate_schema_json.py --schema schemas/adult_public_schema.json --data data/adult.csv
```

### Generating a *candidate* schema from a CSV

`scripts/prepare_schema_from_csv.py` can infer a *candidate* schema from a CSV. **This is not DP-safe**
if the CSV is private. The intended workflow is:

1. Generate candidate once (pre-experiment)
2. Review/edit/freeze as a public side-info artifact
3. Use it in experiments via `--schema`

Example:

```bash
python scripts/prepare_schema_from_csv.py --data data/breast_cancer.csv --out schemas/breast_cancer_public_schema.json --target-col target
```

