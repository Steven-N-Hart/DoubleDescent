# CLI Interface Contract: Experiment Framework

**Branch**: `001-experiment-framework-design` | **Date**: 2026-01-04
**Purpose**: Define the command-line interface for all experiment operations

## Overview

All CLI commands are invoked via Python module execution:

```bash
python -m src.cli.<command> [OPTIONS]
```

## Commands

### 1. run_experiment

**Purpose**: Execute a complete experiment with capacity sweep.

```bash
python -m src.cli.run_experiment --config <path> [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--config` | PATH | Yes | - | Path to experiment JSON config |
| `--output-dir` | PATH | No | `outputs/experiments/` | Output directory |
| `--resume` | FLAG | No | False | Resume from last checkpoint |
| `--dry-run` | FLAG | No | False | Validate config without running |
| `--device` | STR | No | `cuda` | Device: `cuda`, `cpu`, `auto` |
| `--verbose` | FLAG | No | False | Enable verbose logging |

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (all runs completed or resumed) |
| 1 | Configuration error |
| 2 | Runtime error (unrecoverable) |
| 3 | Partial completion (some runs failed/skipped) |

#### Output

```
stdout: Progress updates, completion summary
stderr: Warnings, errors

Files created:
  {output_dir}/{experiment_id}/
    ├── config.json
    ├── data/
    ├── runs/
    ├── results/
    ├── tensorboard/
    └── progress.json
```

#### Examples

```bash
# Run baseline experiment
python -m src.cli.run_experiment --config configs/experiments/baseline.json

# Resume interrupted experiment
python -m src.cli.run_experiment --config outputs/experiments/exp_001/config.json --resume

# Dry run to validate config
python -m src.cli.run_experiment --config my_config.json --dry-run

# Force CPU execution
python -m src.cli.run_experiment --config baseline.json --device cpu
```

---

### 2. generate_data

**Purpose**: Generate synthetic survival data without training.

```bash
python -m src.cli.generate_data --scenario <name> [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--scenario` | STR | Yes* | - | Predefined scenario name |
| `--config` | PATH | Yes* | - | Custom scenario config file |
| `--output` | PATH | Yes | - | Output directory or file |
| `--n-samples` | INT | No | 1000 | Override sample count |
| `--seed` | INT | No | Random | Random seed |
| `--format` | STR | No | `npz` | Output format: `npz`, `csv`, `parquet` |

*Either `--scenario` or `--config` required.

#### Predefined Scenarios

| Name | Description |
|------|-------------|
| `baseline` | Gaussian covariates, 30% censoring |
| `skewed` | Log-normal covariates, 30% censoring |
| `high_cardinality` | 5 categorical (100 levels), 30% censoring |
| `imbalanced` | Gaussian covariates, 90% censoring |

#### Output Format (npz)

```python
# data.npz contains:
{
    'X': ndarray,           # (n_samples, n_features) covariates
    'T': ndarray,           # (n_samples,) observed times
    'E': ndarray,           # (n_samples,) event indicators
    'T_true': ndarray,      # (n_samples,) true event times (before censoring)
    'beta': ndarray,        # (n_features,) ground truth coefficients
}
```

#### Examples

```bash
# Generate baseline data
python -m src.cli.generate_data --scenario baseline --output data/baseline.npz --seed 42

# Generate custom scenario
python -m src.cli.generate_data --config my_scenario.json --output data/custom/

# Generate CSV for external analysis
python -m src.cli.generate_data --scenario skewed --output data/skewed.csv --format csv
```

---

### 3. visualize

**Purpose**: Generate visualizations from experiment results.

```bash
python -m src.cli.visualize --experiment <path> [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--experiment` | PATH | Yes | - | Path to experiment output directory |
| `--output` | PATH | No | `{experiment}/figures/` | Output directory for figures |
| `--format` | STR | No | `png,pdf` | Output formats (comma-separated) |
| `--plots` | STR | No | `all` | Plot types to generate |
| `--dpi` | INT | No | 300 | Resolution for raster formats |

#### Plot Types

| Type | Description |
|------|-------------|
| `double_descent` | Test error vs capacity (main result) |
| `metric_divergence` | C-index vs IBS dual-axis |
| `training_curves` | Loss over epochs per width |
| `gradient_norms` | Gradient norm evolution |
| `weight_norms` | Weight norm evolution |
| `all` | Generate all plot types |

#### Examples

```bash
# Generate all plots
python -m src.cli.visualize --experiment outputs/experiments/exp_001/

# Generate only double descent curve in PDF
python -m src.cli.visualize --experiment exp_001/ --plots double_descent --format pdf

# High-resolution for publication
python -m src.cli.visualize --experiment exp_001/ --dpi 600 --format pdf
```

---

### 4. compare

**Purpose**: Compare results across multiple experiments.

```bash
python -m src.cli.compare --experiments <path1> <path2> ... [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--experiments` | PATH... | Yes | - | Paths to experiment directories |
| `--output` | PATH | Yes | - | Output directory for comparison |
| `--metric` | STR | No | `ibs` | Primary metric for comparison |
| `--labels` | STR... | No | Auto | Custom labels for legend |
| `--format` | STR | No | `png,pdf` | Output formats |

#### Output

```
{output}/
├── comparison_curves.{png,pdf}      # Overlaid double descent curves
├── peak_analysis.{png,pdf}          # Peak location/height comparison
├── summary_table.csv                # Tabular comparison metrics
└── comparison_report.md             # Markdown summary
```

#### Examples

```bash
# Compare baseline vs skewed
python -m src.cli.compare \
    --experiments outputs/experiments/baseline_001/ outputs/experiments/skewed_001/ \
    --output figures/comparison/ \
    --labels "Gaussian" "Log-Normal"

# Compare different censoring rates
python -m src.cli.compare \
    --experiments exp_censor_20/ exp_censor_50/ exp_censor_90/ \
    --metric ibs \
    --labels "20%" "50%" "90%"
```

---

### 5. status

**Purpose**: Check status of running or completed experiments.

```bash
python -m src.cli.status [--experiment <path>] [OPTIONS]
```

#### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--experiment` | PATH | No | - | Specific experiment to check |
| `--all` | FLAG | No | False | List all experiments in outputs/ |
| `--json` | FLAG | No | False | Output as JSON |

#### Output (default)

```
Experiment: baseline_001
Status: RUNNING
Progress: 12/20 widths completed (60%)
Current: width=256, epoch=15234/50000
Elapsed: 2h 34m
Estimated remaining: 1h 45m

Completed widths: 2, 4, 8, 16, 32, 64, 128
Failed widths: none
Pending widths: 256 (running), 512, 1024, 2048
```

#### Output (--json)

```json
{
  "experiment_id": "baseline_001",
  "status": "RUNNING",
  "progress": {
    "completed": 12,
    "total": 20,
    "percentage": 60
  },
  "current_run": {
    "width": 256,
    "epoch": 15234,
    "total_epochs": 50000
  },
  "timing": {
    "elapsed_seconds": 9240,
    "estimated_remaining_seconds": 6300
  }
}
```

---

## Common Options

All commands support these options:

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message |
| `--version` | Show version |
| `--quiet`, `-q` | Suppress non-error output |
| `--log-level` | Set logging level: DEBUG, INFO, WARNING, ERROR |
| `--log-file` | Write logs to file |

---

## Configuration File Schema

### Experiment Config (complete)

```json
{
  "$schema": "./schemas/experiment.json",
  "experiment_id": "baseline_001",
  "name": "Baseline Double Descent",
  "description": "Standard Gaussian covariates, 30% censoring",
  "seed": 42,

  "data": {
    "scenario": "baseline",
    "n_samples": 1000,
    "n_features": 20,
    "n_predictive": 10,
    "censoring_rate": 0.3
  },

  "model": {
    "widths": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "depths": [2],
    "activation": "relu",
    "dropout": 0.0,
    "weight_decay": 0.0
  },

  "training": {
    "epochs": 50000,
    "batch_size": 256,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },

  "splits": {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
  }
}
```

### Scenario Config (for generate_data)

```json
{
  "$schema": "./schemas/scenario.json",
  "name": "custom_scenario",
  "description": "Custom data generation scenario",
  "n_samples": 2000,
  "n_features": 50,
  "n_predictive": 25,
  "covariate_type": "mixed",
  "n_categorical_features": 5,
  "cardinality": 50,
  "censoring_rate": 0.5,
  "weibull_scale": 0.3,
  "weibull_shape": 1.5,
  "correlation_matrix": null
}
```

---

## Error Messages

### Configuration Errors (exit code 1)

```
ERROR: Config file not found: path/to/config.json
ERROR: Invalid config: 'n_predictive' (25) exceeds 'n_features' (20)
ERROR: Unknown scenario: 'invalid_scenario'
ERROR: Schema validation failed: 'width' must be positive integer
```

### Runtime Errors (exit code 2)

```
ERROR: CUDA out of memory. Consider reducing batch_size or max width.
ERROR: Cannot resume: no checkpoint found at path/to/experiment/
ERROR: Data generation failed: censoring rate unreachable (actual: 0.98, target: 0.50)
```

### Partial Completion (exit code 3)

```
WARNING: Run width=2048 failed (OOM), skipping
WARNING: Run width=4096 failed after retry, marking as FAILED
SUMMARY: 18/20 runs completed, 2 skipped/failed
```

---

## Programmatic Interface

All CLI commands are also importable for use in scripts:

```python
from src.cli.run_experiment import run_experiment
from src.cli.generate_data import generate_data
from src.cli.visualize import generate_visualizations
from src.cli.compare import compare_experiments

# Run experiment programmatically
result = run_experiment(
    config_path="configs/experiments/baseline.json",
    output_dir="outputs/experiments/",
    device="cuda",
    resume=False
)

# Generate data
data = generate_data(
    scenario="baseline",
    n_samples=1000,
    seed=42
)
```
