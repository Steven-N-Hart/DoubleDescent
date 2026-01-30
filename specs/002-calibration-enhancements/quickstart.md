# Quickstart: Calibration Decomposition and Experimental Enhancements

**Feature**: 002-calibration-enhancements
**Date**: 2026-01-19

## Overview

This feature adds calibration decomposition metrics (calibration-in-the-large, calibration slope, ICI) to the DoubleDescent framework, increases experimental seeds from 5 to 20, and reduces categorical cardinality from 100 to 10 levels.

## Prerequisites

- Python 3.8+
- Existing DoubleDescent installation
- lifelines package (for Cox regression in calibration slope)
- statsmodels package (for LOESS smoothing in ICI)

## Quick Verification

After implementation, verify the feature works:

```bash
# 1. Run a single experiment and check for calibration metrics
python -m src.cli.run_experiment --config configs/experiments/baseline_sweep.json

# Check output contains calibration columns
head -1 outputs/experiments/*/results/metrics.csv
# Should show: ...,cal_large,cal_slope,ici,...

# 2. Verify 20 seeds are used by default
python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --dry-run
# Should list 20 seed values

# 3. Verify categorical cardinality
cat configs/scenarios/high_cardinality.json | grep cardinality
# Should show: "cardinality": 10

# 4. Run unit tests for calibration
pytest tests/unit/test_calibration.py -v
```

## Using the New Calibration Metrics

### In Python code

```python
from src.metrics.calibration import (
    calibration_in_the_large,
    calibration_slope,
    integrated_calibration_index,
)

# After getting predictions from a model
cal_large = calibration_in_the_large(survival_probs, event_indicators)
cal_slope = calibration_slope(risk_scores, event_times, event_indicators)
ici = integrated_calibration_index(survival_probs_at_t, event_times, event_indicators, time_horizon)
```

### In experiment output

Calibration metrics appear automatically in:
- `outputs/experiments/*/results/metrics.csv` (epoch-level)
- `outputs/experiments/*/results/summary.csv` (final)
- `outputs/baselines/*.csv` (baseline comparisons)

## Interpreting Calibration Metrics

| Metric | Perfect Value | Interpretation |
|--------|---------------|----------------|
| Calibration-in-the-large | 1.0 | O/E < 1 = over-predicting risk; O/E > 1 = under-predicting |
| Calibration slope | 1.0 | < 1 = overfitting (too extreme); > 1 = underfitting (too conservative) |
| ICI | 0.0 | Higher values indicate worse calibration |

## Common Issues

### Missing calibration metrics (NaN values)

- **Cause**: Insufficient events or degenerate predictions
- **Solution**: Check data has adequate events; verify model is training properly

### Calibration slope computation fails

- **Cause**: All events at same time, or convergence failure
- **Solution**: System returns NaN and logs warning; check data distribution

### LOESS smoothing warning

- **Cause**: Insufficient unique predicted probabilities
- **Solution**: Increase sample size or check for constant predictions

## File Changes Summary

| File | Change Type |
|------|-------------|
| `src/metrics/calibration.py` | NEW |
| `src/metrics/__init__.py` | UPDATE |
| `src/metrics/evaluator.py` | UPDATE |
| `src/metrics/results.py` | UPDATE |
| `src/models/baselines.py` | UPDATE |
| `scripts/run_multi_seed.py` | UPDATE |
| `configs/scenarios/high_cardinality.json` | UPDATE |
| `manuscript/main.tex` | UPDATE |
| `manuscript/references.bib` | UPDATE |
| `tests/unit/test_calibration.py` | NEW |
