# Data Model: Calibration Decomposition and Experimental Enhancements

**Feature**: 002-calibration-enhancements
**Date**: 2026-01-19

## Entity Changes

### 1. CalibrationResult (NEW)

Container for calibration decomposition metrics.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| calibration_in_the_large | float | O/E ratio | > 0, NaN if undefined |
| calibration_slope | float | Cox regression coefficient | Any real, NaN if undefined |
| ici | float | Integrated Calibration Index | >= 0, NaN if undefined |
| time_horizon | float | Time point for ICI evaluation | > 0 |

**Validation rules**:
- O/E ratio should be positive (0 observed events → NaN)
- ICI is always non-negative
- time_horizon must be within observed time range

### 2. MetricResult (EXTEND)

Add calibration fields to existing dataclass.

| New Field | Type | Description | Default |
|-----------|------|-------------|---------|
| calibration_in_the_large | Optional[float] | O/E ratio | None |
| calibration_slope | Optional[float] | Calibration slope | None |
| ici | Optional[float] | ICI value | None |

**Serialization updates**:
- `to_dict()`: Include new fields
- `to_csv_row()`: Include new columns
- `from_dict()`: Parse new fields with backward compatibility

### 3. RunSummary (EXTEND)

Add final calibration metrics for summary tables.

| New Field | Type | Description |
|-----------|------|-------------|
| final_test_cal_large | float | Final test O/E ratio |
| final_test_cal_slope | float | Final test calibration slope |
| final_test_ici | float | Final test ICI |

### 4. BaselineResults (EXTEND in baselines.py)

Add calibration metrics to baseline comparison output.

| New Field | Type | Description |
|-----------|------|-------------|
| calibration_in_the_large | Optional[float] | O/E ratio |
| calibration_slope | Optional[float] | Calibration slope |
| ici | Optional[float] | ICI value |

### 5. Scenario Configuration (UPDATE)

high_cardinality.json schema change:

| Field | Old Value | New Value |
|-------|-----------|-----------|
| cardinality | 100 | 10 |
| description | "...100 levels each..." | "...10 levels each..." |

## State Transitions

### Metric Computation Flow

```
Training Complete
    │
    ▼
Risk Scores Generated
    │
    ├──► C-index Computation (unchanged)
    │
    ├──► IBS Computation (unchanged)
    │       │
    │       ▼
    │    Survival Functions Available
    │       │
    │       ├──► Calibration-in-the-large
    │       │    (uses S(T_i|X_i) at observed times)
    │       │
    │       └──► ICI Computation
    │            (uses S(t0|X_i) at median time)
    │
    └──► Calibration Slope
         (uses risk scores directly in Cox regression)
```

## Configuration Schema Changes

### DEFAULT_SEEDS Expansion

```python
# Old (5 seeds)
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]

# New (20 seeds)
DEFAULT_SEEDS = [
    42, 123, 456, 789, 1011,
    2024, 2025, 2026, 3141, 2718,
    1618, 1414, 1732, 9999, 8888,
    7777, 6666, 5555, 4444, 3333,
]
```

## CSV Output Schema Changes

### Epoch-level metrics CSV

Add columns:
- `cal_large` (float or empty)
- `cal_slope` (float or empty)
- `ici` (float or empty)

### Summary CSV

Add columns:
- `final_test_cal_large`
- `final_test_cal_slope`
- `final_test_ici`

### Baseline comparison CSV

Add columns:
- `cal_large`
- `cal_slope`
- `ici`

## Backward Compatibility

### Reading old results

- New calibration fields default to None/empty
- `from_dict()` handles missing fields gracefully
- Aggregation skips missing calibration values

### Writing new results

- All new experiments include calibration metrics
- CSV headers include new columns
- JSON output includes new fields
