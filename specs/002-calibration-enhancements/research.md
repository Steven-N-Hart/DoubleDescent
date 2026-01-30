# Research: Calibration Decomposition and Experimental Enhancements

**Feature**: 002-calibration-enhancements
**Date**: 2026-01-19

## 1. Calibration Metrics Implementation

### 1.1 Calibration-in-the-Large (O/E Ratio)

**Decision**: Implement as ratio of observed to expected events using Poisson regression framework

**Rationale**:
- Per Crowson et al. (2016), calibration-in-the-large measures systematic over/under-prediction
- Formula: `O/E = sum(observed_events) / sum(expected_events)`
- Expected events derived from predicted survival probabilities: `E_i = 1 - S_hat(T_i | X_i)`
- Value of 1.0 indicates perfect calibration; <1 means over-prediction, >1 means under-prediction

**Implementation**:
```python
def calibration_in_the_large(
    survival_probs: np.ndarray,  # S(t|x) at observed times
    event_indicators: np.ndarray
) -> float:
    observed = event_indicators.sum()
    expected = (1 - survival_probs).sum()  # Expected events = 1 - S(T_i|X_i)
    return observed / expected if expected > 0 else np.nan
```

**Alternatives considered**:
- Poisson GLM intercept (more complex, same result for simple O/E)
- Hosmer-Lemeshow test (requires binning, less interpretable)

### 1.2 Calibration Slope

**Decision**: Implement using Cox regression with linear predictor as sole covariate

**Rationale**:
- Per Royston & Altman (2013), calibration slope is the coefficient when regressing observed outcomes on predicted risk
- Slope of 1.0 indicates perfect spread of predictions
- Slope < 1 indicates overfitting (predictions too extreme)
- Slope > 1 indicates underfitting (predictions too conservative)

**Implementation**:
```python
def calibration_slope(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    # Use lifelines CoxPHFitter with risk_scores as sole covariate
    from lifelines import CoxPHFitter
    df = pd.DataFrame({
        'T': event_times,
        'E': event_indicators,
        'risk': risk_scores
    })
    cph = CoxPHFitter()
    cph.fit(df, duration_col='T', event_col='E', formula='risk')
    return cph.params_['risk']
```

**Alternatives considered**:
- Logistic regression at fixed time point (loses time-to-event information)
- Manual optimization (unnecessary complexity)

### 1.3 Integrated Calibration Index (ICI)

**Decision**: Implement using LOESS smoothing at median event time

**Rationale**:
- Per Austin et al. (2020), ICI is mean absolute difference between predicted and smoothed observed probabilities
- LOESS provides flexible non-parametric smoothing
- Single time horizon (median event time) balances informativeness with simplicity
- Clarified in spec: use LOESS, not isotonic regression

**Implementation**:
```python
def integrated_calibration_index(
    survival_probs_at_t: np.ndarray,  # S(t0|x) at fixed time t0
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    time_horizon: float
) -> float:
    # Binary outcome at time_horizon
    observed = ((event_times <= time_horizon) & (event_indicators == 1)).astype(float)

    # LOESS smoothing
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(observed, 1 - survival_probs_at_t, frac=0.3, return_sorted=False)

    # ICI = mean absolute difference
    return np.abs((1 - survival_probs_at_t) - smoothed).mean()
```

**Alternatives considered**:
- Isotonic regression (guarantees monotonicity but less flexible)
- Multiple time horizons (adds complexity without clear benefit for this study)

## 2. Seed Configuration

### 2.1 Expanding from 5 to 20 Seeds

**Decision**: Use 20 deterministic seed values for reproducibility

**Rationale**:
- 4x increase in seeds reduces standard error by ~50% (√4 = 2)
- Enables proper statistical testing of effects (e.g., regularization benefit)
- Maintains original 5 seeds for backwards compatibility

**Selected seeds**:
```python
DEFAULT_SEEDS = [
    42, 123, 456, 789, 1011,           # Original 5
    2024, 2025, 2026, 3141, 2718,      # Set 2 (years + mathematical constants)
    1618, 1414, 1732, 9999, 8888,      # Set 3
    7777, 6666, 5555, 4444, 3333,      # Set 4
]
```

**Alternatives considered**:
- Random seed generation (breaks reproducibility)
- 10 seeds (insufficient for statistical power)
- 50 seeds (excessive computation time)

## 3. Categorical Cardinality

### 3.1 Reducing from 100 to 10 Levels

**Decision**: Change cardinality to 10 levels per categorical feature

**Rationale**:
- With 1000 samples, 5 features × 100 levels = ~2 observations per category (insufficient)
- With 10 levels: 1000 / (5 × 10) = ~20 observations per category-feature combination
- More clinically realistic (e.g., disease stage, treatment type)
- One-hot encoding produces 50 features (manageable) vs 500 (problematic)

**Alternatives considered**:
- 5 levels (too simple, doesn't test high-cardinality challenges)
- 20 levels (borderline sample size per category)
- Embedding approach (already tested and failed in current experiments)

## 4. Manuscript Updates

### 4.1 Required Citations

**Decision**: Add three key calibration methodology papers

**BibTeX entries**:
```bibtex
@article{austin2020graphical,
  title={Graphical calibration curves and the integrated calibration index ({ICI}) for survival models},
  author={Austin, Peter C and Harrell Jr, Frank E and van Klaveren, David},
  journal={Statistics in Medicine},
  volume={39},
  number={21},
  pages={2714--2742},
  year={2020},
  doi={10.1002/sim.8570}
}

@article{crowson2016assessing,
  title={Assessing calibration of prognostic risk scores},
  author={Crowson, Cynthia S and Atkinson, Elizabeth J and Therneau, Terry M},
  journal={Statistical Methods in Medical Research},
  volume={25},
  number={4},
  pages={1692--1706},
  year={2016},
  doi={10.1177/0962280213497434}
}

@article{royston2013external,
  title={External validation of a {Cox} prognostic model: principles and methods},
  author={Royston, Patrick and Altman, Douglas G},
  journal={BMC Medical Research Methodology},
  volume={13},
  number={1},
  pages={33},
  year={2013},
  doi={10.1186/1471-2288-13-33}
}
```

### 4.2 Introduction Text Addition

**Decision**: Add calibration decomposition context to Introduction

**Location**: After paragraph discussing concordance vs IBS divergence (around line 93-94)

**Proposed text**:
> Calibration assessment in survival models can be decomposed into distinct components \citep{crowson2016assessing}. Calibration-in-the-large captures systematic over- or under-prediction of event rates. The calibration slope measures whether risk stratification is appropriately spread across the prediction range. The integrated calibration index (ICI) provides a summary measure of calibration error across the risk distribution \citep{austin2020graphical}. This decomposition enables precise characterization of how and where prognostic models fail.

### 4.3 Terminology Standardization

**Decision**: Use "threshold region" consistently instead of "critical region"

**Rationale**: "Critical region" has statistical testing connotations; "threshold region" is more descriptive of the interpolation threshold phenomenon.

### 4.4 Covariate Structure Description

**Decision**: Add explicit statement in Methods Section 3.1

**Location**: After Gaussian copula description (line 155-156)

**Proposed text**:
> The coefficient vector $\bbeta$ contains 10 predictive features with coefficients drawn uniformly from $[-1, 1]$, and 10 noise features with coefficients fixed at zero.

## 5. Integration Points

### 5.1 MetricResult Extension

**Fields to add**:
```python
# Calibration Decomposition
calibration_in_the_large: Optional[float] = None
calibration_slope: Optional[float] = None
ici: Optional[float] = None
```

### 5.2 MetricEvaluator Extension

**New method**: `_compute_calibration()` following pattern of `_compute_ibs()`

**Integration point**: Call after `_compute_ibs()` in `evaluate()` method

### 5.3 Baseline Extension

**Update both**: `CoxPHBaseline.evaluate()` and `RandomSurvivalForestBaseline.evaluate()`

**Output format**: Include calibration metrics in CSV output

## 6. Testing Strategy

### 6.1 Unit Tests for Calibration Metrics

**Test cases**:
1. Perfect calibration (synthetic data where predictions match outcomes)
2. Over-prediction (O/E < 1, slope < 1)
3. Under-prediction (O/E > 1, slope > 1)
4. Edge cases (all events, no events, tied times)
5. NaN handling for degenerate cases

### 6.2 Integration Tests

**Verify**:
1. Calibration metrics appear in experiment output CSV
2. Aggregation includes calibration metrics
3. Baseline comparison includes calibration
