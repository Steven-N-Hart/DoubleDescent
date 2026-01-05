# Research: Experiment Framework Design

**Branch**: `001-experiment-framework-design` | **Date**: 2026-01-04
**Purpose**: Resolve technical decisions and document best practices for implementation

## 1. Deep Learning Framework

### Decision: PyTorch

**Rationale**:
- Already specified in requirements.txt (torch >= 1.10.0)
- Native support for dynamic computation graphs (easier debugging)
- TensorBoard integration via torch.utils.tensorboard
- Strong ecosystem for survival analysis (pycox, torchtuples)
- GPU/CPU seamless switching with `.to(device)` pattern

**Alternatives Considered**:
- TensorFlow/Keras: More verbose, less Pythonic, would require rewriting pycox patterns
- JAX: Better for research but steeper learning curve, less survival analysis tooling

## 2. DeepSurv Implementation

### Decision: Custom implementation based on pycox patterns

**Rationale**:
- pycox provides reference implementations but we need full control over:
  - Architecture (configurable width/depth)
  - Training loop (epoch-level metric logging, gradient norms)
  - Checkpoint format (must include optimizer state for resumption)
- Custom implementation allows direct Cox partial likelihood optimization
- Reuse pycox utilities for data preprocessing and baseline hazard estimation

**Implementation Pattern**:
```python
class DeepSurv(nn.Module):
    def __init__(self, in_features: int, hidden_widths: List[int], dropout: float = 0.0):
        # MLP with configurable layers
        # Output: single log-risk score

    def forward(self, x: Tensor) -> Tensor:
        # Return log-hazard ratio
```

**Alternatives Considered**:
- Direct pycox.models.CoxPH: Insufficient control over training diagnostics
- scikit-survival GradientBoostingSurvivalAnalysis: Not a neural network

## 3. Cox Partial Likelihood Loss

### Decision: Custom differentiable implementation

**Rationale**:
- Must be differentiable for PyTorch autograd
- Need to handle tied event times (Breslow approximation)
- pycox provides reference but we need gradient norm access

**Implementation Pattern**:
```python
def cox_partial_likelihood_loss(log_hazard: Tensor, durations: Tensor, events: Tensor) -> Tensor:
    """
    Negative log partial likelihood with Breslow tie handling.

    Args:
        log_hazard: (N,) predicted log-hazard ratios
        durations: (N,) observed times
        events: (N,) event indicators (1=event, 0=censored)

    Returns:
        Scalar loss value
    """
    # Sort by duration descending
    # Compute log-sum-exp for risk sets
    # Return negative mean log partial likelihood
```

**Alternatives Considered**:
- pycox.models.loss.CoxPHLoss: Would work but adds dependency coupling
- lifelines partial likelihood: Not GPU-accelerated

## 4. Survival Metrics Implementation

### Decision: scikit-survival for C-index, custom for IBS

**Rationale**:
- scikit-survival's concordance_index_censored is battle-tested
- Integrated Brier Score requires survival function estimates
  - Use Kaplan-Meier for baseline + model predictions
  - sksurv.metrics.integrated_brier_score handles censoring correctly

**Implementation Pattern**:
```python
from sksurv.metrics import concordance_index_censored, integrated_brier_score

def evaluate_model(model, X, durations, events, times):
    risk_scores = model(X).numpy()

    # C-index
    c_index = concordance_index_censored(events, durations, risk_scores)[0]

    # IBS requires survival function estimates
    surv_funcs = estimate_survival_functions(model, X, times)
    ibs = integrated_brier_score(structured_y, structured_y, surv_funcs, times)

    return {"c_index": c_index, "ibs": ibs}
```

**Alternatives Considered**:
- lifelines.utils.concordance_index: Doesn't handle ties as robustly
- Custom IBS: Error-prone with censoring edge cases

## 5. Synthetic Data Generation

### Decision: Inverse transform sampling with Weibull baseline

**Rationale**:
- Research proposal specifies this approach
- Weibull provides closed-form inverse for efficient sampling
- Allows precise control over hazard shape (increasing/decreasing/constant)

**Implementation Pattern**:
```python
def generate_survival_times(X: ndarray, beta: ndarray,
                           lambda_scale: float = 0.5,
                           nu_shape: float = 2.0) -> ndarray:
    """
    Generate survival times via inverse transform sampling.

    T = (-log(U) / (lambda * exp(X @ beta)))^(1/nu)

    where U ~ Uniform(0, 1)
    """
    linear_pred = X @ beta
    u = np.random.uniform(0, 1, len(X))
    T = (-np.log(u) / (lambda_scale * np.exp(linear_pred))) ** (1 / nu_shape)
    return T
```

**Alternatives Considered**:
- Exponential baseline: Too simple, no shape parameter
- Cox-Gompertz: More complex, no closed-form inverse

## 6. Gaussian Copula for Correlated Covariates

### Decision: scipy.stats multivariate normal + marginal transforms

**Rationale**:
- scipy.stats.multivariate_normal for correlated latent variables
- Marginal transforms via CDF inversion
- Simple, well-understood approach

**Implementation Pattern**:
```python
from scipy.stats import multivariate_normal, norm, lognorm

def generate_correlated_covariates(n: int, correlation_matrix: ndarray,
                                   marginal_types: List[str]) -> ndarray:
    """
    Generate covariates with specified marginals and correlation structure.

    1. Sample from multivariate normal with given correlation
    2. Transform each margin to uniform via normal CDF
    3. Transform to target marginal via inverse CDF
    """
    # Latent Gaussian
    Z = multivariate_normal.rvs(cov=correlation_matrix, size=n)

    # Transform to uniforms
    U = norm.cdf(Z)

    # Transform to target marginals
    X = np.column_stack([
        transform_to_marginal(U[:, j], marginal_types[j])
        for j in range(len(marginal_types))
    ])
    return X
```

**Alternatives Considered**:
- Vine copulas: More flexible but complex, overkill for this use case
- Direct simulation: Can't control correlation structure

## 7. Censoring Rate Calibration

### Decision: Binary search on exponential rate parameter

**Rationale**:
- Exponential censoring is standard (non-informative)
- Binary search converges quickly to target rate
- Tolerance of ±2% is achievable in ~10 iterations

**Implementation Pattern**:
```python
def calibrate_censoring_rate(T: ndarray, target_rate: float,
                             tol: float = 0.02, max_iter: int = 20) -> float:
    """
    Find lambda_c such that P(T > C) ≈ target_rate where C ~ Exp(lambda_c).
    """
    low, high = 0.001, 10.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        C = np.random.exponential(1/mid, len(T))
        actual_rate = np.mean(T > C)

        if abs(actual_rate - target_rate) < tol:
            return mid
        elif actual_rate > target_rate:
            low = mid
        else:
            high = mid

    return mid  # Best effort
```

**Alternatives Considered**:
- Weibull censoring: Adds complexity without benefit
- Administrative censoring (fixed time): Less realistic

## 8. Experiment Configuration Schema

### Decision: JSON with jsonschema validation

**Rationale**:
- Human-readable and editable
- Git-friendly (meaningful diffs)
- jsonschema provides validation with clear error messages
- Already in requirements.txt

**Schema Structure**:
```json
{
  "experiment_id": "baseline_001",
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
  }
}
```

**Alternatives Considered**:
- YAML: Slightly more readable but less strict validation
- TOML: Less familiar to research community
- Python dataclasses: Harder to version control

## 9. Checkpoint Format

### Decision: PyTorch native + JSON metadata

**Rationale**:
- torch.save handles model + optimizer state
- Separate JSON for experiment progress (which widths completed)
- Enables atomic checkpoint updates

**File Structure**:
```
checkpoints/
├── progress.json           # {"completed_widths": [2, 4, 8], "current_width": 16}
├── width_002/
│   ├── model.pt            # model.state_dict()
│   ├── optimizer.pt        # optimizer.state_dict()
│   └── metrics.json        # final metrics for this width
├── width_004/
│   └── ...
```

**Alternatives Considered**:
- Single checkpoint file: Harder to resume mid-experiment
- HDF5: Overkill for this use case

## 10. Logging Strategy

### Decision: TensorBoard + CSV dual logging

**Rationale**:
- TensorBoard: Interactive exploration, training curves, histograms
- CSV: Easy pandas loading for publication plots
- Both written per-epoch for full flexibility

**TensorBoard Tags**:
```
train/loss
train/c_index
train/ibs
train/gradient_norm
train/weight_norm
val/c_index
val/ibs
diagnostics/learning_rate
diagnostics/batch_loss_variance
histograms/weights_layer_{i}
```

**CSV Columns**:
```
epoch,width,depth,train_loss,train_c_index,train_ibs,val_c_index,val_ibs,test_c_index,test_ibs,gradient_norm,weight_norm
```

**Alternatives Considered**:
- MLflow: More infrastructure overhead
- Weights & Biases: Requires account, external dependency
- Just TensorBoard: Harder to load into pandas for final plots

## 11. Visualization Libraries

### Decision: Matplotlib for static, TensorBoard for interactive

**Rationale**:
- Matplotlib: Publication-quality, fine-grained control, PDF export
- Seaborn: Nice defaults for statistical plots
- TensorBoard: Already logging there, interactive exploration

**Plot Types**:
1. Double descent curves (test error vs. parameter count)
2. Metric divergence (C-index vs IBS dual-axis)
3. Training dynamics (loss over epochs per width)
4. Parameter histograms (weight distribution evolution)

**Alternatives Considered**:
- Plotly: Good for interactive but HTML output less useful for papers
- Bokeh: Similar to Plotly, more complex

## 12. CLI Framework

### Decision: argparse with subcommands

**Rationale**:
- Standard library, no extra dependencies
- Familiar to research community
- Easy to document with --help

**Command Structure**:
```bash
python -m src.cli.run_experiment --config configs/experiments/baseline.json
python -m src.cli.generate_data --scenario baseline --n-samples 1000 --output data/
python -m src.cli.visualize --experiment outputs/experiments/baseline_001/ --output figures/
python -m src.cli.compare --experiments exp1 exp2 exp3 --output figures/comparison/
```

**Alternatives Considered**:
- Click: More features but extra dependency
- Typer: Nice but adds complexity
- Fire: Too magical, hard to document

## Summary of Key Decisions

| Component | Decision | Key Rationale |
|-----------|----------|---------------|
| DL Framework | PyTorch | Already in requirements, pycox compatibility |
| DeepSurv | Custom implementation | Full control over training loop |
| Loss Function | Custom Cox PL | Need gradient access |
| C-index | scikit-survival | Battle-tested, handles ties |
| IBS | scikit-survival | Proper censoring handling |
| Data Generation | Inverse transform + Weibull | Closed-form, per research proposal |
| Copula | scipy multivariate_normal | Simple, effective |
| Censoring | Binary search calibration | Precise control |
| Config | JSON + jsonschema | Human-readable, validated |
| Checkpoints | PyTorch .pt + JSON | Standard, resumable |
| Logging | TensorBoard + CSV | Interactive + pandas-friendly |
| Visualization | Matplotlib | Publication quality |
| CLI | argparse | Standard library |
