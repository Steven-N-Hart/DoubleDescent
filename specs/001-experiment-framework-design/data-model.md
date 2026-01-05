# Data Model: Experiment Framework Design

**Branch**: `001-experiment-framework-design` | **Date**: 2026-01-04
**Purpose**: Define entity schemas, relationships, and validation rules

## Entity Relationship Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  DataScenario   │────▶│    Experiment    │────▶│  ExperimentRun  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                        │
                               │                        │
                               ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ ModelConfiguration│    │  MetricResult   │
                        └──────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                 ┌─────────────────┐
                                                 │DoubleDescentCurve│
                                                 └─────────────────┘
```

## 1. DataScenario

Defines a synthetic data generation configuration.

### Schema

```python
@dataclass
class DataScenario:
    """Configuration for synthetic survival data generation."""

    # Identity
    name: str                           # Unique identifier (e.g., "baseline", "skewed")
    description: str                    # Human-readable description

    # Sample Configuration
    n_samples: int                      # Total number of samples (default: 1000)
    n_features: int                     # Total number of features (default: 20)
    n_predictive: int                   # Number of non-zero coefficients (default: 10)

    # Covariate Distribution
    covariate_type: CovariateType       # Enum: GAUSSIAN, LOGNORMAL, CATEGORICAL, MIXED
    correlation_matrix: Optional[ndarray]  # Correlation structure (default: identity)

    # For categorical covariates
    n_categorical_features: int = 0     # Number of categorical features
    cardinality: int = 100              # Number of levels per categorical

    # Censoring
    censoring_rate: float               # Target censoring proportion (0.0 to 0.95)
    censoring_distribution: str = "exponential"  # Only exponential supported

    # Weibull Parameters
    weibull_scale: float = 0.5          # lambda parameter
    weibull_shape: float = 2.0          # nu parameter (>1: increasing hazard)

    # Ground Truth
    coefficient_range: Tuple[float, float] = (-1.0, 1.0)  # Range for beta
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["name", "n_samples", "n_features", "covariate_type", "censoring_rate"],
  "properties": {
    "name": {"type": "string", "pattern": "^[a-z_]+$"},
    "description": {"type": "string"},
    "n_samples": {"type": "integer", "minimum": 100, "maximum": 100000},
    "n_features": {"type": "integer", "minimum": 1, "maximum": 1000},
    "n_predictive": {"type": "integer", "minimum": 1},
    "covariate_type": {"enum": ["gaussian", "lognormal", "categorical", "mixed"]},
    "censoring_rate": {"type": "number", "minimum": 0.0, "maximum": 0.95},
    "weibull_scale": {"type": "number", "exclusiveMinimum": 0},
    "weibull_shape": {"type": "number", "exclusiveMinimum": 0}
  }
}
```

### Predefined Scenarios

| Name | Covariate Type | Censoring | Description |
|------|---------------|-----------|-------------|
| `baseline` | Gaussian | 30% | Standard normal covariates |
| `skewed` | LogNormal | 30% | Heavy-tailed biomarker-like |
| `high_cardinality` | Categorical | 30% | 5 categorical, 100 levels each |
| `imbalanced` | Gaussian | 90% | Heavy censoring (rare events) |

---

## 2. ModelConfiguration

Specifies the DeepSurv architecture and training parameters.

### Schema

```python
@dataclass
class ModelConfiguration:
    """Configuration for a single DeepSurv model."""

    # Architecture
    width: int                          # Neurons per hidden layer
    depth: int = 2                      # Number of hidden layers
    activation: str = "relu"            # Activation function
    dropout: float = 0.0                # Dropout rate (0 = disabled)

    # Regularization
    weight_decay: float = 0.0           # L2 regularization (0 = disabled)

    # Training
    epochs: int = 50000                 # Maximum training epochs
    batch_size: int = 256               # Training batch size
    learning_rate: float = 0.001        # Initial learning rate
    optimizer: str = "adam"             # Optimizer type

    # Retry settings
    retry_lr_factor: float = 0.1        # LR reduction on failure retry

    @property
    def n_parameters(self) -> int:
        """Approximate parameter count for P/N ratio calculation."""
        # Input layer: n_features * width
        # Hidden layers: (depth-1) * width * width
        # Output layer: width * 1
        # Biases: depth * width + 1
        # Simplified: O(width^2 * depth)
        pass
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["width"],
  "properties": {
    "width": {"type": "integer", "minimum": 1, "maximum": 4096},
    "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 2},
    "activation": {"enum": ["relu", "tanh", "selu"], "default": "relu"},
    "dropout": {"type": "number", "minimum": 0.0, "maximum": 0.9, "default": 0.0},
    "weight_decay": {"type": "number", "minimum": 0.0, "default": 0.0},
    "epochs": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 50000},
    "batch_size": {"type": "integer", "minimum": 1, "default": 256},
    "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
    "optimizer": {"enum": ["adam", "sgd", "adamw"], "default": "adam"}
  }
}
```

---

## 3. Experiment

Represents a complete experimental configuration.

### Schema

```python
@dataclass
class Experiment:
    """Complete experiment configuration."""

    # Identity
    experiment_id: str                  # Unique identifier (auto-generated if not provided)
    name: str                           # Human-readable name
    description: str = ""               # Optional description

    # Reproducibility
    seed: int                           # Random seed for all operations

    # Data
    data_scenario: DataScenario         # Data generation configuration
    train_ratio: float = 0.6            # Training split ratio
    val_ratio: float = 0.2              # Validation split ratio
    test_ratio: float = 0.2             # Test split ratio

    # Model Sweep
    width_sweep: List[int]              # List of widths to test
    depth_sweep: List[int] = [2]        # List of depths to test
    base_model_config: ModelConfiguration  # Base config (width/depth overridden)

    # Experiment Type
    sweep_type: SweepType = SweepType.WIDTH  # WIDTH or DEPTH

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    status: ExperimentStatus = ExperimentStatus.PENDING
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["seed", "data", "model", "training"],
  "properties": {
    "experiment_id": {"type": "string"},
    "name": {"type": "string"},
    "description": {"type": "string"},
    "seed": {"type": "integer", "minimum": 0},
    "data": {"$ref": "#/definitions/DataScenario"},
    "model": {
      "type": "object",
      "properties": {
        "widths": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "depths": {"type": "array", "items": {"type": "integer"}, "default": [2]},
        "sweep_type": {"enum": ["width", "depth"], "default": "width"}
      }
    },
    "training": {"$ref": "#/definitions/ModelConfiguration"},
    "splits": {
      "type": "object",
      "properties": {
        "train": {"type": "number", "default": 0.6},
        "val": {"type": "number", "default": 0.2},
        "test": {"type": "number", "default": 0.2}
      }
    }
  }
}
```

### State Transitions

```
PENDING ──▶ RUNNING ──▶ COMPLETED
    │          │
    │          ▼
    │      FAILED (can be retried → RUNNING)
    │
    ▼
 CANCELLED
```

---

## 4. ExperimentRun

Represents a single model configuration within an experiment sweep.

### Schema

```python
@dataclass
class ExperimentRun:
    """A single model training run within an experiment."""

    # Identity
    run_id: str                         # Unique within experiment
    experiment_id: str                  # Parent experiment

    # Configuration
    width: int
    depth: int
    model_config: ModelConfiguration    # Full config for this run

    # Status
    status: RunStatus                   # PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
    failure_reason: Optional[str] = None  # If FAILED or SKIPPED

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    epochs_completed: int = 0

    # Checkpoints
    checkpoint_path: Optional[Path] = None
    best_checkpoint_epoch: Optional[int] = None
```

### Run Status

| Status | Description |
|--------|-------------|
| `PENDING` | Not yet started |
| `RUNNING` | Currently training |
| `COMPLETED` | Successfully finished all epochs |
| `FAILED` | Failed after retry (logged as NaN) |
| `SKIPPED` | Skipped due to OOM or other error |

---

## 5. MetricResult

Contains evaluation metrics for a specific model at a specific point.

### Schema

```python
@dataclass
class MetricResult:
    """Metrics captured at a single evaluation point."""

    # Context
    run_id: str
    epoch: int
    split: str                          # "train", "val", or "test"

    # Core Survival Metrics
    c_index: float                      # Concordance index [0, 1]
    integrated_brier_score: float       # IBS (lower is better)
    neg_log_likelihood: float           # Cox partial likelihood loss

    # Training Diagnostics
    gradient_norm: float                # L2 norm of gradients
    weight_norm: float                  # L2 norm of all weights
    learning_rate: float                # Current learning rate
    batch_loss_variance: float          # Variance across batches

    # Timestamp
    timestamp: datetime
```

### CSV Format

```csv
epoch,width,depth,split,c_index,ibs,nll,grad_norm,weight_norm,lr,batch_var,timestamp
0,64,2,train,0.501,0.250,2.345,1.234,0.567,0.001,0.012,2026-01-04T10:00:00
0,64,2,val,0.498,0.255,2.401,,,,,2026-01-04T10:00:01
1,64,2,train,0.523,0.241,2.123,1.198,0.589,0.001,0.011,2026-01-04T10:00:05
...
```

### Validation Rules

- `c_index`: Must be in [0, 1], NaN for failed runs
- `integrated_brier_score`: Must be >= 0, NaN for failed runs
- `gradient_norm`: Must be >= 0, Inf indicates explosion
- `weight_norm`: Must be >= 0

---

## 6. DoubleDescentCurve

Aggregated results across model widths showing the double descent phenomenon.

### Schema

```python
@dataclass
class DoubleDescentCurve:
    """Aggregated metrics across the capacity sweep."""

    # Context
    experiment_id: str
    metric_name: str                    # "c_index", "ibs", "nll"
    split: str                          # "test" (primary), "val", "train"

    # Data Points (one per width/depth)
    capacities: List[int]               # Parameter counts or widths
    values: List[float]                 # Metric values (NaN for failed)
    std_errors: List[float]             # If multiple seeds run

    # Derived
    interpolation_threshold: Optional[int]  # Estimated P ≈ N point
    peak_location: Optional[int]        # Width/capacity at max error
    peak_value: Optional[float]         # Error at peak
    classical_minimum: Optional[float]  # Min error in classical regime
    modern_minimum: Optional[float]     # Min error in over-parameterized regime
```

### Derived Calculations

```python
def find_interpolation_threshold(curve: DoubleDescentCurve,
                                  n_samples: int) -> int:
    """Find capacity closest to P ≈ N."""
    # For survival: consider effective sample size (n_events)
    pass

def find_peak(curve: DoubleDescentCurve) -> Tuple[int, float]:
    """Find the maximum error (interpolation peak)."""
    pass

def classify_regimes(curve: DoubleDescentCurve,
                     threshold: int) -> Dict[str, List[int]]:
    """Classify points into classical/critical/modern regimes."""
    pass
```

---

## 7. Enumerations

```python
from enum import Enum, auto

class CovariateType(Enum):
    GAUSSIAN = auto()
    LOGNORMAL = auto()
    CATEGORICAL = auto()
    MIXED = auto()

class SweepType(Enum):
    WIDTH = auto()      # Vary width, fix depth
    DEPTH = auto()      # Vary depth, fix width

class ExperimentStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class RunStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
```

---

## File Storage Layout

```
outputs/experiments/{experiment_id}/
├── config.json                 # Full experiment configuration
├── data/
│   ├── train.npz              # X, T, E arrays
│   ├── val.npz
│   ├── test.npz
│   └── ground_truth.json      # True beta coefficients
├── runs/
│   ├── width_0002/
│   │   ├── config.json        # Run-specific config
│   │   ├── checkpoints/
│   │   │   ├── epoch_1000.pt
│   │   │   └── best.pt
│   │   ├── metrics.csv        # Per-epoch metrics
│   │   └── status.json        # Run status
│   ├── width_0004/
│   │   └── ...
│   └── ...
├── results/
│   ├── summary.csv            # Final metrics per width
│   └── curves.json            # DoubleDescentCurve data
├── tensorboard/               # TensorBoard logs
└── progress.json              # Experiment progress for resumption
```

---

## Validation Constraints

### Cross-Entity Validation

1. `n_predictive <= n_features` in DataScenario
2. `train_ratio + val_ratio + test_ratio == 1.0` in Experiment
3. All widths in `width_sweep` must be positive integers
4. `seed` must be consistent across all random operations
5. `batch_size <= n_samples * train_ratio`

### Runtime Validation

1. Generated censoring rate within ±2% of target
2. No NaN/Inf in generated covariates
3. All survival times > 0
4. Event indicators are binary (0 or 1)
