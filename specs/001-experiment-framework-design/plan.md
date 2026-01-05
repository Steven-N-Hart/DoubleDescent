# Implementation Plan: Experiment Framework Design

**Branch**: `001-experiment-framework-design` | **Date**: 2026-01-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-experiment-framework-design/spec.md`

## Summary

Design and implement a comprehensive experiment framework for investigating the Double Descent phenomenon in survival analysis. The framework provides synthetic data generation (Weibull hazard with Gaussian copula), DeepSurv model training with capacity sweeps (width/depth), multi-metric evaluation (C-index, IBS, NLL), and visualization with TensorBoard integration. Supports serial execution, checkpoint-based resumption, and full diagnostic logging.

## Technical Context

**Language/Version**: Python 3.8+ (compatible with PyTorch and scikit-survival)
**Primary Dependencies**:
- PyTorch >= 1.10.0 (deep learning)
- scikit-survival >= 0.17.0 (survival metrics)
- pycox >= 0.2.3 (DeepSurv implementation reference)
- lifelines >= 0.27.0 (survival analysis utilities)
- NumPy/SciPy/Pandas (scientific computing)
- Matplotlib/Seaborn (static visualization)
- TensorBoard (interactive logging)

**Storage**: File-based (JSON configs, CSV metrics, PyTorch checkpoints)
**Testing**: pytest with pytest-cov
**Target Platform**: Linux/macOS with CUDA-compatible GPU (CPU fallback)
**Project Type**: Single research project with CLI interface

**Performance Goals**:
- Train models up to 50,000 epochs per configuration
- Support width sweep from 2 to 2048 neurons
- Handle datasets up to N=10,000 samples

**Constraints**:
- Serial execution (one model at a time)
- Must support experiment resumption from checkpoints
- GPU OOM should skip, not crash

**Scale/Scope**:
- 4 predefined data scenarios (Baseline, Skewed, High-Cardinality, Imbalanced)
- ~20 model widths per sweep
- Research-grade reproducibility required

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution template has not been customized for this project. Default research project practices apply:

| Gate | Status | Notes |
|------|--------|-------|
| Library-First | PASS | Framework is a standalone research library |
| CLI Interface | PASS | Experiments run via CLI scripts |
| Test-First | ADVISORY | Tests for core components (data gen, metrics) |
| Observability | PASS | TensorBoard logging, full diagnostics per epoch |
| Simplicity | PASS | Serial execution, file-based storage |

No constitution violations requiring justification.

## Project Structure

### Documentation (this feature)

```text
specs/001-experiment-framework-design/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file
├── research.md          # Phase 0 output - technology decisions
├── data-model.md        # Phase 1 output - entity schemas
├── quickstart.md        # Phase 1 output - getting started guide
├── contracts/           # Phase 1 output - CLI interface specs
│   └── cli-interface.md
└── tasks.md             # Phase 2 output (from /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── __init__.py
├── data/                    # FR-001 to FR-004: Synthetic data generation
│   ├── __init__.py
│   ├── generator.py         # Weibull hazard, inverse transform sampling
│   ├── scenarios.py         # Predefined scenarios (A, B, C, D)
│   ├── copula.py            # Gaussian copula for correlated covariates
│   └── censoring.py         # Censoring rate calibration
│
├── models/                  # FR-005 to FR-007: DeepSurv training
│   ├── __init__.py
│   ├── deepsurv.py          # MLP with Cox partial likelihood
│   ├── trainer.py           # Training loop with diagnostics
│   └── checkpoint.py        # Save/load model checkpoints
│
├── metrics/                 # FR-008 to FR-010: Evaluation metrics
│   ├── __init__.py
│   ├── concordance.py       # C-index calculation
│   ├── brier.py             # Integrated Brier Score
│   └── likelihood.py        # Negative log partial likelihood
│
├── experiments/             # FR-012 to FR-021: Experiment orchestration
│   ├── __init__.py
│   ├── runner.py            # Serial execution, resumption logic
│   ├── config.py            # Experiment configuration (JSON schema)
│   ├── sweep.py             # Width/depth sweep management
│   └── logging.py           # TensorBoard + CSV logging
│
├── visualization/           # FR-011, FR-011a: Visualization
│   ├── __init__.py
│   ├── curves.py            # Double descent curve plots
│   ├── comparison.py        # Multi-scenario comparison
│   └── divergence.py        # Metric divergence analysis
│
└── cli/                     # CLI entry points
    ├── __init__.py
    ├── run_experiment.py    # Main experiment runner
    ├── generate_data.py     # Standalone data generation
    ├── visualize.py         # Post-hoc visualization
    └── compare.py           # Cross-experiment comparison

tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_generator.py    # Data generation tests
│   ├── test_deepsurv.py     # Model architecture tests
│   ├── test_metrics.py      # Metric calculation tests
│   └── test_config.py       # Configuration validation tests
├── integration/
│   ├── test_experiment.py   # Full experiment pipeline
│   └── test_resumption.py   # Checkpoint/resume behavior
└── contract/
    └── test_cli.py          # CLI interface contracts

configs/
├── scenarios/               # Predefined scenario configs
│   ├── baseline.json
│   ├── skewed.json
│   ├── high_cardinality.json
│   └── imbalanced.json
└── experiments/             # Example experiment configs
    └── example_sweep.json

outputs/                     # Generated outputs (gitignored)
├── experiments/             # Experiment results
│   └── {experiment_id}/
│       ├── config.json
│       ├── metrics.csv
│       ├── checkpoints/
│       └── tensorboard/
└── figures/                 # Generated visualizations
    ├── png/
    └── pdf/
```

**Structure Decision**: Single project structure with modular src/ layout. Separates concerns into data generation, model training, metrics, experiment orchestration, and visualization. CLI provides entry points for all major workflows.

## Complexity Tracking

No constitution violations requiring justification. The design follows simplicity principles:
- File-based storage (no database)
- Serial execution (no distributed computing)
- Standard Python project structure
