# Implementation Plan: Calibration Decomposition and Experimental Enhancements

**Branch**: `002-calibration-enhancements` | **Date**: 2026-01-19 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-calibration-enhancements/spec.md`

## Summary

Add calibration decomposition analysis (calibration-in-the-large, calibration slope, ICI) to the DoubleDescent survival analysis framework, increase experimental seeds from 5 to 20 for tighter confidence intervals, reduce categorical cardinality from 100 to 10 levels, and update the manuscript with methodology descriptions and citations.

## Technical Context

**Language/Version**: Python 3.8+ (compatible with 3.8-3.11 per pyproject.toml)
**Primary Dependencies**: PyTorch >=1.10, scikit-survival >=0.17, lifelines >=0.27, scipy >=1.7, numpy >=1.20
**Storage**: File-based (JSON configs, CSV metrics, PyTorch checkpoints)
**Testing**: pytest with pytest-cov
**Target Platform**: Linux server (research workstation)
**Project Type**: Single project (research framework)
**Performance Goals**: N/A (batch research computation, no latency requirements)
**Constraints**: Must integrate with existing MetricEvaluator pipeline; metrics computed after each epoch
**Scale/Scope**: ~1000 samples per experiment, 20 seeds × multiple width configurations

## Constitution Check

*GATE: No project constitution defined (template placeholders only). Proceeding with standard Python best practices.*

- [x] Code follows existing patterns in src/metrics/
- [x] Tests required for new metric functions
- [x] Changes are additive (extend existing classes, don't break interfaces)

## Project Structure

### Documentation (this feature)

```text
specs/002-calibration-enhancements/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── metrics/
│   ├── __init__.py          # UPDATE: Export new calibration functions
│   ├── calibration.py       # NEW: Calibration decomposition metrics
│   ├── evaluator.py         # UPDATE: Add calibration computation
│   └── results.py           # UPDATE: Add calibration fields to MetricResult
├── models/
│   └── baselines.py         # UPDATE: Add calibration to baseline evaluation
└── experiments/
    └── aggregation.py       # UPDATE: Aggregate calibration metrics

scripts/
├── run_multi_seed.py        # UPDATE: Expand DEFAULT_SEEDS to 20
└── run_full_pipeline.sh     # UPDATE: Use 20 seeds in loops

configs/
└── scenarios/
    └── high_cardinality.json  # UPDATE: Change cardinality from 100 to 10

manuscript/
├── main.tex                 # UPDATE: Add covariate description, terminology, citations
└── references.bib           # UPDATE: Add 3 new calibration citations

tests/
└── unit/
    └── test_calibration.py  # NEW: Unit tests for calibration metrics
```

**Structure Decision**: Single project structure maintained. Changes are additive extensions to existing modules.

## Complexity Tracking

No constitution violations. All changes follow existing patterns.
