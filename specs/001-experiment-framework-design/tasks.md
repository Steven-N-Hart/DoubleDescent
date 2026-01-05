# Tasks: Experiment Framework Design

**Input**: Design documents from `/specs/001-experiment-framework-design/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: ADVISORY - Basic tests for core components (data generation, metrics) are included as they are critical for research reproducibility.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths follow plan.md structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure per plan.md (src/data/, src/models/, src/metrics/, src/experiments/, src/visualization/, src/cli/)
- [x] T002 Create package __init__.py files for all modules
- [x] T003 [P] Create configs/ directory with scenarios/ and experiments/ subdirectories
- [x] T004 [P] Create outputs/ directory structure (experiments/, figures/png/, figures/pdf/) and add to .gitignore
- [x] T005 [P] Create JSON schemas for experiment and scenario configs in configs/schemas/
- [x] T006 [P] Setup pytest configuration in pyproject.toml or pytest.ini
- [x] T007 Create tests/ directory structure (unit/, integration/, contract/, conftest.py)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Core Data Types

- [x] T008 [P] Create enumerations (CovariateType, SweepType, ExperimentStatus, RunStatus) in src/data/types.py
- [x] T009 [P] Create DataScenario dataclass in src/data/scenarios.py
- [x] T010 [P] Create ModelConfiguration dataclass in src/models/config.py
- [x] T011 [P] Create Experiment dataclass in src/experiments/config.py
- [x] T012 [P] Create ExperimentRun dataclass in src/experiments/run.py
- [x] T013 [P] Create MetricResult dataclass in src/metrics/results.py
- [x] T014 Create DoubleDescentCurve dataclass in src/visualization/curves.py

### Configuration Validation

- [x] T015 Implement JSON schema validation for experiment configs in src/experiments/config.py
- [x] T016 Implement JSON schema validation for scenario configs in src/data/scenarios.py
- [x] T017 [P] Create predefined scenario configs in configs/scenarios/ (baseline.json, skewed.json, high_cardinality.json, imbalanced.json)

### Metrics Foundation

- [x] T018 [P] Implement C-index calculation using scikit-survival in src/metrics/concordance.py
- [x] T019 [P] Implement Integrated Brier Score calculation in src/metrics/brier.py
- [x] T020 [P] Implement Negative Log Partial Likelihood calculation in src/metrics/likelihood.py
- [x] T021 Create metrics aggregator (compute all metrics for a model) in src/metrics/__init__.py

### Data Generation Foundation

- [x] T022 Implement Gaussian copula for correlated covariates in src/data/copula.py
- [x] T023 Implement covariate generators (Gaussian, LogNormal, Categorical, Mixed) in src/data/generator.py
- [x] T024 Implement Weibull survival time generation using inverse transform in src/data/generator.py
- [x] T025 Implement sparse coefficient generation (first K predictive) in src/data/generator.py
- [x] T026 Implement censoring rate calibration via binary search in src/data/censoring.py
- [x] T027 Implement train/val/test data splitting with seed control in src/data/generator.py

### Model Foundation

- [x] T028 Implement DeepSurv MLP architecture (configurable width/depth) in src/models/deepsurv.py
- [x] T029 Implement Cox partial likelihood loss function in src/models/deepsurv.py
- [x] T030 Implement parameter count calculation for P/N ratio in src/models/deepsurv.py

### Logging Infrastructure

- [x] T031 [P] Implement TensorBoard logging wrapper in src/experiments/logging.py
- [x] T032 [P] Implement CSV metrics writer in src/experiments/logging.py
- [x] T033 Implement combined experiment logger (TensorBoard + CSV) in src/experiments/logging.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Run Baseline Double Descent Experiment (Priority: P1) 🎯 MVP

**Goal**: Execute complete baseline experiment (data generation → training → visualization) in a single workflow

**Independent Test**: Run experiment with small config (3 widths, 100 epochs) and verify metrics CSV and double descent plot are generated

### Tests for User Story 1

- [x] T034 [P] [US1] Unit test for data generation (correct distributions, censoring rate) in tests/unit/test_generator.py
- [x] T035 [P] [US1] Unit test for DeepSurv forward pass and loss in tests/unit/test_deepsurv.py
- [x] T036 [P] [US1] Unit test for metric calculations (C-index, IBS) in tests/unit/test_metrics.py
- [x] T037 [P] [US1] Integration test for full experiment pipeline in tests/integration/test_experiment.py

### Training Loop Implementation

- [x] T038 [US1] Implement training loop with epoch-level metric logging in src/models/trainer.py
- [x] T039 [US1] Implement gradient norm and weight norm tracking per epoch in src/models/trainer.py
- [x] T040 [US1] Implement batch loss variance calculation in src/models/trainer.py
- [x] T041 [US1] Implement training failure detection (NaN loss, divergence) in src/models/trainer.py
- [x] T042 [US1] Implement retry with reduced learning rate on failure in src/models/trainer.py
- [x] T043 [US1] Implement GPU OOM handling (catch, log, skip) in src/models/trainer.py

### Checkpointing

- [x] T044 [US1] Implement model checkpoint saving (model + optimizer state) in src/models/checkpoint.py
- [x] T045 [US1] Implement checkpoint loading for resumption in src/models/checkpoint.py
- [x] T046 [US1] Implement progress.json for tracking completed widths in src/experiments/runner.py

### Experiment Runner

- [x] T047 [US1] Implement width sweep orchestration (serial execution) in src/experiments/sweep.py
- [x] T048 [US1] Implement experiment resumption logic (detect completed, continue from next) in src/experiments/runner.py
- [x] T049 [US1] Implement experiment output directory creation and management in src/experiments/runner.py
- [x] T050 [US1] Implement main experiment runner (data gen → sweep → results) in src/experiments/runner.py

### Basic Visualization

- [x] T051 [US1] Implement double descent curve plot (test error vs capacity) in src/visualization/curves.py
- [x] T052 [US1] Implement PNG and PDF export for plots in src/visualization/curves.py

### CLI: run_experiment

- [x] T053 [US1] Implement run_experiment CLI with argparse in src/cli/run_experiment.py
- [x] T054 [US1] Add --config, --output-dir, --resume, --dry-run, --device, --verbose flags
- [x] T055 [US1] Implement exit codes (0=success, 1=config error, 2=runtime error, 3=partial)
- [x] T056 [US1] Add progress output to stdout and errors to stderr

### CLI: status

- [x] T057 [P] [US1] Implement status CLI for checking experiment progress in src/cli/status.py
- [x] T058 [P] [US1] Add --experiment, --all, --json flags to status CLI

### Example Config

- [x] T059 [US1] Create example experiment config in configs/experiments/example_sweep.json

**Checkpoint**: User Story 1 complete - can run full baseline experiment end-to-end

---

## Phase 4: User Story 2 - Compare Experiments Across Data Scenarios (Priority: P2)

**Goal**: Generate comparative visualizations across multiple completed experiments

**Independent Test**: Run two experiments (baseline, skewed) and generate comparison plot showing both curves

### Tests for User Story 2

- [ ] T060 [P] [US2] Unit test for comparison plot generation in tests/unit/test_comparison.py
- [ ] T061 [P] [US2] Contract test for compare CLI in tests/contract/test_cli.py

### Comparison Visualization

- [ ] T062 [US2] Implement multi-experiment curve overlay in src/visualization/comparison.py
- [ ] T063 [US2] Implement peak analysis visualization (location/height comparison) in src/visualization/comparison.py
- [ ] T064 [US2] Implement summary table generation (CSV) in src/visualization/comparison.py
- [ ] T065 [US2] Implement comparison report generation (Markdown) in src/visualization/comparison.py

### CLI: compare

- [ ] T066 [US2] Implement compare CLI with argparse in src/cli/compare.py
- [ ] T067 [US2] Add --experiments, --output, --metric, --labels, --format flags

**Checkpoint**: User Story 2 complete - can compare experiments across scenarios

---

## Phase 5: User Story 3 - Analyze Metric Divergence (Priority: P2)

**Goal**: Visualize and quantify C-index vs IBS divergence at interpolation threshold

**Independent Test**: Load completed experiment, generate dual-axis divergence plot with highlighted regions

### Tests for User Story 3

- [ ] T068 [P] [US3] Unit test for divergence calculations in tests/unit/test_divergence.py

### Divergence Analysis

- [ ] T069 [US3] Implement dual-axis plot (C-index vs IBS) in src/visualization/divergence.py
- [ ] T070 [US3] Implement interpolation threshold detection in src/visualization/divergence.py
- [ ] T071 [US3] Implement divergence region highlighting in src/visualization/divergence.py
- [ ] T072 [US3] Implement divergence statistics calculation in src/visualization/divergence.py

### CLI: visualize (extended)

- [ ] T073 [US3] Add metric_divergence plot type to visualize CLI in src/cli/visualize.py
- [ ] T074 [US3] Add --plots flag support for selecting specific plot types

**Checkpoint**: User Story 3 complete - can analyze metric divergence

---

## Phase 6: User Story 4 - Configure Synthetic Data Generation (Priority: P3)

**Goal**: Allow custom data generation beyond predefined scenarios

**Independent Test**: Create custom scenario config with log-normal covariates, generate data, verify distribution matches

### Tests for User Story 4

- [ ] T075 [P] [US4] Unit test for custom scenario parsing in tests/unit/test_config.py
- [ ] T076 [P] [US4] Contract test for generate_data CLI in tests/contract/test_cli.py

### Extended Data Generation

- [ ] T077 [US4] Implement custom correlation matrix support in src/data/copula.py
- [ ] T078 [US4] Implement entity embeddings for categorical variables in src/data/generator.py
- [ ] T079 [US4] Implement custom Weibull parameter configuration in src/data/generator.py

### CLI: generate_data

- [ ] T080 [US4] Implement generate_data CLI with argparse in src/cli/generate_data.py
- [ ] T081 [US4] Add --scenario, --config, --output, --n-samples, --seed, --format flags
- [ ] T082 [US4] Implement NPZ, CSV, and Parquet output formats

**Checkpoint**: User Story 4 complete - can generate custom synthetic data

---

## Phase 7: User Story 5 - Reproduce Experiments with Seeds (Priority: P3)

**Goal**: Ensure exact reproducibility with seed control

**Independent Test**: Run same experiment twice with same seed, verify metrics are identical within floating-point tolerance

### Tests for User Story 5

- [ ] T083 [P] [US5] Integration test for reproducibility (same seed → same results) in tests/integration/test_reproducibility.py

### Reproducibility Implementation

- [ ] T084 [US5] Implement global seed setting (numpy, torch, python random) in src/experiments/runner.py
- [ ] T085 [US5] Implement CUDA determinism settings in src/experiments/runner.py
- [ ] T086 [US5] Add seed to all saved configs for traceability in src/experiments/config.py
- [ ] T087 [US5] Implement experiment config export from completed experiment

### Resumption Enhancement

- [ ] T088 [US5] Implement checkpoint validation (verify resuming same config) in src/experiments/runner.py
- [ ] T089 [US5] Add tests for resumption correctness in tests/integration/test_resumption.py

**Checkpoint**: User Story 5 complete - experiments are fully reproducible

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### CLI Enhancements

- [ ] T090 [P] Implement visualize CLI (all plot types) in src/cli/visualize.py
- [ ] T091 [P] Add --help, --version, --quiet, --log-level, --log-file to all CLIs
- [ ] T092 [P] Add training_curves, gradient_norms, weight_norms plot types

### Additional Visualizations

- [ ] T093 [P] Implement training curves plot (loss over epochs per width) in src/visualization/curves.py
- [ ] T094 [P] Implement gradient norm evolution plot in src/visualization/curves.py
- [ ] T095 [P] Implement weight norm evolution plot in src/visualization/curves.py
- [ ] T096 [P] Implement parameter histogram logging to TensorBoard

### Code Quality

- [ ] T097 [P] Add type hints to all public functions
- [ ] T098 [P] Add docstrings (Google style) to all modules
- [ ] T099 Run mypy type checking and fix errors
- [ ] T100 Run ruff/flake8 linting and fix issues
- [ ] T101 Run black formatting on all source files

### Documentation

- [ ] T102 Validate quickstart.md instructions work end-to-end
- [ ] T103 Update CLAUDE.md with final project structure

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (P1): No dependencies on other stories
  - US2 (P2): Requires experiments to exist (can start after US1 checkpoint)
  - US3 (P2): Requires experiment results (can start after US1 checkpoint)
  - US4 (P3): Independent of other stories
  - US5 (P3): Independent of other stories
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

```
Phase 2 (Foundational)
        │
        ▼
   ┌────┴────┐
   │         │
   ▼         ▼
  US1 ────► US2
   │         │
   │    ┌────┘
   ▼    ▼
   US3

  US4 (independent)
  US5 (independent)
```

### Within Each User Story

- Tests (if included) written first
- Data types before functions that use them
- Core implementation before CLI wrappers
- Commit after each task or logical group

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005, T006 can all run in parallel

**Phase 2 (Foundational)**:
- T008-T014 (dataclasses) can all run in parallel
- T018-T020 (metrics) can all run in parallel
- T031-T032 (logging) can run in parallel

**Phase 3 (US1)**:
- T034-T037 (tests) can all run in parallel
- T057-T058 (status CLI) can run parallel with main implementation

**Each Story Phase**:
- Tests can run in parallel with each other
- Independent modules can be developed in parallel

---

## Parallel Example: Phase 2 Foundational

```bash
# Launch all dataclass definitions in parallel:
Task: "Create enumerations in src/data/types.py"
Task: "Create DataScenario dataclass in src/data/scenarios.py"
Task: "Create ModelConfiguration dataclass in src/models/config.py"
Task: "Create Experiment dataclass in src/experiments/config.py"

# Then launch all metrics in parallel:
Task: "Implement C-index calculation in src/metrics/concordance.py"
Task: "Implement Integrated Brier Score in src/metrics/brier.py"
Task: "Implement Negative Log Partial Likelihood in src/metrics/likelihood.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run baseline experiment end-to-end
5. Can now produce double descent curves from synthetic data

### Incremental Delivery

1. **Setup + Foundational** → Core infrastructure ready
2. **Add User Story 1** → Can run experiments and visualize results (MVP!)
3. **Add User Story 2** → Can compare across scenarios
4. **Add User Story 3** → Can analyze metric divergence
5. **Add User Story 4** → Can customize data generation
6. **Add User Story 5** → Full reproducibility guaranteed
7. Each story adds research capability without breaking previous stories

### Suggested MVP Scope

**Phase 1 + Phase 2 + Phase 3 (User Story 1)** delivers:
- Synthetic data generation for baseline scenario
- DeepSurv model training with width sweep
- C-index, IBS, NLL metrics per epoch
- TensorBoard logging
- Double descent curve visualization
- Checkpoint/resume capability

This is sufficient to answer the primary research question: "Does double descent occur in survival models?"

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests pass after each story checkpoint
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently

## Summary

| Phase | User Story | Tasks | Parallel Tasks |
|-------|------------|-------|----------------|
| Phase 1 | Setup | 7 | 4 |
| Phase 2 | Foundational | 26 | 15 |
| Phase 3 | US1 - Baseline Experiment | 26 | 6 |
| Phase 4 | US2 - Compare Scenarios | 8 | 2 |
| Phase 5 | US3 - Metric Divergence | 7 | 1 |
| Phase 6 | US4 - Custom Data Gen | 8 | 2 |
| Phase 7 | US5 - Reproducibility | 7 | 1 |
| Phase 8 | Polish | 12 | 8 |
| **Total** | | **101** | **39** |
