# Tasks: Calibration Decomposition and Experimental Enhancements

**Input**: Design documents from `/specs/002-calibration-enhancements/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: Unit tests included as specified in plan.md Constitution Check.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify dependencies and prepare development environment

- [ ] T001 Verify statsmodels is available for LOESS smoothing (add to pyproject.toml if missing)
- [ ] T002 [P] Verify lifelines CoxPHFitter is available for calibration slope computation

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before user stories can proceed

**⚠️ CRITICAL**: User Story 1 depends on these foundational components

- [ ] T003 Create src/metrics/calibration.py with calibration_in_the_large function per research.md
- [ ] T004 Add calibration_slope function to src/metrics/calibration.py using lifelines CoxPHFitter
- [ ] T005 Add integrated_calibration_index function to src/metrics/calibration.py using LOESS smoothing
- [ ] T006 Add CalibrationResult dataclass to src/metrics/calibration.py per data-model.md
- [ ] T007 Export calibration functions from src/metrics/__init__.py

**Checkpoint**: Calibration metric functions available for integration

---

## Phase 3: User Story 1 - Calibration Decomposition Analysis (Priority: P1) 🎯 MVP

**Goal**: Add three calibration metrics (calibration-in-the-large, calibration slope, ICI) to all model evaluations

**Independent Test**: Run single experiment and verify three new calibration metrics appear in output CSV alongside C-index and IBS

### Tests for User Story 1

- [ ] T008 [P] [US1] Create tests/unit/test_calibration.py with test for perfect calibration (O/E ≈ 1.0)
- [ ] T009 [P] [US1] Add test for over-prediction scenario (O/E < 1, slope < 1) in tests/unit/test_calibration.py
- [ ] T010 [P] [US1] Add test for under-prediction scenario (O/E > 1, slope > 1) in tests/unit/test_calibration.py
- [ ] T011 [P] [US1] Add test for edge cases (all events, no events, degenerate) returning NaN in tests/unit/test_calibration.py

### Implementation for User Story 1

- [ ] T012 [US1] Add calibration fields to MetricResult dataclass in src/metrics/results.py per data-model.md
- [ ] T013 [US1] Update MetricResult.to_dict() and to_csv_row() to include calibration fields in src/metrics/results.py
- [ ] T014 [US1] Update MetricResult.from_dict() for backward compatibility in src/metrics/results.py
- [ ] T015 [US1] Add _compute_calibration method to MetricEvaluator in src/metrics/evaluator.py
- [ ] T016 [US1] Call _compute_calibration in MetricEvaluator.evaluate() after IBS computation in src/metrics/evaluator.py
- [ ] T017 [US1] Add calibration fields to RunSummary dataclass in src/metrics/results.py
- [ ] T018 [US1] Update RunSummary.to_dict() and to_csv_row() in src/metrics/results.py
- [ ] T019 [US1] Add calibration metrics to BaselineResults dataclass in src/models/baselines.py
- [ ] T020 [US1] Update CoxPHBaseline.evaluate() to compute calibration metrics in src/models/baselines.py
- [ ] T021 [US1] Update RandomSurvivalForestBaseline.evaluate() to compute calibration metrics in src/models/baselines.py
- [ ] T022 [US1] Update aggregation to include calibration metrics in src/experiments/aggregation.py

**Checkpoint**: Running `python -m src.cli.run_experiment --config configs/experiments/baseline_sweep.json` produces CSV with cal_large, cal_slope, ici columns

---

## Phase 4: User Story 2 - Increased Statistical Power via 20 Seeds (Priority: P2)

**Goal**: Expand default seeds from 5 to 20 for tighter confidence intervals

**Independent Test**: Run multi-seed script and verify 20 seed values are used by default

### Implementation for User Story 2

- [ ] T023 [US2] Update DEFAULT_SEEDS in scripts/run_multi_seed.py to 20 seeds per research.md
- [ ] T024 [US2] Update seed loops in scripts/run_full_pipeline.sh to use all 20 seeds

**Checkpoint**: Running `python scripts/run_multi_seed.py --config configs/experiments/baseline_sweep.json --dry-run` shows 20 seeds

---

## Phase 5: User Story 3 - Reduced Categorical Cardinality (Priority: P2)

**Goal**: Change categorical cardinality from 100 to 10 levels for meaningful learning

**Independent Test**: Generate categorical data and verify 10 levels per feature

### Implementation for User Story 3

- [ ] T025 [US3] Update cardinality from 100 to 10 in configs/scenarios/high_cardinality.json
- [ ] T026 [US3] Update description in configs/scenarios/high_cardinality.json to reflect 10 levels

**Checkpoint**: `cat configs/scenarios/high_cardinality.json | grep cardinality` shows 10

---

## Phase 6: User Story 4 - Manuscript Text Clarifications (Priority: P3)

**Goal**: Update manuscript with calibration framework description, terminology fixes, and citations

**Independent Test**: Build manuscript and verify new citations appear in references

### Implementation for User Story 4

- [ ] T027 [P] [US4] Add Austin et al. (2020) BibTeX entry to manuscript/references.bib per research.md
- [ ] T028 [P] [US4] Add Crowson et al. (2016) BibTeX entry to manuscript/references.bib per research.md
- [ ] T029 [P] [US4] Add Royston & Altman (2013) BibTeX entry to manuscript/references.bib per research.md
- [ ] T030 [US4] Add calibration decomposition paragraph to Introduction (after line 93) in manuscript/main.tex per research.md
- [ ] T031 [US4] Add covariate structure description to Methods Section 3.1 (after line 165) in manuscript/main.tex
- [ ] T032 [US4] Replace "critical region" with "threshold region" in manuscript/main.tex (line 277)
- [ ] T033 [US4] Update Table 1 Scenario C from K=100 to K=10 in manuscript/main.tex (line 183)
- [ ] T034 [US4] Clarify "replicated experiments" phrasing in manuscript/main.tex (line 311)
- [ ] T035 [US4] Update categorical scenario discussion to reflect K=10 in manuscript/main.tex (lines 323-324)

**Checkpoint**: Run `cd manuscript && ./build.sh` successfully with no missing citation warnings

---

## Phase 7: Polish & Validation

**Purpose**: Final validation and cleanup

- [ ] T036 Run pytest tests/unit/test_calibration.py -v to verify all calibration tests pass
- [ ] T037 Run single experiment to verify calibration metrics in output
- [ ] T038 Run quickstart.md verification steps
- [ ] T039 Verify backward compatibility by loading old results without calibration fields

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS User Story 1
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2)
- **User Story 2 (Phase 4)**: No dependencies on other stories - can start after Phase 1
- **User Story 3 (Phase 5)**: No dependencies on other stories - can start after Phase 1
- **User Story 4 (Phase 6)**: No dependencies on other stories - can start after Phase 1
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Foundational (calibration.py must exist first)
- **User Story 2 (P2)**: Independent - only modifies seed configuration
- **User Story 3 (P2)**: Independent - only modifies scenario configuration
- **User Story 4 (P3)**: Independent - only modifies manuscript files

### Within User Story 1

- Tests (T008-T011) can run in parallel
- T012-T014 (MetricResult) before T015-T016 (MetricEvaluator)
- T017-T018 (RunSummary) can parallel with T015-T016
- T019-T021 (Baselines) after MetricResult changes
- T022 (Aggregation) after all metric changes

### Parallel Opportunities

**After Phase 1 completes:**
- User Story 2, 3, and 4 can all start in parallel (different files, no dependencies)
- User Story 1 must wait for Phase 2 (Foundational)

**Within Phase 2:**
- T003-T006 are sequential (building calibration.py incrementally)
- T007 depends on T003-T006

**Within User Story 1 tests:**
- T008, T009, T010, T011 can all run in parallel

**Within User Story 4:**
- T027, T028, T029 (BibTeX entries) can run in parallel
- T030-T035 (main.tex edits) are sequential to avoid conflicts

---

## Parallel Example: User Story 2, 3, 4 Together

```bash
# These can run simultaneously after Phase 1:

# User Story 2 (Developer A):
Task: "Update DEFAULT_SEEDS in scripts/run_multi_seed.py"
Task: "Update seed loops in scripts/run_full_pipeline.sh"

# User Story 3 (Developer B):
Task: "Update cardinality in configs/scenarios/high_cardinality.json"

# User Story 4 (Developer C):
Task: "Add BibTeX entries to manuscript/references.bib"
Task: "Update manuscript/main.tex with text changes"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (verify dependencies)
2. Complete Phase 2: Foundational (create calibration.py)
3. Complete Phase 3: User Story 1 (integrate calibration metrics)
4. **STOP and VALIDATE**: Run experiment, verify calibration columns in CSV
5. Baselines (Cox PH, RSF) should show O/E ≈ 1.0; DeepSurv should show slope ≠ 1.0

### Incremental Delivery

1. Complete Setup + Foundational + US1 → Calibration metrics working (MVP!)
2. Add User Story 2 → 20 seeds available
3. Add User Story 3 → Categorical scenario improved
4. Add User Story 4 → Manuscript updated
5. Each story adds value without breaking previous stories

### Recommended Execution Order

For a single developer working sequentially:
1. Phase 1: Setup (T001-T002)
2. Phase 2: Foundational (T003-T007)
3. Phase 3: US1 Tests (T008-T011)
4. Phase 3: US1 Implementation (T012-T022)
5. Phase 4: US2 (T023-T024)
6. Phase 5: US3 (T025-T026)
7. Phase 6: US4 (T027-T035)
8. Phase 7: Polish (T036-T039)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- User Stories 2, 3, 4 have no code dependencies on User Story 1
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
