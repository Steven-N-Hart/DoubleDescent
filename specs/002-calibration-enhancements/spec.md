# Feature Specification: Calibration Decomposition and Experimental Enhancements

**Feature Branch**: `002-calibration-enhancements`
**Created**: 2026-01-19
**Status**: Draft
**Input**: Add calibration decomposition analysis, increase seeds to 20, change categorical cardinality to 10, and update manuscript text

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Calibration Decomposition Analysis (Priority: P1)

As a survival analysis researcher, I want to see calibration broken down into components (calibration-in-the-large, calibration slope, and Integrated Calibration Index) so that I can understand whether neural Cox models fail at overall event rate prediction, risk stratification, or both.

**Why this priority**: The calibration decomposition directly addresses reviewer feedback about breaking down calibration beyond the aggregate IBS metric. This is the core scientific enhancement that strengthens the manuscript's contribution.

**Independent Test**: Can be fully tested by running a single experiment configuration and verifying that three new calibration metrics appear in output files alongside existing C-index and IBS values.

**Acceptance Scenarios**:

1. **Given** a trained DeepSurv model with predictions, **When** the evaluation runs, **Then** three additional calibration metrics (calibration-in-the-large, calibration slope, ICI) are computed and saved to results.
2. **Given** baseline models (Cox PH, RSF), **When** baseline evaluation runs, **Then** the same three calibration metrics are computed for comparison.
3. **Given** completed experiments with calibration metrics, **When** results are aggregated across seeds, **Then** mean and standard deviation are reported for all calibration metrics.

---

### User Story 2 - Increased Statistical Power via 20 Seeds (Priority: P2)

As a researcher, I want experiments to run with 20 random seeds instead of 5 so that confidence intervals are tighter and statistical claims (e.g., the 2% regularization benefit) can be properly evaluated for significance.

**Why this priority**: Tighter confidence bands strengthen all statistical claims in the manuscript. The current 5-seed design produces wide error bars that limit interpretability.

**Independent Test**: Can be tested by running a small experiment configuration with 20 seeds and verifying output files contain results for all 20 seeds with aggregated statistics.

**Acceptance Scenarios**:

1. **Given** the multi-seed runner script, **When** executed with default settings, **Then** experiments run for 20 different seed values.
2. **Given** 20 completed seed runs, **When** results are aggregated, **Then** standard errors decrease by approximately 2x compared to 5-seed runs.
3. **Given** the full pipeline script, **When** executed, **Then** all experiments (baseline, extended width, baselines) use the same 20 seeds.

---

### User Story 3 - Reduced Categorical Cardinality (Priority: P2)

As a researcher, I want categorical experiments to use 10 levels per feature instead of 100 so that each category has sufficient samples for meaningful learning and results are more clinically relevant.

**Why this priority**: The current 100-level configuration results in approximately 2 observations per category, which is insufficient. Reducing to 10 levels provides approximately 20 observations per category, enabling meaningful analysis.

**Independent Test**: Can be tested by running the categorical scenario and verifying that models achieve better-than-random concordance (C-index > 0.55).

**Acceptance Scenarios**:

1. **Given** categorical scenario configuration, **When** data is generated, **Then** each of 5 categorical features has exactly 10 levels.
2. **Given** 10-level categorical features with 1000 samples, **When** distributed across categories, **Then** each category has approximately 100 observations on average.
3. **Given** the reduced cardinality scenario, **When** DeepSurv models are trained, **Then** concordance index shows meaningful variation across model widths rather than constant near-random performance.

---

### User Story 4 - Manuscript Text Clarifications (Priority: P3)

As a manuscript reader, I want clear descriptions of the covariate structure, consistent terminology for the threshold region, and accurate scenario descriptions so that methods and results are unambiguous.

**Why this priority**: Addresses specific reviewer questions about terminology and missing methodological details. Lower priority because it does not affect experimental results.

**Independent Test**: Can be tested by reviewing the manuscript for specific text changes and verifying they address the reviewer's questions.

**Acceptance Scenarios**:

1. **Given** the Methods section, **When** reading the Data Generation subsection, **Then** the covariate structure (10 predictive features in [-1,1], 10 noise features at 0) is explicitly stated.
2. **Given** references to "critical region", **When** reading the text, **Then** the term is either defined or replaced with "threshold region" consistently.
3. **Given** the Scenarios table, **When** reading Scenario C description, **Then** K=10 levels is shown instead of K=100.
4. **Given** references to "replicated experiments", **When** reading the text, **Then** the phrasing clarifies whether this means seed replication or external validation.
5. **Given** the Introduction section, **When** reading about calibration, **Then** the calibration decomposition framework is introduced with appropriate citations.
6. **Given** the references, **When** checking bibliography, **Then** Austin et al. (2020), Crowson et al. (2016), and Royston & Altman (2013) are included.

---

### Edge Cases

- What happens when calibration slope cannot be computed (e.g., all events occur at same time)? System should return NaN and log a warning.
- How does system handle zero events in a risk group for ICI computation? System should use appropriate handling for empty bins.
- What happens if a seed run fails partway through? Aggregation should handle incomplete seed sets gracefully.

## Requirements *(mandatory)*

### Functional Requirements

**Calibration Metrics:**
- **FR-001**: System MUST compute calibration-in-the-large as the ratio of observed to expected events (O/E ratio).
- **FR-002**: System MUST compute calibration slope by fitting a model with the linear predictor as the sole covariate, where slope of 1.0 indicates perfect calibration.
- **FR-003**: System MUST compute Integrated Calibration Index (ICI) as the mean absolute difference between predicted and smoothed observed survival probabilities.
- **FR-004**: System MUST compute calibration metrics for both neural network models and classical baselines (Cox PH, RSF).
- **FR-005**: System MUST include calibration metrics in all result outputs (CSV files, aggregated summaries).

**Seed Configuration:**
- **FR-006**: System MUST use 20 random seeds as the default for multi-seed experiments.
- **FR-007**: System MUST maintain reproducibility by using deterministic seed values.
- **FR-008**: System MUST aggregate results across all seeds, computing mean, standard deviation, and standard error.

**Categorical Configuration:**
- **FR-009**: System MUST generate categorical features with 10 levels per feature in the high-cardinality scenario.
- **FR-010**: System MUST update scenario descriptions to reflect the 10-level configuration.

**Manuscript Updates:**
- **FR-011**: Manuscript MUST explicitly describe the covariate structure (number of predictive vs noise features, coefficient ranges).
- **FR-012**: Manuscript MUST use consistent terminology for the interpolation threshold region.
- **FR-013**: Manuscript MUST update Table 1 to show K=10 for Scenario C.
- **FR-014**: Manuscript MUST clarify "replicated experiments" to distinguish seed replication from external validation.
- **FR-015**: Manuscript Introduction MUST discuss calibration decomposition framework (calibration-in-the-large, calibration slope, ICI) as established methodology for assessing prognostic model validity.
- **FR-016**: Manuscript MUST add citations for calibration methodology: Austin et al. (2020) for ICI in survival models, Crowson et al. (2016) for calibration assessment framework, and Royston & Altman (2013) for external validation principles.
- **FR-017**: Manuscript references.bib MUST include the three new calibration methodology citations.

### Key Entities

- **CalibrationResult**: Contains the three calibration decomposition metrics (calibration_in_the_large, calibration_slope, ici) along with the time horizon used for evaluation.
- **MetricResult**: Extended container that now includes calibration metrics alongside existing C-index, IBS, and NLL values.
- **Scenario Configuration**: Defines experimental parameters including categorical cardinality (now 10 instead of 100).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All three calibration metrics (calibration-in-the-large, calibration slope, ICI) are computed and saved for every model evaluation.
- **SC-002**: Classical baselines (Cox PH, RSF) show calibration-in-the-large close to 1.0 (within 0.8-1.2 range), confirming expected good calibration.
- **SC-003**: DeepSurv models show calibration slope significantly different from 1.0, quantifying the calibration failure.
- **SC-004**: Standard errors for all metrics decrease by approximately 50% compared to 5-seed experiments (due to 4x sample size increase).
- **SC-005**: Categorical scenario with K=10 achieves mean concordance index above 0.55 (better than random).
- **SC-006**: All manuscript text changes pass reviewer validation for clarity and completeness.

## Assumptions

- The calibration slope computation will use Cox regression, fitting the original risk scores as the sole predictor.
- The ICI computation will use LOESS smoothing at a fixed time horizon (median event time), consistent with Austin et al. (2020) methodology.
- The 15 additional seeds will be deterministic values chosen for reproducibility.
- The reduced cardinality (K=10) will require re-running the categorical experiments but not other scenarios.
- Manuscript text changes can be made independently of code changes.

## Clarifications

### Session 2026-01-19

- Q: ICI smoothing method (isotonic regression vs LOESS)? → A: LOESS smoothing, consistent with Austin et al. (2020) methodology
- Q: Time horizon for ICI evaluation (single vs multiple)? → A: Single time horizon at median event time from training data

## Out of Scope

- Adding new visualization panels for calibration decomposition (can be follow-up work).
- Implementing post-hoc recalibration methods (mentioned in discussion but not implemented).
- Testing DeepHit as an alternative to DeepSurv (mentioned as future work).
- Adding intermediate censoring rates (e.g., 60%) for N_events analysis.
