# Feature Specification: Experiment Framework Design

**Feature Branch**: `001-experiment-framework-design`
**Created**: 2026-01-04
**Status**: Draft
**Input**: User description: "Based on the documents, we need to create a design plan and how we will structure the repository and experiments"

## Clarifications

### Session 2026-01-04

- Q: Should the framework prioritize single comprehensive experiments or support parallel/distributed execution? → A: Serial execution - one model at a time, simpler implementation, easier debugging
- Q: When an experiment is interrupted mid-sweep, what should happen on restart? → A: Resume from checkpoint - detect completed widths and continue from the next pending one
- Q: What format should be used for storing experiment results and configurations? → A: JSON configs + CSV metrics - human-readable configs, tabular metrics for easy analysis
- Q: How should the system handle model configurations that fail to train? → A: Retry with fallback - attempt training once more with reduced learning rate, then skip if still fails
- Q: At what frequency should evaluation metrics be computed during training? → A: Epoch-based logging - evaluate every epoch and store full training curves for maximum flexibility
- Q: Should network depth be fixed or configurable? → A: Depth sweep experiments - support varying both depth and width as separate experiment types
- Q: How should the true model coefficients be generated for synthetic data? → A: Sparse signal - only first K features have non-zero coefficients (e.g., 10 of 20), rest are noise
- Q: Should additional training diagnostics be logged beyond core survival metrics? → A: Full diagnostics - gradient norm, weight norm (L2), learning rate, batch loss variance, and parameter histograms
- Q: What should happen when a model configuration exceeds available GPU memory? → A: Skip configuration - log OOM error, record as failed, continue to next width
- Q: What format should visualizations be saved in? → A: Both PNG and PDF (for viewing and publications), plus TensorBoard logs for interactive exploration

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Baseline Double Descent Experiment (Priority: P1)

A researcher wants to run the baseline experiment (Scenario A) to confirm whether double descent exists in DeepSurv models with standard Gaussian covariates. They configure the sample size, feature count, and model width range, then execute the experiment pipeline which generates synthetic data, trains models across the capacity sweep, evaluates metrics, and produces visualizations of the double descent curve.

**Why this priority**: This is the foundational experiment that validates the core hypothesis. All other experiments build on this baseline.

**Independent Test**: Can be fully tested by running a single experiment configuration and verifying that C-index and Integrated Brier Score curves are generated across the model width sweep, delivering the primary evidence of double descent presence.

**Acceptance Scenarios**:

1. **Given** a configured baseline experiment with Gaussian covariates, **When** the researcher executes the experiment pipeline, **Then** the system generates synthetic survival data, trains DeepSurv models across the specified width range, and outputs evaluation metrics (C-index, IBS, NLL) for each model configuration.
2. **Given** completed experiment runs, **When** the researcher requests visualization, **Then** the system produces plots showing test error vs model capacity with clear indication of the interpolation threshold region.
3. **Given** an experiment in progress, **When** the researcher checks status, **Then** the system reports current progress including completed width configurations and estimated remaining time.

---

### User Story 2 - Compare Experiments Across Data Scenarios (Priority: P2)

A researcher wants to compare double descent behavior across different data scenarios (Baseline, Skewed, High-Cardinality, Imbalanced) to test hypotheses about how data characteristics affect the interpolation threshold and peak severity.

**Why this priority**: Comparative analysis across scenarios is essential to answer the research questions about distributional sensitivity and censoring effects.

**Independent Test**: Can be tested by running at least two different scenario experiments and generating comparative visualizations showing how the double descent curves differ.

**Acceptance Scenarios**:

1. **Given** completed experiments for multiple scenarios, **When** the researcher requests comparison analysis, **Then** the system generates side-by-side visualizations of double descent curves with annotations highlighting peak location differences.
2. **Given** skewed data experiment results alongside baseline results, **When** the researcher analyzes peak characteristics, **Then** the system reports quantitative differences in peak height, width, and location.
3. **Given** experiments with different censoring rates (20%, 50%, 90%), **When** the researcher compares interpolation thresholds, **Then** the system calculates and displays how the peak location relates to the number of events vs total sample size.

---

### User Story 3 - Analyze Metric Divergence (Priority: P2)

A researcher wants to investigate the hypothesis that C-index fails to detect overfitting that Brier Score reveals. They run experiments and analyze how the two metrics diverge at the interpolation threshold.

**Why this priority**: Demonstrating metric divergence is a key contribution of the research, providing practical guidance for model selection.

**Independent Test**: Can be tested by examining the correlation between C-index and IBS across model widths, particularly around the interpolation threshold.

**Acceptance Scenarios**:

1. **Given** a completed experiment with both metrics recorded, **When** the researcher requests metric divergence analysis, **Then** the system generates dual-axis plots showing C-index and IBS curves with regions of divergence highlighted.
2. **Given** multiple experiments, **When** the researcher queries for metric behavior at the interpolation threshold, **Then** the system reports quantified divergence statistics (e.g., C-index continues increasing while IBS spikes).

---

### User Story 4 - Configure Synthetic Data Generation (Priority: P3)

A researcher wants to customize the synthetic data generation parameters to explore specific scenarios beyond the four predefined ones, adjusting covariate distributions, censoring mechanisms, correlation structures, and true model coefficients.

**Why this priority**: Flexibility in data generation enables extended research beyond the core experiments.

**Independent Test**: Can be tested by specifying custom data generation parameters and verifying the generated data matches the requested characteristics.

**Acceptance Scenarios**:

1. **Given** a custom data configuration specifying log-normal covariates with specific mean and variance, **When** the researcher generates data, **Then** the generated covariates match the specified distribution (verifiable via summary statistics and histogram).
2. **Given** a target censoring rate and censoring distribution type, **When** the researcher generates data, **Then** the actual censoring rate in the generated data is within 2% of the target.
3. **Given** a correlation matrix for covariates, **When** the researcher generates data using the Gaussian copula approach, **Then** the empirical correlation matrix of generated data approximates the specified correlation structure.

---

### User Story 5 - Reproduce Experiments with Seeds (Priority: P3)

A researcher wants to reproduce previous experiment results exactly for validation or to continue interrupted experiments.

**Why this priority**: Reproducibility is essential for scientific validity and for efficient workflow management.

**Independent Test**: Can be tested by running the same experiment twice with the same seed and verifying identical results.

**Acceptance Scenarios**:

1. **Given** a random seed value, **When** the researcher runs an experiment twice with the same seed, **Then** both runs produce identical results (data, trained models, metrics).
2. **Given** a previously completed experiment, **When** the researcher requests the experiment configuration, **Then** the system returns all parameters needed to reproduce the experiment including the random seed.

---

### Edge Cases

- What happens when model training fails to converge for very narrow or very wide network configurations? → System retries once with reduced learning rate; if still fails, logs the failure with NaN metrics and continues to the next width.
- How does the system handle when the generated survival times include extreme values (near-zero or very large)?
- What happens when the target censoring rate cannot be achieved within reasonable parameter bounds?
- How does the system handle interrupted experiments (e.g., power failure, manual stop)? → System resumes from checkpoint, detecting completed model widths and continuing from the next pending configuration.
- What happens when GPU memory is insufficient for the widest network configurations? → System logs OOM error, records configuration as failed, and continues to the next width in the sweep.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST generate synthetic survival data using inverse transform sampling with Weibull baseline hazard, with sparse ground-truth coefficients (only first K of N features are predictive, remainder are noise)
- **FR-002**: System MUST support four covariate distribution types: Gaussian, Log-Normal, High-Cardinality Categorical, and Mixed
- **FR-003**: System MUST implement a Gaussian copula for generating correlated covariates
- **FR-004**: System MUST control censoring rates by calibrating the censoring distribution parameter
- **FR-005**: System MUST train DeepSurv models (MLP with Cox partial likelihood loss) across configurable width and depth ranges, supporting both width-sweep and depth-sweep experiment types
- **FR-006**: System MUST support training without regularization (weight decay = 0, no dropout) to observe pure double descent
- **FR-007**: System MUST support training with configurable regularization (weight decay, dropout) for comparison
- **FR-008**: System MUST calculate Concordance Index (C-index) for discrimination evaluation
- **FR-009**: System MUST calculate Integrated Brier Score (IBS) for calibration evaluation
- **FR-010**: System MUST calculate Negative Log Partial Likelihood for loss tracking
- **FR-011**: System MUST generate visualizations of test metrics vs model capacity (double descent curves) in both PNG and PDF formats
- **FR-011a**: System MUST support TensorBoard logging for interactive exploration of training metrics and diagnostics
- **FR-012**: System MUST persist experiment configurations as JSON files and metrics as CSV files for human readability and easy analysis
- **FR-013**: System MUST support reproducibility through random seed control
- **FR-014**: System MUST support both One-Hot Encoding and Entity Embeddings for categorical variables
- **FR-015**: System MUST track training progress and support checkpointing for long-running experiments
- **FR-016**: System MUST execute model training serially (one model configuration at a time) for simplicity and debugging ease
- **FR-017**: System MUST support experiment resumption by detecting completed model configurations and continuing from the next pending width
- **FR-018**: System MUST handle training failures by retrying once with reduced learning rate, then logging failure with NaN metrics and continuing if retry fails
- **FR-019**: System MUST evaluate and log metrics (C-index, IBS, NLL) every epoch to capture full training curves for flexible post-hoc analysis
- **FR-020**: System MUST log training diagnostics including gradient norm, weight norm (L2), learning rate, batch loss variance, and parameter histograms per epoch
- **FR-021**: System MUST handle GPU out-of-memory errors by logging the failure, recording the configuration as failed, and continuing to the next width

### Key Entities

- **Experiment**: Represents a complete experimental run with configuration, data splits, model results, and evaluation metrics
- **DataScenario**: Defines a synthetic data generation configuration (distribution types, sample size, feature count, censoring rate, correlation structure)
- **ModelConfiguration**: Specifies the DeepSurv architecture (number of layers, width, activation) and training parameters (optimizer, epochs, regularization)
- **MetricResult**: Contains evaluation metrics (C-index, IBS, NLL) for a specific model at a specific epoch or final evaluation
- **DoubleDescentCurve**: Aggregated results across model widths showing the relationship between capacity and generalization error

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can execute a complete baseline experiment (data generation through visualization) within a single workflow invocation
- **SC-002**: Experiment results are reproducible - running the same configuration with the same seed produces identical metrics within floating-point tolerance
- **SC-003**: The system successfully trains models across the full capacity sweep from under-parameterized (P << N) to over-parameterized (P >> N) regimes
- **SC-004**: Generated double descent visualizations clearly show the relationship between model capacity and test error with identifiable interpolation threshold regions
- **SC-005**: Metric divergence between C-index and IBS is quantifiable and visualizable for analysis
- **SC-006**: Researchers can compare results across multiple data scenarios in unified visualizations
- **SC-007**: All experiment configurations and results are persisted in a structured format enabling retrospective analysis
- **SC-008**: The system handles datasets with censoring rates from 0% to 95% without failure

## Assumptions

- Researchers have access to GPU resources for training deep learning models (CUDA-compatible)
- Python 3.8+ environment with standard scientific computing libraries (NumPy, SciPy, PyTorch/TensorFlow)
- DeepSurv architecture follows the standard implementation with configurable hidden layer widths
- Training will use Adam optimizer as the default (configurable)
- Standard train/validation/test split ratios (e.g., 60/20/20) will be used unless specified
- Sample size N=1000 is the default for most experiments as specified in the research proposal
- Width range [2, 4, ..., 2048] covers sufficient capacity sweep for observing double descent
- Training for up to 50,000 epochs without early stopping is computationally feasible for the target model sizes
