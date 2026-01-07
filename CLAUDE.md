# DoubleDescent Development Guidelines

Auto-generated from feature plans. Last updated: 2026-01-04

## Project Overview

Research framework for investigating the Double Descent phenomenon in survival analysis (time-to-event models). Uses synthetic data generation with DeepSurv neural networks to map generalization behavior across under-parameterized to over-parameterized regimes.

## Active Technologies

- **Language**: Python 3.8+ (compatible with PyTorch and scikit-survival)
- **Deep Learning**: PyTorch >= 1.10.0
- **Survival Analysis**: scikit-survival, pycox, lifelines
- **Visualization**: Matplotlib, Seaborn, TensorBoard
- **Testing**: pytest with pytest-cov
- **Storage**: File-based (JSON configs, CSV metrics, PyTorch checkpoints)

## Project Structure

```text
src/
├── data/           # Synthetic data generation (Weibull hazard, Gaussian copula)
├── models/         # DeepSurv implementation (MLP + Cox partial likelihood)
├── metrics/        # C-index, Integrated Brier Score, NLL
├── experiments/    # Experiment orchestration, checkpointing, resumption
├── visualization/  # Double descent curves, metric divergence plots
└── cli/            # Command-line interface entry points

tests/
├── unit/           # Component tests
├── integration/    # Pipeline tests
└── contract/       # CLI interface tests

configs/
├── scenarios/      # Data generation configs (baseline, skewed, etc.)
└── experiments/    # Full experiment configs

outputs/            # Generated outputs (gitignored)
├── experiments/    # Experiment results
└── figures/        # Visualization outputs
```

## Key Commands

```bash
# Run experiment
python -m src.cli.run_experiment --config configs/experiments/baseline.json

# Generate data only
python -m src.cli.generate_data --scenario baseline --output data/

# Visualize results
python -m src.cli.visualize --experiment outputs/experiments/exp_001/

# Compare experiments
python -m src.cli.compare --experiments exp1/ exp2/ --output figures/

# Run tests
pytest tests/ -v --cov=src

# Code quality
ruff check src/ tests/
black src/ tests/ --check
mypy src/
```

## Code Style

- Follow PEP 8 with Black formatting
- Type hints required for all public functions
- Docstrings in Google style
- Maximum line length: 100 characters

## Key Design Decisions

1. **Serial Execution**: One model configuration at a time (no parallelism)
2. **Checkpoint Resumption**: Experiments can resume from last completed width
3. **Epoch-Level Logging**: Metrics logged every epoch for full training curves
4. **Dual Output**: TensorBoard for interactive exploration, CSV for publication plots
5. **Sparse Coefficients**: Only first K of N features are predictive in synthetic data

## Recent Changes

- 001-experiment-framework-design: Initial framework design

<!-- MANUAL ADDITIONS START -->

## Manuscript Writing Guidelines

The manuscript in `manuscript/` follows JASA (Journal of the American Statistical Association) conventions. When editing or extending the manuscript:

### Style Requirements

1. **Scientific register**: Use formal academic prose. Avoid colloquialisms, contractions, and informal phrasing.

2. **No AI-isms**: Eliminate phrases common in AI-generated text:
   - No "In conclusion," "Furthermore," "Moreover," "Additionally"
   - No "comprehensive," "crucial," "essential" as filler adjectives
   - No "it is important to note that" or similar hedging
   - No sandwich structure (intro-body-summary within paragraphs)

3. **No em-dashes in prose**: Avoid `--` for parenthetical remarks. Use commas, parentheses, or restructure into separate sentences.

4. **No bullet points or numbered lists**: Convert all lists to flowing prose paragraphs.

5. **Sentence variety**: Mix medium-length declarative sentences with longer explanatory ones. Avoid uniform sentence length.

6. **Citation style**: Author-year format using natbib (`\citep{}`, `\citet{}`, `\citealt{}`).

7. **Reference order**: Use `unsrtnat` bibliography style (citation order, not alphabetical).

### Building the Manuscript

```bash
# Full build with cleanup
./manuscript/build.sh

# Or manually:
cd manuscript
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### File Structure

- `manuscript/main.tex` - Main document
- `manuscript/references.bib` - Bibliography
- `manuscript/build.sh` - Build script (compiles and cleans intermediate files)

<!-- MANUAL ADDITIONS END -->
