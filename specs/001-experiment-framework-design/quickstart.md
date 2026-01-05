# Quickstart Guide: Double Descent Experiment Framework

**Branch**: `001-experiment-framework-design` | **Date**: 2026-01-04

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- ~10GB disk space for experiment outputs

## Installation

```bash
# Clone repository
git clone <repository-url>
cd DoubleDescent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Run Your First Experiment

### 1. Quick Test (5 minutes)

Run a minimal experiment to verify everything works:

```bash
# Generate small test config
cat > configs/experiments/test_quick.json << 'EOF'
{
  "experiment_id": "test_quick",
  "name": "Quick Test",
  "seed": 42,
  "data": {
    "scenario": "baseline",
    "n_samples": 200,
    "n_features": 10,
    "n_predictive": 5,
    "censoring_rate": 0.3
  },
  "model": {
    "widths": [4, 16, 64],
    "depths": [2],
    "activation": "relu"
  },
  "training": {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
EOF

# Run experiment
python -m src.cli.run_experiment --config configs/experiments/test_quick.json

# Check results
python -m src.cli.status --experiment outputs/experiments/test_quick/
```

### 2. Baseline Experiment (Full)

Run the standard baseline experiment from the research proposal:

```bash
python -m src.cli.run_experiment --config configs/scenarios/baseline.json
```

This will:
- Generate 1000 samples with Gaussian covariates
- Train DeepSurv models with widths [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
- Save metrics every epoch
- Take approximately 4-8 hours on a modern GPU

### 3. Monitor Progress

```bash
# Check status
python -m src.cli.status --experiment outputs/experiments/baseline_001/

# View TensorBoard
tensorboard --logdir outputs/experiments/baseline_001/tensorboard/
```

### 4. Generate Visualizations

After the experiment completes:

```bash
python -m src.cli.visualize \
    --experiment outputs/experiments/baseline_001/ \
    --output figures/baseline/ \
    --format png,pdf
```

This creates:
- `double_descent.pdf` - Test error vs model capacity
- `metric_divergence.pdf` - C-index vs IBS comparison
- `training_curves.pdf` - Loss over epochs for each width

## Run All Four Scenarios

Execute the complete experimental suite:

```bash
# Scenario A: Baseline (Gaussian)
python -m src.cli.run_experiment --config configs/scenarios/baseline.json

# Scenario B: Skewed (Log-Normal)
python -m src.cli.run_experiment --config configs/scenarios/skewed.json

# Scenario C: High-Cardinality (Categorical)
python -m src.cli.run_experiment --config configs/scenarios/high_cardinality.json

# Scenario D: Imbalanced (90% censoring)
python -m src.cli.run_experiment --config configs/scenarios/imbalanced.json
```

Then compare results:

```bash
python -m src.cli.compare \
    --experiments outputs/experiments/baseline*/ outputs/experiments/skewed*/ \
                  outputs/experiments/high_cardinality*/ outputs/experiments/imbalanced*/ \
    --output figures/comparison/ \
    --labels "Baseline" "Skewed" "High-Card" "Imbalanced"
```

## Resume Interrupted Experiments

If an experiment is interrupted (Ctrl+C, power failure, etc.):

```bash
# Resume from last checkpoint
python -m src.cli.run_experiment \
    --config outputs/experiments/baseline_001/config.json \
    --resume
```

The framework automatically:
- Detects completed model widths
- Skips already-trained configurations
- Continues from the next pending width

## Customize Experiments

### Custom Data Scenario

```bash
# Create custom scenario config
cat > configs/scenarios/my_scenario.json << 'EOF'
{
  "name": "my_custom_scenario",
  "n_samples": 2000,
  "n_features": 30,
  "n_predictive": 15,
  "covariate_type": "lognormal",
  "censoring_rate": 0.5,
  "weibull_scale": 0.3,
  "weibull_shape": 1.5
}
EOF

# Generate data only
python -m src.cli.generate_data \
    --config configs/scenarios/my_scenario.json \
    --output data/my_scenario.npz \
    --seed 123
```

### Custom Model Sweep

```bash
# Create experiment with custom widths
cat > configs/experiments/depth_sweep.json << 'EOF'
{
  "experiment_id": "depth_sweep_001",
  "name": "Depth Sweep Experiment",
  "seed": 42,
  "data": {
    "scenario": "baseline",
    "n_samples": 1000,
    "n_features": 20,
    "n_predictive": 10,
    "censoring_rate": 0.3
  },
  "model": {
    "widths": [64],
    "depths": [1, 2, 3, 4, 5, 6],
    "sweep_type": "depth"
  },
  "training": {
    "epochs": 50000,
    "batch_size": 256,
    "learning_rate": 0.001
  }
}
EOF
```

## Directory Structure

After running experiments:

```
DoubleDescent/
├── configs/
│   ├── scenarios/           # Data generation configs
│   └── experiments/         # Full experiment configs
├── outputs/
│   └── experiments/
│       └── baseline_001/
│           ├── config.json        # Experiment configuration
│           ├── data/              # Generated data (train/val/test splits)
│           ├── runs/              # Per-width results and checkpoints
│           ├── results/           # Aggregated results
│           ├── tensorboard/       # TensorBoard logs
│           └── progress.json      # Resumption state
└── figures/
    └── baseline/
        ├── double_descent.pdf
        ├── metric_divergence.pdf
        └── training_curves.pdf
```

## Common Issues

### CUDA Out of Memory

For wide networks (1024+), you may hit GPU memory limits:

```bash
# Option 1: Reduce batch size
# Edit config: "batch_size": 128

# Option 2: Force CPU (slower but works)
python -m src.cli.run_experiment --config config.json --device cpu
```

The framework automatically skips OOM configurations and continues.

### Slow Training

- Reduce epochs for initial testing: `"epochs": 1000`
- Use fewer widths: `"widths": [8, 64, 512]`
- Reduce sample size: `"n_samples": 500`

### Reproducibility Issues

Always set the seed in your config:

```json
{
  "seed": 42,
  ...
}
```

Same seed = identical results (within floating-point tolerance).

## Next Steps

1. **Analyze Results**: Open TensorBoard to explore training dynamics
2. **Compare Scenarios**: Use the compare command to test hypotheses
3. **Metric Divergence**: Check if C-index hides overfitting that IBS reveals
4. **Publication Figures**: Generate PDF plots with `--dpi 600`

## Getting Help

```bash
# Command help
python -m src.cli.run_experiment --help
python -m src.cli.generate_data --help
python -m src.cli.visualize --help
python -m src.cli.compare --help
```
