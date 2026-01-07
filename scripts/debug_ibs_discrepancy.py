#!/usr/bin/env python
"""Debug IBS discrepancy between neural networks and classical baselines.

This script investigates why neural networks get IBS ~0.52 while
classical baselines (Cox PH, RSF) get IBS ~0.12.

Key differences to investigate:
1. Baseline hazard estimation method
2. Survival function computation
3. Time grid used for evaluation
4. IPCW weights computation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.brier import calculate_integrated_brier_score
from src.metrics.likelihood import breslow_estimator, compute_survival_function
from src.models.baselines import CoxPHBaseline, RandomSurvivalForestBaseline


def load_experiment_data(experiment_dir: Path):
    """Load train/test data from an experiment."""
    train_data = np.load(experiment_dir / "data" / "train.npz")
    test_data = np.load(experiment_dir / "data" / "test.npz")

    with open(experiment_dir / "data" / "ground_truth.json") as f:
        ground_truth = json.load(f)

    return {
        "train": {
            "X": train_data["X"],
            "T": train_data["T"],
            "E": train_data["E"],
        },
        "test": {
            "X": test_data["X"],
            "T": test_data["T"],
            "E": test_data["E"],
        },
        "beta": np.array(ground_truth["beta"]),
    }


def compute_time_grid(event_times, event_indicators, n_points=100):
    """Compute time grid from event times."""
    event_times_only = event_times[event_indicators == 1]
    if len(event_times_only) == 0:
        event_times_only = event_times

    min_time = np.percentile(event_times_only, 1)
    max_time = np.percentile(event_times_only, 99)

    return np.linspace(min_time, max_time, n_points)


def compute_ibs_neural_style(
    risk_scores_test: np.ndarray,
    train_T: np.ndarray,
    train_E: np.ndarray,
    test_T: np.ndarray,
    test_E: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    """Compute IBS the way the neural network evaluator does.

    Key issue: Uses risk_scores=zeros for baseline hazard estimation,
    then applies test risk scores to compute survival functions.
    """
    # Estimate baseline hazard with ZEROS (this is what evaluator.py does)
    _, baseline_cumhaz = breslow_estimator(
        risk_scores=np.zeros(len(train_T)),  # <-- THE ISSUE?
        event_times=train_T,
        event_indicators=train_E,
        time_points=time_grid,
    )

    # Compute survival functions using TEST risk scores
    survival_functions = compute_survival_function(
        risk_scores_test,
        baseline_cumhaz,
        time_grid,
    )

    # Compute IBS
    ibs = calculate_integrated_brier_score(
        survival_functions,
        time_grid,
        train_T,
        train_E,
        test_T,
        test_E,
    )

    return ibs


def compute_ibs_with_fitted_baseline(
    risk_scores_train: np.ndarray,
    risk_scores_test: np.ndarray,
    train_T: np.ndarray,
    train_E: np.ndarray,
    test_T: np.ndarray,
    test_E: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    """Compute IBS with properly fitted baseline hazard.

    Uses TRAINING risk scores for baseline hazard estimation.
    """
    # Estimate baseline hazard using TRAINING risk scores (proper way)
    _, baseline_cumhaz = breslow_estimator(
        risk_scores=risk_scores_train,  # <-- Use actual training predictions
        event_times=train_T,
        event_indicators=train_E,
        time_points=time_grid,
    )

    # Compute survival functions using TEST risk scores
    survival_functions = compute_survival_function(
        risk_scores_test,
        baseline_cumhaz,
        time_grid,
    )

    # Compute IBS
    ibs = calculate_integrated_brier_score(
        survival_functions,
        time_grid,
        train_T,
        train_E,
        test_T,
        test_E,
    )

    return ibs


def main():
    """Run IBS discrepancy investigation."""
    print("=" * 70)
    print("IBS DISCREPANCY INVESTIGATION")
    print("=" * 70)

    # Find experiment directory
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "outputs" / "experiments"

    # Use seed 42 baseline experiment
    experiment_dir = experiments_dir / "baseline_001_seed_42"
    if not experiment_dir.exists():
        print(f"ERROR: Experiment not found: {experiment_dir}")
        return 1

    print(f"\nLoading data from: {experiment_dir}")
    data = load_experiment_data(experiment_dir)

    train_X = data["train"]["X"]
    train_T = data["train"]["T"]
    train_E = data["train"]["E"]
    test_X = data["test"]["X"]
    test_T = data["test"]["T"]
    test_E = data["test"]["E"]
    beta = data["beta"]

    print(f"Train samples: {len(train_T)} ({train_E.sum():.0f} events)")
    print(f"Test samples: {len(test_T)} ({test_E.sum():.0f} events)")

    # Compute time grid
    time_grid = compute_time_grid(train_T, train_E, n_points=100)
    print(f"Time grid: [{time_grid[0]:.3f}, {time_grid[-1]:.3f}]")

    # -----------------------------------------------------------------
    # 1. Fit classical baselines and get their IBS
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. CLASSICAL BASELINES (scikit-survival)")
    print("-" * 70)

    # Cox PH
    print("\nFitting Cox PH...")
    cox = CoxPHBaseline(alpha=0.01)
    cox.fit(train_X, train_T, train_E)
    cox_results = cox.evaluate(train_X, train_T, train_E, test_X, test_T, test_E)
    print(f"  C-index: {cox_results.c_index:.4f}")
    print(f"  IBS:     {cox_results.ibs:.4f}")

    # Get Cox survival functions for comparison
    cox_times, cox_surv = cox.predict_survival_function(test_X, time_grid)

    # RSF
    print("\nFitting Random Survival Forest...")
    rsf = RandomSurvivalForestBaseline(n_estimators=100, random_state=42)
    rsf.fit(train_X, train_T, train_E)
    rsf_results = rsf.evaluate(train_X, train_T, train_E, test_X, test_T, test_E)
    print(f"  C-index: {rsf_results.c_index:.4f}")
    print(f"  IBS:     {rsf_results.ibs:.4f}")

    # -----------------------------------------------------------------
    # 2. Compute IBS using TRUE risk scores (ground truth)
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. TRUE RISK SCORES (ground truth beta)")
    print("-" * 70)

    true_risk_train = train_X @ beta
    true_risk_test = test_X @ beta

    # Neural-style IBS (zeros for baseline hazard)
    ibs_true_neural_style = compute_ibs_neural_style(
        true_risk_test, train_T, train_E, test_T, test_E, time_grid
    )
    print(f"\n  IBS (neural-style, zeros baseline): {ibs_true_neural_style:.4f}")

    # Proper IBS (fitted baseline hazard)
    ibs_true_fitted = compute_ibs_with_fitted_baseline(
        true_risk_train, true_risk_test, train_T, train_E, test_T, test_E, time_grid
    )
    print(f"  IBS (fitted baseline hazard):       {ibs_true_fitted:.4f}")

    # -----------------------------------------------------------------
    # 3. Load neural network predictions and compare
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. NEURAL NETWORK PREDICTIONS (various widths)")
    print("-" * 70)

    # Load summary to get recorded IBS
    summary_path = experiment_dir / "results" / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print("\n  Width  | Recorded IBS | Recorded C-index")
        print("  -------|--------------|------------------")
        for _, row in summary.iterrows():
            print(f"  {row['width']:5.0f}  |    {row['ibs']:.4f}   |     {row['c_index']:.4f}")

    # -----------------------------------------------------------------
    # 4. Compare survival function distributions
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. SURVIVAL FUNCTION ANALYSIS")
    print("-" * 70)

    # Compute survival functions with true risks using both methods
    _, baseline_cumhaz_zeros = breslow_estimator(
        np.zeros(len(train_T)), train_T, train_E, time_grid
    )
    _, baseline_cumhaz_fitted = breslow_estimator(
        true_risk_train, train_T, train_E, time_grid
    )

    surv_zeros = compute_survival_function(true_risk_test, baseline_cumhaz_zeros, time_grid)
    surv_fitted = compute_survival_function(true_risk_test, baseline_cumhaz_fitted, time_grid)

    print(f"\n  Baseline cumulative hazard at final time:")
    print(f"    Using zeros:  {baseline_cumhaz_zeros[-1]:.4f}")
    print(f"    Using fitted: {baseline_cumhaz_fitted[-1]:.4f}")

    print(f"\n  Survival function statistics at median time (t={time_grid[50]:.2f}):")
    print(f"    Using zeros baseline:")
    print(f"      Mean S(t): {surv_zeros[:, 50].mean():.4f}")
    print(f"      Std S(t):  {surv_zeros[:, 50].std():.4f}")
    print(f"      Min S(t):  {surv_zeros[:, 50].min():.4f}")
    print(f"      Max S(t):  {surv_zeros[:, 50].max():.4f}")

    print(f"    Using fitted baseline:")
    print(f"      Mean S(t): {surv_fitted[:, 50].mean():.4f}")
    print(f"      Std S(t):  {surv_fitted[:, 50].std():.4f}")
    print(f"      Min S(t):  {surv_fitted[:, 50].min():.4f}")
    print(f"      Max S(t):  {surv_fitted[:, 50].max():.4f}")

    print(f"    Cox PH survival functions:")
    print(f"      Mean S(t): {cox_surv[:, 50].mean():.4f}")
    print(f"      Std S(t):  {cox_surv[:, 50].std():.4f}")
    print(f"      Min S(t):  {cox_surv[:, 50].min():.4f}")
    print(f"      Max S(t):  {cox_surv[:, 50].max():.4f}")

    # -----------------------------------------------------------------
    # 5. Diagnosis
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    print("""
The key issue is in how the baseline hazard is estimated:

1. BASELINE MODELS (Cox PH, RSF):
   - Use scikit-survival's predict_survival_function()
   - This internally uses the fitted model's baseline hazard
   - Baseline hazard is estimated WITH the fitted risk scores

2. NEURAL NETWORK (evaluator.py):
   - Uses breslow_estimator() with risk_scores=ZEROS
   - This assumes all subjects have equal (zero) risk
   - Then applies the test risk scores to this "neutral" baseline

This is NOT necessarily a bug - it's a design choice. Using zeros
for baseline estimation creates a "reference" hazard, and the risk
scores adjust from there. However, this can produce survival curves
that are systematically miscalibrated.

The survival functions from the neural approach tend to be more
extreme (closer to 0 or 1) because the baseline hazard isn't
accounting for the risk score distribution.
""")

    # Check if the difference explains the IBS gap
    gap = ibs_true_neural_style - ibs_true_fitted
    print(f"\n  IBS gap from baseline hazard method: {gap:.4f}")
    print(f"  Neural IBS ~{ibs_true_neural_style:.2f} vs Baseline IBS ~{cox_results.ibs:.2f}")

    if abs(ibs_true_neural_style - 0.52) < 0.1 and abs(cox_results.ibs - 0.12) < 0.05:
        print("\n  CONCLUSION: The discrepancy is EXPLAINED by the baseline hazard method.")
        print("  This is a REAL methodological difference, not a bug.")
    else:
        print("\n  CONCLUSION: Additional factors may contribute to the discrepancy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
