#!/usr/bin/env python3
"""Embedding experiment runner for categorical features.

This script runs experiments using DeepSurvEmbedding instead of one-hot encoding
for categorical features, avoiding the p >> n problem.

Usage:
    python scripts/run_embedding_experiment.py --scenario high_cardinality --seeds 5
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.scenarios import DataScenario, get_scenario
from src.data.generator import SurvivalDataGenerator, SurvivalDataEmbedding
from src.data.types import CovariateType
from src.models.deepsurv import DeepSurvEmbedding, cox_ph_loss
from src.models.config import ModelConfiguration
from src.metrics.concordance import calculate_c_index as concordance_index
from src.metrics.brier import integrated_brier_score_simple


DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run embedding experiments for categorical features.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="high_cardinality",
        help="Scenario name (default: high_cardinality)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to run (default: 5)",
    )

    parser.add_argument(
        "--widths",
        type=str,
        default=None,
        help="Comma-separated list of widths (default: standard sweep)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of training epochs (default: 10000)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Output directory (default: outputs/experiments/)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device (default: auto)",
    )

    return parser.parse_args()


def split_embedding_data(
    data: SurvivalDataEmbedding,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[SurvivalDataEmbedding, SurvivalDataEmbedding, SurvivalDataEmbedding]:
    """Split embedding data into train/val/test sets."""
    rng = np.random.default_rng(seed)
    n = len(data.T)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset(idx):
        return SurvivalDataEmbedding(
            X_continuous=data.X_continuous[idx],
            X_categorical=data.X_categorical[idx],
            categorical_cardinalities=data.categorical_cardinalities,
            T=data.T[idx],
            E=data.E[idx],
            T_true=data.T_true[idx],
            beta=data.beta,
        )

    return subset(train_idx), subset(val_idx), subset(test_idx)


def train_embedding_model(
    model: DeepSurvEmbedding,
    train_data: SurvivalDataEmbedding,
    val_data: SurvivalDataEmbedding,
    device: torch.device,
    epochs: int = 10000,
    lr: float = 0.001,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Train an embedding model.

    Returns:
        Tuple of (best_model, metrics_dict)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert data to tensors
    X_cont_train = torch.from_numpy(train_data.X_continuous).float().to(device)
    X_cat_train = torch.from_numpy(train_data.X_categorical).long().to(device)
    T_train = torch.from_numpy(train_data.T).float().to(device)
    E_train = torch.from_numpy(train_data.E).float().to(device)

    X_cont_val = torch.from_numpy(val_data.X_continuous).float().to(device)
    X_cat_val = torch.from_numpy(val_data.X_categorical).long().to(device)

    best_val_metric = float('inf')
    best_state = None
    metrics = {"epochs": [], "train_loss": [], "val_c_index": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        risk_scores = model(X_cont_train, X_cat_train)
        loss = cox_ph_loss(risk_scores, T_train, E_train)

        loss.backward()
        optimizer.step()

        # Evaluate periodically
        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_risk = model(X_cont_val, X_cat_val).cpu().numpy().ravel()

            val_c_index = concordance_index(
                val_data.T, val_data.E, val_risk
            )

            # Use negative C-index as "loss" to track (want to maximize C-index)
            if 1 - val_c_index < best_val_metric:
                best_val_metric = 1 - val_c_index
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            metrics["epochs"].append(epoch)
            metrics["train_loss"].append(loss.item())
            metrics["val_c_index"].append(val_c_index)

            if verbose and epoch % 1000 == 0:
                print(f"  Epoch {epoch}: loss={loss.item():.4f}, val_c={val_c_index:.4f}",
                      file=sys.stderr)

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, metrics


def evaluate_embedding_model(
    model: DeepSurvEmbedding,
    test_data: SurvivalDataEmbedding,
    train_data: SurvivalDataEmbedding,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test data."""
    model.eval()

    X_cont_test = torch.from_numpy(test_data.X_continuous).float().to(device)
    X_cat_test = torch.from_numpy(test_data.X_categorical).long().to(device)

    with torch.no_grad():
        test_risk = model(X_cont_test, X_cat_test).cpu().numpy().ravel()

    c_index = concordance_index(test_data.T, test_data.E, test_risk)

    # IBS requires survival function estimation
    # For now, use a simplified estimate
    ibs = integrated_brier_score_simple(
        test_data.T, test_data.E, test_risk, train_data.T, train_data.E
    )

    return {
        "c_index": c_index,
        "ibs": ibs,
    }


def run_single_experiment(
    scenario: DataScenario,
    seed: int,
    widths: List[int],
    epochs: int,
    output_dir: Path,
    device: torch.device,
    verbose: bool = True,
) -> Dict:
    """Run embedding experiment for a single seed."""
    experiment_id = f"{scenario.name}_embedding_seed_{seed}"
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "runs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Experiment: {experiment_id}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    # Generate data
    if verbose:
        print("Generating data with embeddings...", file=sys.stderr)

    generator = SurvivalDataGenerator(scenario, seed=seed)
    full_data = generator.generate_for_embedding()
    train_data, val_data, test_data = split_embedding_data(full_data, seed)

    if verbose:
        print(f"  Categorical features: {len(full_data.categorical_cardinalities)}", file=sys.stderr)
        print(f"  Cardinalities: {full_data.categorical_cardinalities}", file=sys.stderr)
        print(f"  Train: {len(train_data.T)}, Val: {len(val_data.T)}, Test: {len(test_data.T)}",
              file=sys.stderr)

    # Run width sweep
    results = []
    for width in widths:
        if verbose:
            print(f"\n[Width={width}] Training...", file=sys.stderr)

        # Create model
        config = ModelConfiguration(
            width=width,
            depth=2,
            activation="relu",
            dropout=0.0,
            learning_rate=0.001,
        )

        model = DeepSurvEmbedding(
            n_continuous=train_data.X_continuous.shape[1],
            categorical_cardinalities=train_data.categorical_cardinalities,
            config=config,
        )

        n_params = model.n_parameters()

        # Train
        model, train_metrics = train_embedding_model(
            model, train_data, val_data, device, epochs=epochs, verbose=verbose
        )

        # Evaluate
        test_metrics = evaluate_embedding_model(model, test_data, train_data, device)

        if verbose:
            print(f"  n_params={n_params}, c_index={test_metrics['c_index']:.4f}, "
                  f"ibs={test_metrics['ibs']:.4f}", file=sys.stderr)

        results.append({
            "width": width,
            "depth": 2,
            "n_parameters": n_params,
            "c_index": test_metrics["c_index"],
            "ibs": test_metrics["ibs"],
        })

        # Save run metrics
        run_dir = exp_dir / "runs" / f"width_{width:04d}_depth_02"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({
                "final": test_metrics,
                "training": train_metrics,
            }, f, indent=2)

    # Save summary
    with open(exp_dir / "results" / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["width", "depth", "n_parameters", "c_index", "ibs"])
        writer.writeheader()
        writer.writerows(results)

    # Save progress
    progress = {
        "completed": True,
        "n_widths": len(widths),
        "last_updated": datetime.now().isoformat(),
    }
    with open(exp_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)

    return {"experiment_id": experiment_id, "results": results}


def aggregate_results(
    experiment_ids: List[str],
    output_dir: Path,
    base_id: str,
) -> None:
    """Aggregate results across seeds."""
    all_results = []

    for exp_id in experiment_ids:
        summary_path = output_dir / exp_id / "results" / "summary.csv"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["experiment_id"] = exp_id
                    all_results.append(row)

    if not all_results:
        return

    # Group by width
    from collections import defaultdict
    by_width = defaultdict(list)
    for r in all_results:
        by_width[int(r["width"])].append(r)

    # Compute mean and std
    aggregated = []
    for width in sorted(by_width.keys()):
        rows = by_width[width]
        c_indices = [float(r["c_index"]) for r in rows]
        ibs_values = [float(r["ibs"]) for r in rows]

        aggregated.append({
            "width": width,
            "depth": 2,
            "n_parameters": rows[0]["n_parameters"],
            "c_index_mean": np.mean(c_indices),
            "c_index_std": np.std(c_indices),
            "ibs_mean": np.mean(ibs_values),
            "ibs_std": np.std(ibs_values),
            "n_seeds": len(rows),
        })

    # Save aggregated results
    agg_dir = output_dir / f"{base_id}_aggregated" / "results"
    agg_dir.mkdir(parents=True, exist_ok=True)

    with open(agg_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "width", "depth", "n_parameters",
            "c_index_mean", "c_index_std",
            "ibs_mean", "ibs_std", "n_seeds"
        ])
        writer.writeheader()
        writer.writerows(aggregated)

    print(f"\nAggregated results saved to: {agg_dir}/summary.csv", file=sys.stderr)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Get scenario
    try:
        scenario = get_scenario(args.scenario)
    except ValueError:
        # Create a custom embedding scenario if not predefined
        scenario = DataScenario(
            name=args.scenario,
            description=f"Custom embedding scenario: {args.scenario}",
            covariate_type=CovariateType.CATEGORICAL,
            n_categorical_features=5,
            cardinality=100,
            n_features=20,
            n_predictive=5,
            censoring_rate=0.3,
        )

    # Ensure it's categorical
    if scenario.covariate_type not in (CovariateType.CATEGORICAL, CovariateType.MIXED):
        print(f"ERROR: Scenario must be categorical or mixed, got {scenario.covariate_type}",
              file=sys.stderr)
        return 1

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Get widths
    if args.widths:
        widths = [int(w.strip()) for w in args.widths.split(",")]
    else:
        widths = DEFAULT_WIDTHS

    # Get seeds
    seeds = DEFAULT_SEEDS[:args.seeds]

    print(f"Running embedding experiments for scenario: {scenario.name}", file=sys.stderr)
    print(f"Seeds: {seeds}", file=sys.stderr)
    print(f"Widths: {widths}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)

    # Run experiments
    experiment_ids = []
    for seed in seeds:
        result = run_single_experiment(
            scenario=scenario,
            seed=seed,
            widths=widths,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=device,
        )
        experiment_ids.append(result["experiment_id"])

    # Aggregate results
    base_id = f"{scenario.name}_embedding"
    aggregate_results(experiment_ids, args.output_dir, base_id)

    print("\nAll experiments completed!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
