"""Run real-data experiments (METABRIC and SUPPORT) with width sweeps.

Usage:
    python scripts/run_real_data.py
    python scripts/run_real_data.py --dataset metabric
    python scripts/run_real_data.py --dataset support
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.real_data import load_real_dataset
from src.data.generator import SurvivalData, DataSplitter
from src.models.deepsurv import DeepSurv
from src.models.trainer import train_with_retry
from src.models.config import ModelConfiguration
from src.metrics.evaluator import MetricEvaluator

SEEDS = [42, 123, 456, 789, 1011, 1414, 1618, 1732, 2024, 2025,
         2026, 2718, 3141, 3333, 4444, 5555, 6666, 7777, 8888, 9999]

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

TRAINING = {
    "epochs": 2000,
    "batch_size": 256,
    "learning_rate": 0.001,
    "optimizer": "adam",
}


def run_real_data_experiment(dataset_name: str, output_base: Path) -> None:
    """Run width sweep on a real dataset across all seeds."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} dataset...")
    data = load_real_dataset(dataset_name)
    n_features = data.X.shape[1]
    n_samples = data.X.shape[0]
    n_events = int(data.E.sum())
    print(f"  n_samples={n_samples}, n_features={n_features}, "
          f"n_events={n_events} ({100*n_events/n_samples:.0f}%)")

    for seed in SEEDS:
        seed_id = f"{dataset_name}_seed_{seed}"
        seed_dir = output_base / seed_id
        results_dir = seed_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Check if already done
        summary_path = results_dir / "summary.csv"
        if summary_path.exists():
            print(f"SKIP: {seed_id} (already completed)")
            continue

        print(f"\nRUN: {seed_id}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Split data
        splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        splits = splitter.split(data)
        train_data, val_data, test_data = splits

        rows = []
        for width in WIDTHS:
            config = ModelConfiguration(
                width=width,
                depth=2,
                activation="relu",
                dropout=0.0,
                weight_decay=0.0,
                epochs=TRAINING["epochs"],
                batch_size=TRAINING["batch_size"],
                learning_rate=TRAINING["learning_rate"],
                optimizer=TRAINING["optimizer"],
            )

            n_params = config.n_parameters(n_features)
            print(f"  width={width:>4d}  params={n_params:>10d}  ", end="", flush=True)

            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                run_id = f"{dataset_name}_s{seed}_w{width}"
                state, metrics, model = train_with_retry(
                    n_features=n_features,
                    config=config,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    device=device,
                    run_id=run_id,
                )

                # Get final test metrics from last epoch
                test_metrics = [m for m in metrics if m.split == "test"]
                if test_metrics:
                    last = test_metrics[-1]
                    c_idx = last.c_index
                    ibs = last.integrated_brier_score
                    nll = last.neg_log_likelihood
                else:
                    c_idx = ibs = nll = float("nan")
                print(f"C={c_idx:.4f}  IBS={ibs:.4f}")

                rows.append({
                    "width": width,
                    "depth": 2,
                    "n_parameters": n_params,
                    "c_index": c_idx,
                    "ibs": ibs,
                    "nll": nll,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({
                    "width": width,
                    "depth": 2,
                    "n_parameters": n_params,
                    "c_index": float("nan"),
                    "ibs": float("nan"),
                    "nll": float("nan"),
                })

        # Save results
        import csv
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Save config
        config_out = {
            "dataset": dataset_name,
            "seed": seed,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_events": n_events,
            "widths": WIDTHS,
            "training": TRAINING,
        }
        with open(seed_dir / "config.json", "w") as f:
            json.dump(config_out, f, indent=2)

        print(f"  Saved: {summary_path}")


def run_baselines(dataset_name: str, output_base: Path) -> None:
    """Run Cox PH and RSF baselines on real data."""
    from lifelines import CoxPHFitter
    from sksurv.ensemble import RandomSurvivalForest
    import pandas as pd

    print(f"\nRunning baselines for {dataset_name}...")
    data = load_real_dataset(dataset_name)

    baseline_dir = output_base / f"{dataset_name}_baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in SEEDS:
        np.random.seed(seed)
        splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        splits = splitter.split(data)
        train_data, val_data, test_data = splits

        # Cox PH
        try:
            feature_cols = [f"x{i}" for i in range(train_data.X.shape[1])]
            df_train = pd.DataFrame(train_data.X, columns=feature_cols)
            df_train["T"] = train_data.T
            df_train["E"] = train_data.E

            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df_train, duration_col="T", event_col="E")
            cph_c = cph.concordance_index_
        except Exception as e:
            cph_c = float("nan")
            print(f"  Cox PH failed (seed={seed}): {e}")

        # RSF
        try:
            from sksurv.util import Surv
            y_train = Surv.from_arrays(train_data.E.astype(bool), train_data.T)
            y_test = Surv.from_arrays(test_data.E.astype(bool), test_data.T)
            rsf = RandomSurvivalForest(n_estimators=100, random_state=seed)
            rsf.fit(train_data.X, y_train)
            rsf_c = rsf.score(test_data.X, y_test)
        except Exception as e:
            rsf_c = float("nan")
            print(f"  RSF failed (seed={seed}): {e}")

        results.append({"seed": seed, "cox_ph_c": cph_c, "rsf_c": rsf_c})
        print(f"  seed={seed}: Cox={cph_c:.4f}, RSF={rsf_c:.4f}")

    import csv
    with open(baseline_dir / "baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {baseline_dir / 'baselines.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Run real-data experiments")
    parser.add_argument("--dataset", choices=["metabric", "support", "all"],
                        default="all", help="Dataset to run")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run Cox PH and RSF baselines")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/experiments"),
                        help="Output directory")
    args = parser.parse_args()

    datasets = ["metabric", "support"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        if args.baselines_only:
            run_baselines(ds, args.output_dir)
        else:
            run_real_data_experiment(ds, args.output_dir)
            run_baselines(ds, args.output_dir)


if __name__ == "__main__":
    main()
