"""Cross-seed results aggregation for uncertainty quantification."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..visualization.curves import DoubleDescentCurve


def load_experiment_summary(
    experiment_dir: Path,
) -> Optional[Dict[str, List[float]]]:
    """Load summary.csv from an experiment directory.

    Args:
        experiment_dir: Path to experiment output directory.

    Returns:
        Dictionary with columns as keys, values as lists. None if not found.
    """
    summary_path = experiment_dir / "results" / "summary.csv"
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Convert to column-based dict
    result = {}
    for key in rows[0].keys():
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (ValueError, TypeError):
                values.append(np.nan)
        result[key] = values

    return result


def aggregate_multi_seed_results(
    experiment_ids: List[str],
    output_dir: Union[str, Path],
    base_id: str,
) -> Optional[Dict]:
    """Aggregate results from multiple seed runs.

    Args:
        experiment_ids: List of experiment IDs (one per seed).
        output_dir: Base output directory containing experiment folders.
        base_id: Base experiment ID for aggregated output.

    Returns:
        Dictionary with aggregated results, or None if no data.
    """
    output_dir = Path(output_dir)

    # Collect all summaries
    summaries = []
    for exp_id in experiment_ids:
        exp_dir = output_dir / exp_id
        summary = load_experiment_summary(exp_dir)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        return None

    # Get common widths across all seeds
    all_widths = [s["width"] for s in summaries]
    common_widths = sorted(set(all_widths[0]))

    # Aggregate metrics for each width
    metrics = ["c_index", "ibs", "nll"]
    aggregated_data = {
        "width": [],
        "depth": [],
        "n_parameters": [],
    }

    for metric in metrics:
        aggregated_data[f"{metric}_mean"] = []
        aggregated_data[f"{metric}_std"] = []
        aggregated_data[f"{metric}_min"] = []
        aggregated_data[f"{metric}_max"] = []
        aggregated_data[f"{metric}_n"] = []

    for width in common_widths:
        # Find index for this width in each summary
        values_by_metric = {m: [] for m in metrics}
        depth_val = None
        n_params_val = None

        for summary in summaries:
            try:
                idx = summary["width"].index(width)
                for metric in metrics:
                    val = summary[metric][idx]
                    if not np.isnan(val):
                        values_by_metric[metric].append(val)
                if depth_val is None:
                    depth_val = summary["depth"][idx]
                    n_params_val = summary["n_parameters"][idx]
            except (ValueError, IndexError):
                continue

        aggregated_data["width"].append(int(width))
        aggregated_data["depth"].append(int(depth_val) if depth_val else 2)
        aggregated_data["n_parameters"].append(
            int(n_params_val) if n_params_val else 0
        )

        for metric in metrics:
            vals = values_by_metric[metric]
            if vals:
                aggregated_data[f"{metric}_mean"].append(np.mean(vals))
                aggregated_data[f"{metric}_std"].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                aggregated_data[f"{metric}_min"].append(np.min(vals))
                aggregated_data[f"{metric}_max"].append(np.max(vals))
                aggregated_data[f"{metric}_n"].append(len(vals))
            else:
                aggregated_data[f"{metric}_mean"].append(np.nan)
                aggregated_data[f"{metric}_std"].append(np.nan)
                aggregated_data[f"{metric}_min"].append(np.nan)
                aggregated_data[f"{metric}_max"].append(np.nan)
                aggregated_data[f"{metric}_n"].append(0)

    # Create output directory
    agg_dir = output_dir / f"{base_id}_aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "results").mkdir(exist_ok=True)

    # Write aggregated summary CSV
    fieldnames = list(aggregated_data.keys())
    rows = []
    for i in range(len(aggregated_data["width"])):
        rows.append({k: aggregated_data[k][i] for k in fieldnames})

    with open(agg_dir / "results" / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Also write a simple summary (compatible with existing format)
    simple_fieldnames = ["width", "depth", "n_parameters", "c_index", "ibs", "nll"]
    simple_rows = []
    for i in range(len(aggregated_data["width"])):
        simple_rows.append({
            "width": aggregated_data["width"][i],
            "depth": aggregated_data["depth"][i],
            "n_parameters": aggregated_data["n_parameters"][i],
            "c_index": aggregated_data["c_index_mean"][i],
            "ibs": aggregated_data["ibs_mean"][i],
            "nll": aggregated_data["nll_mean"][i],
        })

    with open(agg_dir / "results" / "summary_means.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=simple_fieldnames)
        writer.writeheader()
        writer.writerows(simple_rows)

    # Create DoubleDescentCurve objects with std_errors
    curves = {}
    for metric in metrics:
        curve = DoubleDescentCurve(
            experiment_id=f"{base_id}_aggregated",
            metric_name=metric,
            split="test",
        )
        curve.capacities = aggregated_data["width"]
        curve.values = aggregated_data[f"{metric}_mean"]
        curve.std_errors = aggregated_data[f"{metric}_std"]

        # Estimate n_samples from first seed's data
        first_exp_dir = output_dir / experiment_ids[0]
        ground_truth_path = first_exp_dir / "data" / "ground_truth.json"
        if ground_truth_path.exists():
            with open(ground_truth_path) as f:
                gt = json.load(f)
            n_train = int(gt.get("n_samples", 1000) * 0.6)
        else:
            n_train = 600

        curve.analyze(n_train)
        curves[metric] = curve.to_dict()

    with open(agg_dir / "results" / "curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    # Save metadata
    metadata = {
        "base_experiment_id": base_id,
        "n_seeds": len(summaries),
        "experiment_ids": experiment_ids,
        "widths": aggregated_data["width"],
    }
    with open(agg_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "aggregated_data": aggregated_data,
        "curves": curves,
        "output_dir": str(agg_dir),
        "n_seeds": len(summaries),
    }


def load_aggregated_curves(
    aggregated_dir: Union[str, Path],
) -> Dict[str, DoubleDescentCurve]:
    """Load aggregated curves from an aggregated experiment directory.

    Args:
        aggregated_dir: Path to aggregated experiment directory.

    Returns:
        Dictionary of metric name to DoubleDescentCurve.
    """
    aggregated_dir = Path(aggregated_dir)
    curves_path = aggregated_dir / "results" / "curves.json"

    if not curves_path.exists():
        raise FileNotFoundError(f"Curves file not found: {curves_path}")

    with open(curves_path) as f:
        curves_data = json.load(f)

    return {
        name: DoubleDescentCurve.from_dict(data)
        for name, data in curves_data.items()
    }


def compare_scenarios(
    scenario_dirs: Dict[str, Path],
    metric: str = "c_index",
) -> Tuple[Dict[str, DoubleDescentCurve], Dict]:
    """Load curves from multiple scenarios for comparison.

    Args:
        scenario_dirs: Dictionary mapping scenario name to aggregated dir path.
        metric: Metric to compare.

    Returns:
        Tuple of (curves dict, comparison statistics).
    """
    curves = {}
    stats = {}

    for name, dir_path in scenario_dirs.items():
        try:
            all_curves = load_aggregated_curves(dir_path)
            if metric in all_curves:
                curves[name] = all_curves[metric]

                # Compute comparison stats
                curve = all_curves[metric]
                stats[name] = {
                    "peak_location": curve.peak_location,
                    "peak_value": curve.peak_value,
                    "classical_minimum": curve.classical_minimum,
                    "modern_minimum": curve.modern_minimum,
                    "interpolation_threshold": curve.interpolation_threshold,
                }
        except FileNotFoundError:
            continue

    return curves, stats
