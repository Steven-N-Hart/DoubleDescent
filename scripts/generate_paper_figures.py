#!/usr/bin/env python
"""Generate publication-quality figures for the double descent paper."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_experiment_data(experiment_dir: Path) -> dict:
    """Load all data from an experiment directory."""
    data = {}

    # Load summary CSV
    summary_path = experiment_dir / "results" / "summary.csv"
    if summary_path.exists():
        data['summary'] = pd.read_csv(summary_path)

    # Load curves JSON
    curves_path = experiment_dir / "results" / "curves.json"
    if curves_path.exists():
        with open(curves_path) as f:
            data['curves'] = json.load(f)

    # Load train curves
    train_curves_path = experiment_dir / "results" / "train_curves.json"
    if train_curves_path.exists():
        with open(train_curves_path) as f:
            data['train_curves'] = json.load(f)

    return data


def figure1_main_double_descent(
    data: dict,
    output_path: Path,
    metric: str = "ibs",
):
    """Figure 1: Main double descent curve with train and test.

    Shows test error (IBS) vs model capacity with training error overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    df = data['summary']
    widths = df['width'].values

    # Get train and test values
    if 'train_curves' in data and metric in data['train_curves']:
        train_vals = np.array(data['train_curves'][metric]['values'])
    else:
        train_vals = None

    test_vals = df[metric].values if metric in df.columns else df['c_index'].values

    # Plot test curve (primary)
    ax.plot(widths, test_vals, 'o-', linewidth=2.5, markersize=8,
            color='#e74c3c', label='Test', zorder=3)

    # Plot train curve
    if train_vals is not None:
        ax.plot(widths, train_vals, 's--', linewidth=2, markersize=6,
                color='#2ecc71', label='Train', alpha=0.8, zorder=2)

    # Mark interpolation threshold (approximate P ≈ N)
    n_train = 600  # 60% of 1000
    threshold_idx = np.argmin(np.abs(df['n_parameters'].values - n_train))
    threshold_width = widths[threshold_idx]

    ax.axvline(threshold_width, color='gray', linestyle=':', linewidth=2,
               label=f'P ≈ N ({threshold_width})', alpha=0.7)

    # Mark the peak (worst test performance)
    if metric == 'c_index':
        peak_idx = np.argmin(test_vals)
    else:
        peak_idx = np.argmax(test_vals)

    ax.scatter([widths[peak_idx]], [test_vals[peak_idx]],
               s=200, color='#e74c3c', marker='*',
               edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Model Width (log scale)')

    metric_labels = {
        'c_index': 'Concordance Index',
        'ibs': 'Integrated Brier Score',
        'nll': 'Negative Log-Likelihood',
    }
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title('Double Descent in Survival Analysis')

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Underparameterized', xy=(4, ax.get_ylim()[1]*0.95),
                fontsize=9, ha='center', alpha=0.7)
    ax.annotate('Overparameterized', xy=(512, ax.get_ylim()[1]*0.95),
                fontsize=9, ha='center', alpha=0.7)

    plt.tight_layout()
    fig.savefig(f"{output_path}.png")
    fig.savefig(f"{output_path}.pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}.png/pdf")


def figure2_cindex_vs_ibs(
    data: dict,
    output_path: Path,
):
    """Figure 2: C-index vs IBS divergence.

    Shows both metrics on same plot to illustrate discrimination/calibration divergence.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    df = data['summary']
    widths = df['width'].values
    c_index = df['c_index'].values
    ibs = df['ibs'].values

    # Primary axis: C-index (higher is better)
    color1 = '#3498db'
    ax1.set_xlabel('Model Width (log scale)')
    ax1.set_ylabel('Concordance Index (↑ better)', color=color1)
    line1 = ax1.plot(widths, c_index, 'o-', linewidth=2.5, markersize=8,
                     color=color1, label='C-index')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)

    # Secondary axis: IBS (lower is better)
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Integrated Brier Score (↓ better)', color=color2)
    line2 = ax2.plot(widths, ibs, 's-', linewidth=2.5, markersize=8,
                     color=color2, label='IBS')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Mark where they diverge most
    # Normalize both metrics to [0,1] for comparison
    c_norm = (c_index - c_index.min()) / (c_index.max() - c_index.min() + 1e-8)
    ibs_norm = (ibs - ibs.min()) / (ibs.max() - ibs.min() + 1e-8)
    divergence = np.abs(c_norm - (1 - ibs_norm))  # Should align if no divergence

    max_div_idx = np.argmax(divergence)
    ax1.axvline(widths[max_div_idx], color='purple', linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'Max divergence (width={widths[max_div_idx]})')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

    ax1.set_title('Discrimination vs Calibration Across Model Capacity')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_path}.png")
    fig.savefig(f"{output_path}.pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}.png/pdf")


def figure3_censoring_comparison(
    baseline_data: dict,
    high_censoring_data: dict,
    output_path: Path,
):
    """Figure 3: Effect of censoring rate on threshold location.

    Overlays curves from 30% and 90% censoring scenarios.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    df_baseline = baseline_data['summary']
    df_high = high_censoring_data['summary']

    widths_b = df_baseline['width'].values
    widths_h = df_high['width'].values

    # Use C-index for comparison (more stable than IBS for high censoring)
    cindex_b = df_baseline['c_index'].values
    cindex_h = df_high['c_index'].values

    # Plot both scenarios
    ax.plot(widths_b, cindex_b, 'o-', linewidth=2.5, markersize=8,
            color='#3498db', label='30% Censoring (Baseline)')
    ax.plot(widths_h, cindex_h, 's-', linewidth=2.5, markersize=8,
            color='#e74c3c', label='90% Censoring (High)')

    # Mark peaks (minimum C-index) for each
    peak_b = np.argmin(cindex_b)
    peak_h = np.argmin(cindex_h)

    ax.scatter([widths_b[peak_b]], [cindex_b[peak_b]], s=200,
               color='#3498db', marker='*', edgecolors='black',
               linewidths=1.5, zorder=5)
    ax.scatter([widths_h[peak_h]], [cindex_h[peak_h]], s=200,
               color='#e74c3c', marker='*', edgecolors='black',
               linewidths=1.5, zorder=5)

    # Mark effective sample sizes
    n_train = 600
    n_events_baseline = int(n_train * 0.7)  # 70% events for 30% censoring
    n_events_high = int(n_train * 0.1)  # 10% events for 90% censoring

    # Annotate threshold locations - position text to avoid overlap
    # Baseline annotation: upper right area
    ax.annotate(f'Peak @ w={widths_b[peak_b]}\n(events≈{n_events_baseline})',
                xy=(widths_b[peak_b], cindex_b[peak_b]),
                xytext=(128, 0.76),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3498db', alpha=0.9))

    # High censoring annotation: position inside plot area
    ax.annotate(f'Peak @ w={widths_h[peak_h]}\n(events≈{n_events_high})',
                xy=(widths_h[peak_h], cindex_h[peak_h]),
                xytext=(64, 0.72),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9))

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Model Width (log scale)')
    ax.set_ylabel('Test Concordance Index')
    ax.set_title('Effect of Censoring Rate on Double Descent')

    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_path}.png")
    fig.savefig(f"{output_path}.pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}.png/pdf")


def figure4_regularization_comparison(
    baseline_data: dict,
    regularized_data: dict,
    output_path: Path,
):
    """Figure 4: Effect of L2 regularization on double descent.

    Compares baseline (no regularization) vs regularized (weight decay) curves.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    df_baseline = baseline_data['summary']
    df_reg = regularized_data['summary']

    widths_b = df_baseline['width'].values
    widths_r = df_reg['width'].values

    # Use C-index for comparison
    cindex_b = df_baseline['c_index'].values
    cindex_r = df_reg['c_index'].values

    # Plot both scenarios
    ax.plot(widths_b, cindex_b, 'o-', linewidth=2.5, markersize=8,
            color='#e74c3c', label='No Regularization')
    ax.plot(widths_r, cindex_r, 's-', linewidth=2.5, markersize=8,
            color='#2ecc71', label='Weight Decay (λ=0.01)')

    # Mark peaks (minimum C-index) for each
    peak_b = np.argmin(cindex_b)
    peak_r = np.argmin(cindex_r)

    ax.scatter([widths_b[peak_b]], [cindex_b[peak_b]], s=200,
               color='#e74c3c', marker='*', edgecolors='black',
               linewidths=1.5, zorder=5)
    ax.scatter([widths_r[peak_r]], [cindex_r[peak_r]], s=200,
               color='#2ecc71', marker='*', edgecolors='black',
               linewidths=1.5, zorder=5)

    # Calculate improvement at peak
    peak_improvement = cindex_r[peak_b] - cindex_b[peak_b]

    # Add shaded region showing improvement
    ax.fill_between(widths_b, cindex_b, cindex_r,
                    where=(cindex_r > cindex_b),
                    alpha=0.2, color='#2ecc71',
                    label='Regularization benefit')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Model Width (log scale)')
    ax.set_ylabel('Test Concordance Index')
    ax.set_title('Effect of L2 Regularization on Double Descent')

    # Add annotation about improvement
    mid_idx = len(widths_b) // 2
    ax.annotate(f'Peak improvement: +{peak_improvement:.3f}',
                xy=(widths_b[peak_b], (cindex_b[peak_b] + cindex_r[peak_b])/2),
                xytext=(256, 0.74),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_path}.png")
    fig.savefig(f"{output_path}.pdf")
    plt.close(fig)
    print(f"  Saved: {output_path}.png/pdf")


def table1_experimental_scenarios(output_path: Path):
    """Table 1: Experimental scenarios summary."""

    scenarios = [
        {
            'Scenario': 'A (Baseline)',
            'Covariate Type': 'Gaussian',
            'N': 1000,
            'Features': 20,
            'Predictive': 10,
            'Censoring': '30%',
            'Purpose': 'Standard setting',
        },
        {
            'Scenario': 'B (Skewed)',
            'Covariate Type': 'Log-normal',
            'N': 1000,
            'Features': 20,
            'Predictive': 10,
            'Censoring': '30%',
            'Purpose': 'Non-Gaussian covariates',
        },
        {
            'Scenario': 'C (Categorical)',
            'Covariate Type': 'Categorical',
            'N': 1000,
            'Features': 20,
            'Predictive': 10,
            'Censoring': '30%',
            'Purpose': 'High cardinality',
        },
        {
            'Scenario': 'D (High Censoring)',
            'Covariate Type': 'Gaussian',
            'N': 1000,
            'Features': 20,
            'Predictive': 10,
            'Censoring': '90%',
            'Purpose': 'Rare events',
        },
    ]

    df = pd.DataFrame(scenarios)

    # Save as CSV
    df.to_csv(f"{output_path}.csv", index=False)

    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='lcccccp{3cm}')
    with open(f"{output_path}.tex", 'w') as f:
        f.write("% Table 1: Experimental Scenarios\n")
        f.write(latex)

    print(f"  Saved: {output_path}.csv, {output_path}.tex")
    return df


def main():
    """Generate all paper figures."""

    # Paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = project_root / "outputs" / "experiments" / "baseline_001"
    high_censoring_dir = project_root / "outputs" / "experiments" / "high_censoring_001"
    regularized_dir = project_root / "outputs" / "experiments" / "regularized_001"

    print("Generating paper figures...")
    print(f"Output directory: {output_dir}\n")

    # Load baseline data
    if baseline_dir.exists():
        baseline_data = load_experiment_data(baseline_dir)
        print(f"Loaded baseline experiment from {baseline_dir}")
    else:
        print(f"ERROR: Baseline experiment not found at {baseline_dir}")
        return 1

    # Figure 1: Main double descent
    print("\n1. Main Double Descent Figure (C-index)")
    figure1_main_double_descent(
        baseline_data,
        output_dir / "fig1_double_descent_cindex",
        metric='c_index'
    )

    print("\n1b. Main Double Descent Figure (IBS)")
    figure1_main_double_descent(
        baseline_data,
        output_dir / "fig1_double_descent_ibs",
        metric='ibs'
    )

    # Figure 2: C-index vs IBS divergence
    print("\n2. C-index vs IBS Divergence")
    figure2_cindex_vs_ibs(
        baseline_data,
        output_dir / "fig2_cindex_vs_ibs"
    )

    # Figure 3: Censoring comparison (requires high censoring experiment)
    if high_censoring_dir.exists():
        print("\n3. Censoring Rate Comparison")
        high_censoring_data = load_experiment_data(high_censoring_dir)
        figure3_censoring_comparison(
            baseline_data,
            high_censoring_data,
            output_dir / "fig3_censoring_comparison"
        )
    else:
        print(f"\n3. Skipping censoring comparison (run high_censoring experiment first)")

    # Figure 4: Regularization comparison (requires regularized experiment)
    if regularized_dir.exists():
        print("\n4. Regularization Comparison")
        regularized_data = load_experiment_data(regularized_dir)
        figure4_regularization_comparison(
            baseline_data,
            regularized_data,
            output_dir / "fig4_regularization_comparison"
        )
    else:
        print(f"\n4. Skipping regularization comparison (run regularized experiment first)")

    # Table 1: Experimental scenarios
    print("\n5. Experimental Scenarios Table")
    table1_experimental_scenarios(output_dir / "table1_scenarios")

    print(f"\n✓ Paper figures generated in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
