"""
Prediction Markets Calibration - Visualization Examples
Creates calibration plots and analysis visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.calibration import calibration_curve


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_calibration_curve(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10,
    title: str = "Calibration Plot",
    save_path: str = None
) -> None:
    """
    Create a calibration plot (reliability diagram)

    Args:
        predictions: Predicted probabilities (0-1)
        outcomes: Actual outcomes (0 or 1)
        n_bins: Number of bins
        title: Plot title
        save_path: Optional path to save figure
    """
    # Calculate calibration curve using sklearn
    fraction_of_positives, mean_predicted_value = calibration_curve(
        outcomes, predictions, n_bins=n_bins, strategy='uniform'
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-',
            label='Model', linewidth=2, markersize=8)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add diagonal shading for reference
    ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration plot to {save_path}")

    plt.show()


def plot_calibration_comparison(
    platform_data: Dict[str, Tuple[List[float], List[int]]],
    n_bins: int = 10,
    title: str = "Platform Calibration Comparison",
    save_path: str = None
) -> None:
    """
    Compare calibration curves for multiple platforms

    Args:
        platform_data: Dict mapping platform names to (predictions, outcomes) tuples
        n_bins: Number of bins
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (platform, (predictions, outcomes)) in enumerate(platform_data.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            outcomes, predictions, n_bins=n_bins, strategy='uniform'
        )

        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                label=platform, linewidth=2, markersize=8, color=colors[i % len(colors)])

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)

    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def plot_brier_score_distribution(
    market_data: List[Dict],
    category_field: str = "category",
    title: str = "Brier Score by Category",
    save_path: str = None
) -> None:
    """
    Create box plot of Brier scores by category

    Args:
        market_data: List of market dictionaries with predictions and outcomes
        category_field: Field name for category
        title: Plot title
        save_path: Optional path to save figure
    """
    # Calculate individual Brier scores
    for market in market_data:
        pred = market.get("predicted_probability")
        outcome = market.get("actual_outcome")
        if pred is not None and outcome is not None:
            market["brier_score"] = (pred - outcome) ** 2

    # Convert to DataFrame
    df = pd.DataFrame(market_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Box plot
    categories = sorted(df[category_field].unique())
    data_by_category = [df[df[category_field] == cat]["brier_score"].dropna()
                        for cat in categories]

    bp = ax.boxplot(data_by_category, labels=categories, patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#1f77b4')
        patch.set_alpha(0.7)

    # Formatting
    ax.set_xlabel('Category', fontsize=14)
    ax.set_ylabel('Brier Score', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    # Add mean line
    ax.axhline(y=df["brier_score"].mean(), color='r', linestyle='--',
               label=f'Overall Mean: {df["brier_score"].mean():.3f}', linewidth=2)
    ax.legend(fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Brier score distribution to {save_path}")

    plt.show()


def plot_favorite_longshot_bias(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 20,
    title: str = "Favorite-Longshot Bias Analysis",
    save_path: str = None
) -> None:
    """
    Analyze and visualize favorite-longshot bias

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        n_bins: Number of bins (use more for this analysis)
        title: Plot title
        save_path: Optional path to save figure
    """
    # Calculate calibration data
    fraction_of_positives, mean_predicted_value = calibration_curve(
        outcomes, predictions, n_bins=n_bins, strategy='uniform'
    )

    # Calculate deviation from perfect calibration
    deviations = fraction_of_positives - mean_predicted_value

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Calibration curve with emphasis on extremes
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-',
             label='Market Calibration', linewidth=2, markersize=8, color='#1f77b4')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Highlight extreme regions
    ax1.axvspan(0, 0.2, alpha=0.2, color='red', label='Longshots')
    ax1.axvspan(0.8, 1, alpha=0.2, color='green', label='Favorites')

    ax1.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax1.set_ylabel('Fraction of Positives', fontsize=14)
    ax1.set_title('Calibration with Extreme Probabilities Highlighted', fontsize=14)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Deviation from calibration
    ax2.bar(mean_predicted_value, deviations, width=0.04, alpha=0.7, color='#ff7f0e')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax2.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax2.set_ylabel('Calibration Error\n(Observed - Predicted)', fontsize=14)
    ax2.set_title('Deviation from Perfect Calibration', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotations for bias
    if len(deviations) > 0:
        # Check extremes
        low_prob_mask = mean_predicted_value < 0.2
        high_prob_mask = mean_predicted_value > 0.8

        if low_prob_mask.any():
            low_bias = np.mean(deviations[low_prob_mask])
            ax2.text(0.1, ax2.get_ylim()[1] * 0.9,
                    f'Longshot bias: {low_bias:.3f}',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

        if high_prob_mask.any():
            high_bias = np.mean(deviations[high_prob_mask])
            ax2.text(0.9, ax2.get_ylim()[1] * 0.9,
                    f'Favorite bias: {high_bias:.3f}',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved favorite-longshot analysis to {save_path}")

    plt.show()


def plot_volume_vs_calibration(
    market_data: List[Dict],
    volume_field: str = "volume",
    n_quantiles: int = 5,
    title: str = "Calibration by Volume Tier",
    save_path: str = None
) -> None:
    """
    Analyze calibration quality by market volume

    Args:
        market_data: List of market dictionaries
        volume_field: Field name for volume
        n_quantiles: Number of volume tiers to create
        title: Plot title
        save_path: Optional path to save figure
    """
    df = pd.DataFrame(market_data)

    # Create volume quantiles
    df['volume_tier'] = pd.qcut(df[volume_field], q=n_quantiles,
                                 labels=[f'Q{i+1}' for i in range(n_quantiles)])

    # Calculate metrics by tier
    metrics = []
    for tier in df['volume_tier'].unique():
        tier_data = df[df['volume_tier'] == tier]
        predictions = tier_data['predicted_probability'].values
        outcomes = tier_data['actual_outcome'].values

        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(outcomes))
        predictions = predictions[mask]
        outcomes = outcomes[mask]

        if len(predictions) > 0:
            brier = np.mean((predictions - outcomes) ** 2)
            metrics.append({
                'tier': tier,
                'brier_score': brier,
                'n_markets': len(predictions),
                'avg_volume': tier_data[volume_field].mean()
            })

    metrics_df = pd.DataFrame(metrics)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Brier score by volume tier
    ax1.bar(metrics_df['tier'], metrics_df['brier_score'], alpha=0.7, color='#1f77b4')
    ax1.set_xlabel('Volume Tier', fontsize=14)
    ax1.set_ylabel('Brier Score', fontsize=14)
    ax1.set_title('Brier Score by Volume Tier\n(Lower is Better)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample size annotations
    for i, row in metrics_df.iterrows():
        ax1.text(i, row['brier_score'] + 0.005, f"n={row['n_markets']}",
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Market count by tier
    ax2.bar(metrics_df['tier'], metrics_df['n_markets'], alpha=0.7, color='#ff7f0e')
    ax2.set_xlabel('Volume Tier', fontsize=14)
    ax2.set_ylabel('Number of Markets', fontsize=14)
    ax2.set_title('Sample Size by Volume Tier', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved volume analysis to {save_path}")

    plt.show()


def plot_time_to_resolution_analysis(
    market_data: List[Dict],
    time_field: str = "days_to_resolution",
    title: str = "Calibration Over Time",
    save_path: str = None
) -> None:
    """
    Analyze how calibration changes as markets approach resolution

    Args:
        market_data: List of market dictionaries with time information
        time_field: Field name for days/hours to resolution
        title: Plot title
        save_path: Optional path to save figure
    """
    df = pd.DataFrame(market_data)

    # Create time bins
    time_bins = [0, 1, 3, 7, 14, 30, 90, 365]
    bin_labels = ['<1d', '1-3d', '3-7d', '1-2w', '2-4w', '1-3m', '>3m']

    df['time_bin'] = pd.cut(df[time_field], bins=time_bins, labels=bin_labels[:len(time_bins)-1])

    # Calculate metrics by time bin
    metrics = []
    for bin_label in df['time_bin'].unique():
        if pd.isna(bin_label):
            continue

        bin_data = df[df['time_bin'] == bin_label]
        predictions = bin_data['predicted_probability'].values
        outcomes = bin_data['actual_outcome'].values

        mask = ~(np.isnan(predictions) | np.isnan(outcomes))
        predictions = predictions[mask]
        outcomes = outcomes[mask]

        if len(predictions) > 10:  # Minimum sample size
            brier = np.mean((predictions - outcomes) ** 2)
            metrics.append({
                'time_bin': bin_label,
                'brier_score': brier,
                'n_markets': len(predictions)
            })

    metrics_df = pd.DataFrame(metrics)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(range(len(metrics_df)), metrics_df['brier_score'],
            marker='o', markersize=10, linewidth=2.5, color='#1f77b4')

    ax.set_xlabel('Time to Resolution', fontsize=14)
    ax.set_ylabel('Brier Score', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df['time_bin'])
    ax.grid(True, alpha=0.3)

    # Add sample size annotations
    for i, row in metrics_df.iterrows():
        ax.annotate(f"n={row['n_markets']}", xy=(i, row['brier_score']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved time analysis to {save_path}")

    plt.show()


def create_summary_dashboard(
    platform_data: Dict[str, List[Dict]],
    save_path: str = "calibration_dashboard.png"
) -> None:
    """
    Create comprehensive summary dashboard

    Args:
        platform_data: Dictionary mapping platform names to market data lists
        save_path: Path to save dashboard
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {'Kalshi': '#1f77b4', 'Polymarket': '#ff7f0e'}

    # Calculate overall metrics
    all_metrics = {}
    for platform, markets in platform_data.items():
        df = pd.DataFrame(markets)
        predictions = df['predicted_probability'].dropna().values
        outcomes = df['actual_outcome'].dropna().values

        mask = ~(np.isnan(predictions) | np.isnan(outcomes))
        predictions = predictions[mask]
        outcomes = outcomes[mask]

        all_metrics[platform] = {
            'brier': np.mean((predictions - outcomes) ** 2),
            'predictions': predictions,
            'outcomes': outcomes,
            'n_markets': len(predictions)
        }

    # 1. Calibration Curves Comparison (top left, span 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for platform, metrics in all_metrics.items():
        frac_pos, mean_pred = calibration_curve(
            metrics['outcomes'], metrics['predictions'], n_bins=10
        )
        ax1.plot(mean_pred, frac_pos, 's-', label=platform,
                linewidth=2.5, markersize=10, color=colors.get(platform))

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency', fontsize=12)
    ax1.set_title('Calibration Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Brier Score Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    platforms = list(all_metrics.keys())
    brier_scores = [all_metrics[p]['brier'] for p in platforms]
    bars = ax2.bar(platforms, brier_scores, color=[colors.get(p) for p in platforms], alpha=0.7)
    ax2.set_ylabel('Brier Score', fontsize=12)
    ax2.set_title('Overall Brier Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, score in zip(bars, brier_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11)

    # 3-5. Category breakdowns (middle row)
    # This would need category data - placeholder for now

    # 6. Sample sizes
    ax6 = fig.add_subplot(gs[2, 0])
    sample_sizes = [all_metrics[p]['n_markets'] for p in platforms]
    ax6.bar(platforms, sample_sizes, color=[colors.get(p) for p in platforms], alpha=0.7)
    ax6.set_ylabel('Number of Markets', fontsize=12)
    ax6.set_title('Sample Sizes', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Prediction Markets Calibration Dashboard', fontsize=18, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved dashboard to {save_path}")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(42)

    # Create sample predictions and outcomes
    n_samples = 1000
    predictions = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic probabilities
    outcomes = (np.random.random(n_samples) < predictions).astype(int)

    print("Creating example visualizations...")

    # Example 1: Basic calibration plot
    plot_calibration_curve(predictions, outcomes, title="Example Calibration Plot")

    # Example 2: Favorite-longshot bias
    plot_favorite_longshot_bias(predictions, outcomes)

    print("\nVisualization examples complete!")
    print("Use these functions with your real market data.")
