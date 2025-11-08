"""
Timeseries Analysis Implementation for Prediction Markets
Includes time normalization, multi-sampling, and change point detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# TIME NORMALIZATION METHODS
# =============================================================================

class TimeNormalizer:
    """Normalize time across markets with different durations"""

    @staticmethod
    def get_price_at_time(price_history: List[Tuple[int, float]],
                          target_timestamp: int,
                          max_distance: int = 3600) -> Optional[float]:
        """
        Get price closest to target timestamp

        Args:
            price_history: List of (timestamp, price) tuples
            target_timestamp: Target Unix timestamp
            max_distance: Maximum acceptable distance in seconds

        Returns:
            Price at target time, or None if no price within max_distance
        """
        if not price_history:
            return None

        closest = min(price_history, key=lambda p: abs(p[0] - target_timestamp))

        if abs(closest[0] - target_timestamp) <= max_distance:
            return closest[1]

        return None

    @staticmethod
    def normalize_by_time_to_close(market: Dict,
                                   time_windows: Dict[str, int]) -> Dict[str, float]:
        """
        Sample market at fixed time windows before close

        Args:
            market: Market dict with 'close_time' and 'price_history'
            time_windows: Dict mapping names to seconds before close

        Returns:
            Dict mapping window names to prices
        """
        close_time = market['close_time']
        price_history = market.get('price_history', [])

        if isinstance(price_history, list) and len(price_history) > 0:
            # Assume format: [(timestamp, price), ...]
            pass
        else:
            # No price history available
            return {}

        sampled_prices = {}

        for window_name, seconds_before in time_windows.items():
            target_time = close_time - seconds_before

            price = TimeNormalizer.get_price_at_time(
                price_history,
                target_time,
                max_distance=6*3600  # 6 hour tolerance
            )

            if price is not None:
                sampled_prices[window_name] = price

        return sampled_prices

    @staticmethod
    def normalize_by_percentage(market: Dict,
                               n_samples: int = 10) -> List[Tuple[float, float]]:
        """
        Sample at percentage points of market lifetime

        Args:
            market: Market dict with 'open_time', 'close_time', 'price_history'
            n_samples: Number of evenly-spaced samples

        Returns:
            List of (percentage, price) tuples
        """
        open_time = market['open_time']
        close_time = market['close_time']
        price_history = market.get('price_history', [])

        if not price_history:
            return []

        total_duration = close_time - open_time
        sampled = []

        for i in range(n_samples):
            pct = i / (n_samples - 1) if n_samples > 1 else 0
            target_time = open_time + int(pct * total_duration)

            price = TimeNormalizer.get_price_at_time(
                price_history,
                target_time,
                max_distance=int(total_duration / (n_samples * 2))
            )

            if price is not None:
                sampled.append((pct, price))

        return sampled


# =============================================================================
# MULTI-SAMPLE DATASET CREATION
# =============================================================================

def create_multi_sample_dataset(markets: List[Dict],
                               time_windows_days: List[float],
                               include_metadata: bool = True) -> pd.DataFrame:
    """
    Create panel dataset with multiple samples per market

    Args:
        markets: List of market dictionaries
        time_windows_days: List of days before close to sample at
        include_metadata: Whether to include category, volume, etc.

    Returns:
        DataFrame with columns:
            - market_id
            - days_to_close
            - prediction
            - outcome
            - category (optional)
            - volume (optional)
            - timestamp (when prediction was made)
    """
    rows = []

    for market in markets:
        market_id = market.get('ticker') or market.get('id')
        outcome = market.get('actual_outcome')
        category = market.get('category', 'Unknown')
        volume = market.get('volume', 0)
        close_time = market.get('close_time')

        if outcome is None or close_time is None:
            continue

        for days_before in time_windows_days:
            seconds_before = int(days_before * 24 * 3600)
            target_time = close_time - seconds_before

            price = TimeNormalizer.get_price_at_time(
                market.get('price_history', []),
                target_time,
                max_distance=6*3600
            )

            if price is not None:
                row = {
                    'market_id': market_id,
                    'days_to_close': days_before,
                    'prediction': price,
                    'outcome': outcome,
                    'timestamp': target_time
                }

                if include_metadata:
                    row['category'] = category
                    row['volume'] = volume

                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_calibration_by_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate calibration metrics for each time window

    Args:
        df: Multi-sample DataFrame from create_multi_sample_dataset

    Returns:
        DataFrame with calibration metrics by time window
    """
    results = []

    for days in sorted(df['days_to_close'].unique()):
        subset = df[df['days_to_close'] == days]

        predictions = subset['prediction'].values
        outcomes = subset['outcome'].values

        # Brier score
        brier = np.mean((predictions - outcomes) ** 2)

        # ECE (Expected Calibration Error)
        ece = calculate_ece_simple(predictions, outcomes)

        # Mean absolute error
        mae = np.mean(np.abs(predictions - outcomes))

        results.append({
            'days_to_close': days,
            'brier_score': brier,
            'ece': ece,
            'mae': mae,
            'n_markets': len(subset),
            'mean_prediction': np.mean(predictions),
            'mean_outcome': np.mean(outcomes)
        })

    return pd.DataFrame(results)


def calculate_ece_simple(predictions: np.ndarray,
                        outcomes: np.ndarray,
                        n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges[:-1]) - 1

    ece = 0.0
    total = len(predictions)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(outcomes[mask])
            bin_conf = np.mean(predictions[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


# =============================================================================
# CHANGE POINT DETECTION
# =============================================================================

class ChangePointDetector:
    """Detect change points in market price timeseries"""

    @staticmethod
    def detect_cusum(price_history: List[Tuple[int, float]],
                     threshold: float = 3.0) -> Optional[Dict]:
        """
        Detect single change point using CUSUM

        Args:
            price_history: List of (timestamp, price) tuples
            threshold: Sensitivity (higher = less sensitive)

        Returns:
            Dict with changepoint info, or None if no changepoint
        """
        if len(price_history) < 5:
            return None

        prices = np.array([p[1] for p in price_history])
        timestamps = np.array([p[0] for p in price_history])

        # Calculate mean and std
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        if std_price == 0:
            return None

        # Center and cumsum
        centered = prices - mean_price
        cumsum = np.cumsum(centered)

        # Detect exceedance
        threshold_value = threshold * std_price
        exceeded = np.where(np.abs(cumsum) > threshold_value)[0]

        if len(exceeded) > 0:
            cp_idx = exceeded[0]
            return {
                'index': int(cp_idx),
                'timestamp': int(timestamps[cp_idx]),
                'price_before': float(prices[cp_idx - 1] if cp_idx > 0 else prices[cp_idx]),
                'price_after': float(prices[cp_idx]),
                'magnitude': float(abs(prices[cp_idx] - prices[cp_idx - 1]) if cp_idx > 0 else 0)
            }

        return None

    @staticmethod
    def detect_pelt(price_history: List[Tuple[int, float]],
                    penalty: float = 10) -> List[Dict]:
        """
        Detect multiple change points using PELT algorithm

        Requires: pip install ruptures

        Args:
            price_history: List of (timestamp, price) tuples
            penalty: Controls number of changepoints (higher = fewer)

        Returns:
            List of changepoint dicts
        """
        try:
            import ruptures as rpt
        except ImportError:
            print("Warning: ruptures library not installed. Install with: pip install ruptures")
            return []

        if len(price_history) < 10:
            return []

        prices = np.array([p[1] for p in price_history])
        timestamps = np.array([p[0] for p in price_history])

        # Reshape for ruptures (needs 2D array)
        signal = prices.reshape(-1, 1)

        try:
            # PELT algorithm
            algo = rpt.Pelt(model="l2", min_size=3).fit(signal)
            changepoint_indices = algo.predict(pen=penalty)

            # Extract details
            cp_details = []
            for cp_idx in changepoint_indices[:-1]:  # Last is always end
                if cp_idx > 0 and cp_idx < len(prices):
                    cp_details.append({
                        'index': int(cp_idx),
                        'timestamp': int(timestamps[cp_idx]),
                        'price_before': float(prices[cp_idx - 1]),
                        'price_after': float(prices[cp_idx]),
                        'magnitude': float(abs(prices[cp_idx] - prices[cp_idx - 1]))
                    })

            return cp_details

        except Exception as e:
            print(f"PELT detection failed: {e}")
            return []

    @staticmethod
    def detect_simple_threshold(price_history: List[Tuple[int, float]],
                               price_change_threshold: float = 0.1) -> List[Dict]:
        """
        Simple heuristic: detect when price changes by more than threshold

        Args:
            price_history: List of (timestamp, price) tuples
            price_change_threshold: Minimum absolute price change to flag

        Returns:
            List of changepoint dicts
        """
        changepoints = []

        for i in range(1, len(price_history)):
            prev_price = price_history[i-1][1]
            curr_price = price_history[i][1]
            change = abs(curr_price - prev_price)

            if change >= price_change_threshold:
                changepoints.append({
                    'index': i,
                    'timestamp': price_history[i][0],
                    'price_before': prev_price,
                    'price_after': curr_price,
                    'magnitude': change
                })

        return changepoints


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_recency_bias_with_changepoints(markets_with_cps: List[Dict],
                                          hours_after_cp: int = 24,
                                          hours_stable: int = 72) -> Dict:
    """
    Test if markets are less calibrated immediately after change points
    (evidence of recency bias)

    Args:
        markets_with_cps: Markets with 'changepoints' field
        hours_after_cp: Window after changepoint to sample
        hours_stable: Minimum hours from any CP to be "stable"

    Returns:
        Dict with comparison metrics
    """
    post_cp_predictions = []
    post_cp_outcomes = []
    stable_predictions = []
    stable_outcomes = []

    for market in markets_with_cps:
        changepoints = market.get('changepoints', [])
        if not changepoints:
            continue

        outcome = market.get('actual_outcome')
        if outcome is None:
            continue

        # Get prices after changepoints
        for cp in changepoints:
            target_time = cp['timestamp'] + (hours_after_cp * 3600)
            price = TimeNormalizer.get_price_at_time(
                market.get('price_history', []),
                target_time,
                max_distance=hours_after_cp * 3600
            )

            if price is not None:
                post_cp_predictions.append(price)
                post_cp_outcomes.append(outcome)

        # Get prices during stable periods
        # (more than hours_stable from any changepoint)
        price_history = market.get('price_history', [])
        for ts, price in price_history:
            # Check distance to all changepoints
            min_distance = min(
                [abs(ts - cp['timestamp']) for cp in changepoints]
                if changepoints else [float('inf')]
            )

            if min_distance > hours_stable * 3600:
                stable_predictions.append(price)
                stable_outcomes.append(outcome)

    # Calculate metrics
    if len(post_cp_predictions) > 0 and len(stable_predictions) > 0:
        post_cp_brier = np.mean(
            [(p - o)**2 for p, o in zip(post_cp_predictions, post_cp_outcomes)]
        )
        stable_brier = np.mean(
            [(p - o)**2 for p, o in zip(stable_predictions, stable_outcomes)]
        )

        return {
            'post_changepoint_brier': post_cp_brier,
            'stable_period_brier': stable_brier,
            'brier_difference': post_cp_brier - stable_brier,
            'recency_bias_detected': post_cp_brier > stable_brier,
            'n_post_cp': len(post_cp_predictions),
            'n_stable': len(stable_predictions)
        }

    return {'error': 'Insufficient data'}


def compare_calibration_by_category_over_time(df: pd.DataFrame,
                                              categories: List[str]) -> pd.DataFrame:
    """
    Compare how calibration evolves for different market categories

    Args:
        df: Multi-sample DataFrame with 'category' column
        categories: List of categories to compare

    Returns:
        DataFrame with calibration by category and time window
    """
    results = []

    for category in categories:
        cat_data = df[df['category'] == category]

        for days in sorted(cat_data['days_to_close'].unique()):
            subset = cat_data[cat_data['days_to_close'] == days]

            if len(subset) < 10:  # Minimum sample size
                continue

            predictions = subset['prediction'].values
            outcomes = subset['outcome'].values

            brier = np.mean((predictions - outcomes) ** 2)

            results.append({
                'category': category,
                'days_to_close': days,
                'brier_score': brier,
                'n_markets': len(subset)
            })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_calibration_over_time(df: pd.DataFrame,
                               save_path: Optional[str] = None):
    """
    Plot how calibration changes as markets approach resolution

    Args:
        df: Output from aggregate_calibration_by_time_window
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Brier Score over time
    ax1.plot(df['days_to_close'], df['brier_score'],
             marker='o', linewidth=2.5, markersize=10, color='#1f77b4')
    ax1.set_xlabel('Days to Close', fontsize=14)
    ax1.set_ylabel('Brier Score', fontsize=14)
    ax1.set_title('Calibration Improves Near Resolution', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # So time flows left to right

    # Add sample size annotations
    for _, row in df.iterrows():
        ax1.annotate(f"n={row['n_markets']}",
                    xy=(row['days_to_close'], row['brier_score']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, alpha=0.7)

    # Plot 2: ECE over time
    ax2.plot(df['days_to_close'], df['ece'],
             marker='s', linewidth=2.5, markersize=10, color='#ff7f0e')
    ax2.set_xlabel('Days to Close', fontsize=14)
    ax2.set_ylabel('Expected Calibration Error', fontsize=14)
    ax2.set_title('ECE Over Time', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_category_comparison_over_time(df: pd.DataFrame,
                                      save_path: Optional[str] = None):
    """
    Plot calibration evolution for different categories

    Args:
        df: Output from compare_calibration_by_category_over_time
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    categories = df['category'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    for i, category in enumerate(categories):
        cat_data = df[df['category'] == category].sort_values('days_to_close')

        ax.plot(cat_data['days_to_close'], cat_data['brier_score'],
               marker='o', linewidth=2, markersize=8,
               label=category, color=colors[i])

    ax.set_xlabel('Days to Close', fontsize=14)
    ax.set_ylabel('Brier Score', fontsize=14)
    ax.set_title('Calibration by Category Over Time', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_changepoints_on_timeseries(market: Dict,
                                   changepoints: List[Dict],
                                   save_path: Optional[str] = None):
    """
    Visualize detected changepoints on price timeseries

    Args:
        market: Market dict with price_history
        changepoints: List of changepoint dicts from detector
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    price_history = market.get('price_history', [])
    if not price_history:
        print("No price history available")
        return

    timestamps = [p[0] for p in price_history]
    prices = [p[1] for p in price_history]
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]

    # Plot price line
    ax.plot(dates, prices, linewidth=2, color='#1f77b4', label='Market Price')

    # Mark changepoints
    for cp in changepoints:
        cp_date = datetime.fromtimestamp(cp['timestamp'])
        cp_price = cp['price_after']

        ax.axvline(cp_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.scatter([cp_date], [cp_price], color='red', s=100, zorder=5)

        # Annotate magnitude
        ax.annotate(f"Δ{cp['magnitude']:.2f}",
                   xy=(cp_date, cp_price),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.set_title(f"Change Points: {market.get('title', 'Unknown Market')}",
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Timeseries Analysis for Prediction Markets")
    print("=" * 60)

    # This would use real market data from API
    print("\nExample workflow:")
    print("1. Load markets with price history")
    print("2. Create multi-sample dataset at fixed time windows")
    print("3. Analyze calibration evolution over time")
    print("4. Detect change points and test for recency bias")
    print("5. Compare categories")

    # Synthetic example
    print("\n" + "=" * 60)
    print("Run with real data from api_implementation_examples.py")
