"""
End-to-End Example: Timeseries Analysis Workflow
Demonstrates complete pipeline from data collection to analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Import our custom modules
from api_implementation_examples import KalshiClient, PolymarketClient
from timeseries_implementation import (
    TimeNormalizer,
    create_multi_sample_dataset,
    aggregate_calibration_by_time_window,
    ChangePointDetector,
    analyze_recency_bias_with_changepoints,
    plot_calibration_over_time,
    plot_changepoints_on_timeseries
)


# =============================================================================
# STEP 1: DATA COLLECTION WITH PRICE HISTORY
# =============================================================================

def collect_markets_with_history(max_markets=100):
    """
    Collect markets from Kalshi/Polymarket including price history

    Note: This requires API support for historical prices
    For hackathon, you may need to:
    1. Sample prices at multiple points in time for each market
    2. Use archived data if available
    3. Focus on markets with available history
    """
    print("Step 1: Collecting markets with price history...")

    # Kalshi example
    kalshi = KalshiClient()

    # Get settled markets
    settled = kalshi.get_all_settled_markets(max_markets=max_markets)

    markets_with_history = []

    for market in settled:
        # Extract basic info
        market_data = kalshi.extract_market_data(market)

        # TODO: Get price history
        # For Kalshi, you might need to:
        # 1. Use candlestick/OHLC endpoint if available
        # 2. Sample at known intervals
        # 3. Reconstruct from trade data

        # Simulated price history for demonstration
        # In real implementation, fetch from API
        market_data['price_history'] = generate_synthetic_price_history(
            market_data['open_time'],
            market_data['close_time'],
            market_data['last_price'] / 100.0,  # Convert to probability
            market_data['actual_outcome']
        )

        markets_with_history.append(market_data)

    print(f"  Collected {len(markets_with_history)} markets with history")
    return markets_with_history


def generate_synthetic_price_history(open_time, close_time, final_price, outcome):
    """
    Generate synthetic price history for demonstration

    In production, replace with actual API calls

    Args:
        open_time: Market open timestamp
        close_time: Market close timestamp
        final_price: Final price before resolution
        outcome: Actual outcome (0 or 1)

    Returns:
        List of (timestamp, price) tuples
    """
    duration = close_time - open_time
    n_samples = min(100, max(10, duration // (24 * 3600)))  # Daily samples

    timestamps = np.linspace(open_time, close_time, n_samples)

    # Generate realistic price path
    # Start around 0.5, drift toward outcome with noise
    start_price = 0.5 + np.random.normal(0, 0.1)
    start_price = np.clip(start_price, 0.01, 0.99)

    # Prices should trend toward outcome over time
    time_pct = np.linspace(0, 1, n_samples)
    drift = (outcome - start_price) * time_pct

    # Add noise that decreases over time (less uncertainty near end)
    noise = np.random.normal(0, 0.1 * (1 - time_pct**0.5), n_samples)

    prices = start_price + drift + noise
    prices = np.clip(prices, 0.01, 0.99)

    # Ensure final price matches
    prices[-1] = final_price

    return [(int(ts), float(p)) for ts, p in zip(timestamps, prices)]


# =============================================================================
# STEP 2: CATEGORIZE MARKETS
# =============================================================================

def categorize_markets(markets):
    """
    Categorize markets by topic

    Uses keyword matching on titles
    """
    print("\nStep 2: Categorizing markets...")

    categories = {
        'Politics': ['election', 'president', 'congress', 'vote', 'senate', 'democrat', 'republican'],
        'Sports': ['nfl', 'nba', 'mlb', 'nhl', 'win', 'championship', 'playoff', 'soccer', 'football'],
        'Economics': ['gdp', 'unemployment', 'inflation', 'fed', 'interest', 'stock', 'market', 'economy'],
        'Weather': ['temperature', 'rain', 'storm', 'hurricane', 'snow', 'weather', 'climate'],
        'Technology': ['ai', 'tech', 'release', 'launch', 'product', 'software', 'crypto', 'bitcoin'],
    }

    for market in markets:
        title = market.get('title', '').lower()
        subtitle = market.get('subtitle', '').lower()
        text = title + ' ' + subtitle

        # Find matching category
        matched = False
        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                market['category'] = category
                matched = True
                break

        if not matched:
            market['category'] = 'Other'

    # Print distribution
    category_counts = pd.Series([m['category'] for m in markets]).value_counts()
    print("  Category distribution:")
    for cat, count in category_counts.items():
        print(f"    {cat}: {count}")

    return markets


# =============================================================================
# STEP 3: TIME NORMALIZATION & MULTI-SAMPLING
# =============================================================================

def create_timeseries_dataset(markets):
    """
    Create multi-sample dataset at fixed time windows
    """
    print("\nStep 3: Creating timeseries dataset...")

    # Define time windows to sample
    TIME_WINDOWS = [
        30,   # 30 days before
        21,   # 3 weeks
        14,   # 2 weeks
        7,    # 1 week
        3,    # 3 days
        1,    # 1 day
        0.04  # ~1 hour
    ]

    # Create dataset
    df = create_multi_sample_dataset(
        markets,
        time_windows_days=TIME_WINDOWS,
        include_metadata=True
    )

    print(f"  Created dataset with {len(df)} observations from {df['market_id'].nunique()} markets")
    print(f"  Time windows: {sorted(df['days_to_close'].unique())}")

    return df


# =============================================================================
# STEP 4: CALIBRATION ANALYSIS OVER TIME
# =============================================================================

def analyze_calibration_evolution(df):
    """
    Analyze how calibration changes over time
    """
    print("\nStep 4: Analyzing calibration evolution...")

    # Aggregate by time window
    calibration_df = aggregate_calibration_by_time_window(df)

    print("\n  Calibration by time window:")
    print(calibration_df[['days_to_close', 'brier_score', 'ece', 'n_markets']].to_string(index=False))

    # Test hypothesis: calibration improves near resolution
    early_brier = calibration_df[calibration_df['days_to_close'] >= 14]['brier_score'].mean()
    late_brier = calibration_df[calibration_df['days_to_close'] < 7]['brier_score'].mean()

    improvement = early_brier - late_brier
    print(f"\n  Early (>14 days) Brier: {early_brier:.4f}")
    print(f"  Late (<7 days) Brier: {late_brier:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement/early_brier*100:.1f}%)")

    if improvement > 0:
        print("  ✅ Calibration improves as markets approach resolution")
    else:
        print("  ⚠️  No clear improvement detected")

    # Visualize
    plot_calibration_over_time(calibration_df, save_path='calibration_over_time.png')

    return calibration_df


# =============================================================================
# STEP 5: CHANGE POINT DETECTION
# =============================================================================

def detect_all_changepoints(markets, method='pelt'):
    """
    Detect changepoints in all markets

    Args:
        markets: List of market dicts with price_history
        method: 'pelt', 'cusum', or 'threshold'
    """
    print(f"\nStep 5: Detecting changepoints using {method.upper()}...")

    detector = ChangePointDetector()
    markets_with_cps = []

    for market in markets:
        if method == 'pelt':
            cps = detector.detect_pelt(market['price_history'], penalty=10)
        elif method == 'cusum':
            cp = detector.detect_cusum(market['price_history'], threshold=2.5)
            cps = [cp] if cp is not None else []
        else:  # threshold
            cps = detector.detect_simple_threshold(market['price_history'], price_change_threshold=0.15)

        market['changepoints'] = cps
        markets_with_cps.append(market)

    # Statistics
    total_cps = sum(len(m['changepoints']) for m in markets_with_cps)
    markets_with_cps_count = sum(1 for m in markets_with_cps if len(m['changepoints']) > 0)

    print(f"  Total changepoints detected: {total_cps}")
    print(f"  Markets with at least one changepoint: {markets_with_cps_count}/{len(markets_with_cps)}")
    print(f"  Average changepoints per market: {total_cps/len(markets_with_cps):.2f}")

    # Show examples
    print("\n  Example markets with changepoints:")
    examples = [m for m in markets_with_cps if len(m['changepoints']) > 0][:3]
    for market in examples:
        print(f"\n    {market['title']}")
        for cp in market['changepoints']:
            print(f"      {datetime.fromtimestamp(cp['timestamp'])}: Δ{cp['magnitude']:.3f}")

    return markets_with_cps


# =============================================================================
# STEP 6: RECENCY BIAS ANALYSIS
# =============================================================================

def test_recency_bias(markets_with_cps):
    """
    Test if markets are less calibrated after changepoints
    (evidence of recency bias / overreaction)
    """
    print("\nStep 6: Testing for recency bias...")

    result = analyze_recency_bias_with_changepoints(
        markets_with_cps,
        hours_after_cp=24,
        hours_stable=72
    )

    if 'error' not in result:
        print(f"\n  Post-changepoint Brier: {result['post_changepoint_brier']:.4f} (n={result['n_post_cp']})")
        print(f"  Stable period Brier: {result['stable_period_brier']:.4f} (n={result['n_stable']})")
        print(f"  Difference: {result['brier_difference']:.4f}")

        if result['recency_bias_detected']:
            print("  ✅ RECENCY BIAS DETECTED: Markets less calibrated after changepoints")
            print("     → Markets may overreact to new information")
        else:
            print("  ℹ️  No clear recency bias detected")
    else:
        print(f"  ⚠️  {result['error']}")

    return result


# =============================================================================
# STEP 7: CATEGORY COMPARISON
# =============================================================================

def compare_categories(df):
    """
    Compare calibration evolution across categories
    """
    print("\nStep 7: Comparing categories...")

    from timeseries_implementation import compare_calibration_by_category_over_time

    categories = ['Politics', 'Sports', 'Economics']
    category_df = compare_calibration_by_category_over_time(df, categories)

    # Show final Brier scores by category
    print("\n  Calibration by category (at 1 day before close):")
    final_day = category_df[category_df['days_to_close'] <= 1].groupby('category')['brier_score'].mean()
    for cat, brier in final_day.items():
        print(f"    {cat}: {brier:.4f}")

    # Find best/worst
    best = final_day.idxmin()
    worst = final_day.idxmax()
    print(f"\n  Best calibrated: {best} ({final_day[best]:.4f})")
    print(f"  Worst calibrated: {worst} ({final_day[worst]:.4f})")

    return category_df


# =============================================================================
# STEP 8: VISUALIZE EXAMPLE MARKET
# =============================================================================

def visualize_example_market(markets_with_cps):
    """
    Visualize changepoints on a specific market
    """
    print("\nStep 8: Visualizing example market with changepoints...")

    # Find market with changepoints
    examples = [m for m in markets_with_cps if len(m['changepoints']) > 2]

    if examples:
        market = examples[0]
        print(f"  Market: {market['title']}")
        plot_changepoints_on_timeseries(
            market,
            market['changepoints'],
            save_path='changepoints_example.png'
        )
    else:
        print("  No markets with multiple changepoints found")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """
    Complete end-to-end workflow
    """
    print("\n" + "="*70)
    print("PREDICTION MARKETS TIMESERIES CALIBRATION ANALYSIS")
    print("="*70)

    # Step 1: Collect data
    markets = collect_markets_with_history(max_markets=200)

    # Step 2: Categorize
    markets = categorize_markets(markets)

    # Save markets to file
    with open('markets_with_history.json', 'w') as f:
        json.dump(markets, f, indent=2)
    print("\nSaved markets to markets_with_history.json")

    # Step 3: Create timeseries dataset
    df = create_timeseries_dataset(markets)

    # Save dataset
    df.to_csv('timeseries_dataset.csv', index=False)
    print("Saved dataset to timeseries_dataset.csv")

    # Step 4: Analyze calibration evolution
    calibration_df = analyze_calibration_evolution(df)

    # Step 5: Detect changepoints
    markets_with_cps = detect_all_changepoints(markets, method='pelt')

    # Step 6: Test recency bias
    recency_result = test_recency_bias(markets_with_cps)

    # Step 7: Compare categories
    category_df = compare_categories(df)

    # Step 8: Visualize example
    visualize_example_market(markets_with_cps)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - markets_with_history.json")
    print("  - timeseries_dataset.csv")
    print("  - calibration_over_time.png")
    print("  - changepoints_example.png")

    # Summary of findings
    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)

    early_brier = calibration_df[calibration_df['days_to_close'] >= 14]['brier_score'].mean()
    late_brier = calibration_df[calibration_df['days_to_close'] < 7]['brier_score'].mean()
    improvement_pct = (early_brier - late_brier) / early_brier * 100

    print(f"\n1. TEMPORAL CALIBRATION:")
    print(f"   Calibration improves {improvement_pct:.1f}% as markets approach resolution")

    if 'error' not in recency_result:
        if recency_result['recency_bias_detected']:
            print(f"\n2. RECENCY BIAS:")
            print(f"   Markets overreact to changepoints by {recency_result['brier_difference']:.4f} Brier points")
        else:
            print(f"\n2. RECENCY BIAS:")
            print(f"   No significant recency bias detected")

    print(f"\n3. SAMPLE SIZE:")
    print(f"   {len(df)} total predictions from {df['market_id'].nunique()} markets")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run complete workflow
    main()
