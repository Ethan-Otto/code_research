# Timeseries Analysis for Prediction Markets Calibration
## Research on Sampling, Normalization, and Change Point Detection

## Executive Summary

This document explores the challenges and solutions for analyzing prediction market calibration over time, with focus on:
1. **Time normalization** across markets with different durations
2. **Sampling strategies** for meaningful timeseries analysis
3. **Change point detection** for identifying information shocks
4. **Practical implementation** approaches for hackathon analysis

---

## The Core Challenge: Markets as Different "Forecasters"

### The Problem

Prediction markets have fundamentally different characteristics:

| Market Characteristic | Example A | Example B |
|----------------------|-----------|-----------|
| **Duration** | 365 days | 7 days |
| **Resolution Date** | Dec 31, 2025 | Next Sunday |
| **Data Points** | 365 price snapshots | 7 price snapshots |
| **Volatility** | Low (long time to update) | High (rapid reactions) |

**Question**: How do we compare calibration across these markets as if they were "the same forecaster"?

### Why This Matters

If we naively sample at fixed time intervals:
- Long markets get over-represented (365 samples vs 7)
- Short markets get under-represented
- Temporal patterns (e.g., "accuracy improves near close") get obscured
- Different markets are at different "stages" of their lifecycle

---

## Solution 1: Time-to-Resolution Normalization

### Concept

Instead of calendar time, use **"time remaining until resolution"** as the universal time axis.

### Implementation Approaches

#### Approach 1A: Fixed Time Windows Before Close

Sample all markets at standardized checkpoints:

```python
TIME_WINDOWS = {
    '30d_before': 30 * 24 * 3600,  # seconds
    '7d_before': 7 * 24 * 3600,
    '1d_before': 1 * 24 * 3600,
    '1h_before': 3600,
    'final': 0
}

def normalize_by_time_to_close(market_history):
    """
    Sample market prices at fixed intervals before resolution

    Args:
        market_history: dict with 'resolution_time' and 'price_history'

    Returns:
        dict of {window_name: price} for each checkpoint
    """
    resolution_ts = market_history['resolution_time']
    prices = market_history['price_history']  # [(timestamp, price), ...]

    sampled_prices = {}

    for window_name, seconds_before in TIME_WINDOWS.items():
        target_time = resolution_ts - seconds_before

        # Find closest price to target time
        closest_price = min(
            prices,
            key=lambda p: abs(p[0] - target_time)
        )

        sampled_prices[window_name] = closest_price[1]

    return sampled_prices
```

**Advantages**:
- ✅ All markets contribute equally at each time window
- ✅ Easy to implement and interpret
- ✅ Directly tests hypothesis: "calibration improves near resolution"
- ✅ Works even with sparse price data

**Disadvantages**:
- ❌ Markets shorter than window (e.g., 2-day market, 30-day window) get excluded
- ❌ Loses information between windows
- ❌ Assumes discrete snapshots represent continuous process

**Research Evidence**:
Good Judgment Project found that **weighting recent opinions more strongly** improved accuracy. Analysis of Kalshi markets showed Brier Skill Score peaked at **3 weeks before close** (0.513) vs midpoint (0.367).

---

#### Approach 1B: Percentage of Lifetime Normalization

Normalize time as percentage of market's total lifetime:

```python
def normalize_by_lifetime_percentage(market_history, n_samples=10):
    """
    Sample at percentage points of market lifetime

    Args:
        market_history: dict with 'open_time', 'close_time', 'price_history'
        n_samples: number of evenly-spaced samples

    Returns:
        list of (percentage, price) tuples
    """
    open_ts = market_history['open_time']
    close_ts = market_history['close_time']
    total_duration = close_ts - open_ts

    prices = sorted(market_history['price_history'], key=lambda x: x[0])

    sampled = []

    for i in range(n_samples):
        pct = i / (n_samples - 1)  # 0%, 11%, 22%, ..., 100%
        target_time = open_ts + (pct * total_duration)

        # Find closest price
        closest = min(prices, key=lambda p: abs(p[0] - target_time))
        sampled.append((pct, closest[1]))

    return sampled

# Now aggregate across ALL markets at same percentage points
def aggregate_by_percentage(all_markets, n_samples=10):
    """Aggregate calibration at each percentage of lifetime"""

    percentage_bins = [[] for _ in range(n_samples)]

    for market in all_markets:
        samples = normalize_by_lifetime_percentage(market, n_samples)
        for i, (pct, price) in enumerate(samples):
            percentage_bins[i].append({
                'prediction': price,
                'outcome': market['actual_outcome']
            })

    # Calculate Brier score at each percentage point
    calibration_over_time = []
    for i, bin_data in enumerate(percentage_bins):
        preds = [d['prediction'] for d in bin_data]
        outcomes = [d['outcome'] for d in bin_data]

        brier = np.mean([(p - o)**2 for p, o in zip(preds, outcomes)])
        calibration_over_time.append({
            'percentage': i / (n_samples - 1),
            'brier_score': brier,
            'n_markets': len(bin_data)
        })

    return calibration_over_time
```

**Advantages**:
- ✅ Works for ALL markets regardless of duration
- ✅ Captures full lifecycle dynamics
- ✅ Natural way to compare "early" vs "late" predictions
- ✅ Statistically elegant

**Disadvantages**:
- ❌ Assumes lifecycle is meaningful (but 1-day and 365-day markets are different!)
- ❌ "50% of lifetime" means different things (12 hours vs 6 months)
- ❌ Information arrival isn't uniform across lifetime

**When to Use**:
When you want to answer: "How does calibration evolve as markets mature from 0% → 100% of their lifetime?"

---

#### Approach 1C: Hybrid Approach (RECOMMENDED)

Combine both methods:

1. **Short-term markets** (< 30 days): Use percentage normalization
2. **Long-term markets** (≥ 30 days): Use fixed time windows
3. **All markets**: Use "final price before resolution" as universal endpoint

```python
def hybrid_normalization(market):
    """Adaptive normalization based on market duration"""

    duration_days = (market['close_time'] - market['open_time']) / (24*3600)

    if duration_days < 30:
        # Short-term: use percentage
        return normalize_by_lifetime_percentage(market, n_samples=5)
    else:
        # Long-term: use fixed windows
        return normalize_by_time_to_close(market)
```

---

## Solution 2: Multi-Sample Timeseries Analysis

### Can We Sample More Than Once Meaningfully?

**YES!** In fact, sampling multiple times from the same market is not only valid but **increases statistical power**.

### Key Insight: Panel Data Structure

Each market can contribute **multiple observations** at different time points. This creates a **panel data structure**:

```
Market_ID | Time_to_Close | Prediction | Outcome | Category | Volume
----------|---------------|------------|---------|----------|--------
Market_1  | 30 days       | 0.60       | 1       | Politics | High
Market_1  | 7 days        | 0.65       | 1       | Politics | High
Market_1  | 1 day         | 0.70       | 1       | Politics | High
Market_2  | 30 days       | 0.40       | 0       | Sports   | Low
Market_2  | 7 days        | 0.35       | 0       | Sports   | Low
...
```

### Statistical Validity

**Why this is valid**:
1. Each row is a **different forecast** at different information states
2. Markets naturally update as new information arrives
3. Analogous to: comparing a weather forecast 7 days out vs 1 day out

**Caution - Autocorrelation**:
Samples from the same market are **not independent**. Must account for this:

```python
# WRONG: Treat all samples as independent
from sklearn.metrics import brier_score_loss
all_predictions = []
all_outcomes = []
for market in markets:
    for timepoint in ['30d', '7d', '1d']:
        all_predictions.append(market[timepoint]['price'])
        all_outcomes.append(market['outcome'])

# This ignores within-market correlation!
brier = brier_score_loss(all_outcomes, all_predictions)

# CORRECT: Use panel data methods
import statsmodels.api as sm

# Create panel dataframe
panel_df = create_panel_structure(markets)

# Account for market-level clustering
model = sm.GEE.from_formula(
    "outcome ~ prediction + C(time_to_close)",
    data=panel_df,
    groups=panel_df['market_id'],
    cov_struct=sm.cov_struct.Exchangeable()
)
result = model.fit()
```

### Research Evidence

From **Good Judgment Project**:
- Aggregation weighted **recent opinions more strongly** than older ones
- This implies they sampled forecasters multiple times over a question's lifetime
- **Temporal recency weighting** improved calibration

From **Kalshi analysis**:
- Markets scored "on every day that it's open"
- Aggregated into relative score grading performance **over time**
- Brier scores calculated using consensus forecasts at **end of each time period**

### Practical Implementation for Hackathon

```python
def create_multi_sample_dataset(markets, sample_points):
    """
    Create dataset with multiple samples per market

    Args:
        markets: list of market dicts with price history
        sample_points: list of time-to-close values (in days)

    Returns:
        pandas DataFrame with panel structure
    """
    import pandas as pd

    rows = []

    for market in markets:
        market_id = market['ticker']
        outcome = market['actual_outcome']
        category = market.get('category', 'Unknown')

        for days_before in sample_points:
            price = get_price_at_time_before_close(
                market,
                days_before * 24 * 3600
            )

            if price is not None:
                rows.append({
                    'market_id': market_id,
                    'days_to_close': days_before,
                    'prediction': price,
                    'outcome': outcome,
                    'category': category,
                    'volume': market.get('volume', 0)
                })

    return pd.DataFrame(rows)

# Usage
sample_points = [30, 21, 14, 7, 3, 1, 0.04]  # 30 days to 1 hour
dataset = create_multi_sample_dataset(markets, sample_points)

# Analyze calibration by time-to-close
for days in sample_points:
    subset = dataset[dataset['days_to_close'] == days]
    brier = np.mean((subset['prediction'] - subset['outcome'])**2)
    print(f"{days} days before: Brier = {brier:.4f}, N = {len(subset)}")
```

**Output might look like**:
```
30 days before: Brier = 0.1523, N = 847
21 days before: Brier = 0.1402, N = 892
14 days before: Brier = 0.1256, N = 923
7 days before:  Brier = 0.1087, N = 981
3 days before:  Brier = 0.0945, N = 994
1 day before:   Brier = 0.0821, N = 998
1 hour before:  Brier = 0.0762, N = 1000
```

This shows **clear improvement** in calibration as resolution approaches!

---

## Solution 3: Change Point Detection

### Why Detect Change Points?

Change points identify **when markets dramatically update beliefs** due to:
- Major news events
- Information shocks
- Polls/data releases
- Unexpected developments

### Methods

#### Method 3A: CUSUM (Cumulative Sum)

**Best for**: Detecting single major shift in a market

```python
import numpy as np

def detect_changepoint_cusum(price_history, threshold=3):
    """
    Detect change point using CUSUM method

    Args:
        price_history: list of (timestamp, price) tuples
        threshold: sensitivity parameter (higher = less sensitive)

    Returns:
        (changepoint_index, changepoint_timestamp) or None
    """
    prices = np.array([p[1] for p in price_history])
    timestamps = np.array([p[0] for p in price_history])

    # Calculate mean
    mean_price = np.mean(prices)

    # Center the series
    centered = prices - mean_price

    # Cumulative sum
    cumsum = np.cumsum(centered)

    # Detect when CUSUM exceeds threshold * std
    std_price = np.std(prices)
    threshold_value = threshold * std_price

    # Find first point exceeding threshold
    exceeded = np.where(np.abs(cumsum) > threshold_value)[0]

    if len(exceeded) > 0:
        cp_index = exceeded[0]
        return cp_index, timestamps[cp_index]

    return None

# Example usage
market = markets[0]
changepoint = detect_changepoint_cusum(market['price_history'])

if changepoint:
    cp_index, cp_time = changepoint
    print(f"Change detected at {datetime.fromtimestamp(cp_time)}")
    print(f"Price before: {market['price_history'][cp_index-1][1]:.2f}")
    print(f"Price after: {market['price_history'][cp_index][1]:.2f}")
```

**Advantages**:
- ✅ Simple, fast, interpretable
- ✅ Good for online/real-time detection
- ✅ Works well for single major shift

**Disadvantages**:
- ❌ Only detects one change point
- ❌ Sensitive to threshold parameter
- ❌ Assumes normal distribution

---

#### Method 3B: PELT (Pruned Exact Linear Time)

**Best for**: Detecting multiple change points in complex markets

```python
import ruptures as rpt

def detect_multiple_changepoints_pelt(price_history, penalty=3):
    """
    Detect multiple change points using PELT algorithm

    Args:
        price_history: list of (timestamp, price) tuples
        penalty: controls number of change points (higher = fewer CPs)

    Returns:
        list of (index, timestamp, price_before, price_after) tuples
    """
    prices = np.array([p[1] for p in price_history])
    timestamps = np.array([p[0] for p in price_history])

    # PELT algorithm (l2 = squared error cost)
    algo = rpt.Pelt(model="l2", min_size=3).fit(prices)
    changepoints = algo.predict(pen=penalty)

    # Extract details
    cp_details = []
    for cp_idx in changepoints[:-1]:  # Last one is always end of series
        if cp_idx > 0 and cp_idx < len(prices):
            cp_details.append({
                'index': cp_idx,
                'timestamp': timestamps[cp_idx],
                'price_before': prices[cp_idx - 1],
                'price_after': prices[cp_idx],
                'magnitude': abs(prices[cp_idx] - prices[cp_idx - 1])
            })

    return cp_details

# Example usage
import ruptures as rpt
import matplotlib.pyplot as plt

market = markets[0]
changepoints = detect_multiple_changepoints_pelt(market['price_history'], penalty=10)

print(f"Detected {len(changepoints)} change points:")
for cp in changepoints:
    print(f"  Time: {datetime.fromtimestamp(cp['timestamp'])}")
    print(f"  Magnitude: {cp['magnitude']:.3f}")
    print(f"  Direction: {'↑' if cp['price_after'] > cp['price_before'] else '↓'}")
```

**Advantages**:
- ✅ Detects multiple change points
- ✅ Exact algorithm (not heuristic)
- ✅ Fast: O(n) complexity
- ✅ Works well with noisy data

**Disadvantages**:
- ❌ Requires tuning penalty parameter
- ❌ May detect false positives in volatile markets

---

#### Method 3C: Bayesian Change Point Detection

**Best for**: Principled uncertainty quantification

```python
import numpy as np
from scipy import stats

def bayesian_changepoint_detection(price_history, prior_changepoint_prob=0.01):
    """
    Bayesian online change point detection

    Args:
        price_history: list of (timestamp, price) tuples
        prior_changepoint_prob: prior probability of changepoint at each step

    Returns:
        Array of changepoint probabilities for each timestamp
    """
    prices = np.array([p[1] for p in price_history])
    n = len(prices)

    # Run length (time since last changepoint)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1

    changepoint_probs = np.zeros(n)

    # Hazard function (constant hazard)
    hazard = 1 / (1/prior_changepoint_prob)

    for t in range(1, n):
        # Evaluate predictive probability
        # (Simplified - full implementation requires more stats)

        # Growth probabilities
        R[1:t+1, t] = R[0:t, t-1] * (1 - hazard)

        # Changepoint probability
        R[0, t] = np.sum(R[0:t, t-1] * hazard)

        # Normalize
        R[:, t] = R[:, t] / np.sum(R[:, t])

        # Probability of changepoint at time t
        changepoint_probs[t] = R[0, t]

    return changepoint_probs
```

---

### Using Change Points in Calibration Analysis

#### Application 1: Recency Bias Detection

**Hypothesis**: Markets overreact to change points, then mean-revert

```python
def analyze_post_changepoint_calibration(markets_with_cps):
    """
    Compare calibration of predictions made right after change points
    vs predictions made during stable periods
    """

    post_cp_predictions = []
    stable_predictions = []

    for market in markets_with_cps:
        changepoints = market['changepoints']

        for cp in changepoints:
            # Get prediction within 24h after changepoint
            post_cp_price = get_price_after(market, cp['timestamp'], hours=24)
            post_cp_predictions.append({
                'prediction': post_cp_price,
                'outcome': market['outcome']
            })

        # Get predictions during stable periods (>72h from any CP)
        stable_prices = get_stable_period_prices(market, changepoints, min_hours=72)
        for price in stable_prices:
            stable_predictions.append({
                'prediction': price,
                'outcome': market['outcome']
            })

    # Compare calibration
    post_cp_brier = calculate_brier_score(
        [p['prediction'] for p in post_cp_predictions],
        [p['outcome'] for p in post_cp_predictions]
    )

    stable_brier = calculate_brier_score(
        [p['prediction'] for p in stable_predictions],
        [p['outcome'] for p in stable_predictions]
    )

    print(f"Post-changepoint Brier: {post_cp_brier:.4f}")
    print(f"Stable period Brier: {stable_brier:.4f}")

    if post_cp_brier > stable_brier:
        print("⚠️ Markets less calibrated after major news (recency bias!)")
```

#### Application 2: Event-Driven Analysis

Identify what types of events cause change points:

```python
def categorize_changepoints(market, external_events):
    """
    Match change points to external events

    Args:
        market: market with detected changepoints
        external_events: list of (timestamp, event_description) tuples

    Returns:
        Changepoints annotated with likely causes
    """
    annotated_cps = []

    for cp in market['changepoints']:
        cp_time = cp['timestamp']

        # Find events within ±24 hours
        nearby_events = [
            e for e in external_events
            if abs(e[0] - cp_time) < 24*3600
        ]

        annotated_cps.append({
            **cp,
            'likely_cause': nearby_events[0][1] if nearby_events else 'Unknown',
            'has_known_cause': len(nearby_events) > 0
        })

    return annotated_cps
```

---

## Practical Recommendations for Hackathon

### Minimum Viable Analysis (Day 2-3)

If time is limited, prioritize:

**1. Time-to-Resolution Normalization** (3 hours)
```python
# Sample at fixed windows
WINDOWS = [30, 7, 1, 0.04]  # days before close
dataset = create_multi_sample_dataset(markets, WINDOWS)

# Plot calibration over time
for window in WINDOWS:
    subset = dataset[dataset['days_to_close'] == window]
    brier = calculate_brier_score(subset['prediction'], subset['outcome'])
    plot_calibration_curve(subset['prediction'], subset['outcome'],
                          title=f"Calibration at {window} days before close")
```

**2. Simple Change Point Detection** (2 hours)
```python
# Use PELT on high-volume markets
high_vol_markets = [m for m in markets if m['volume'] > threshold]

for market in high_vol_markets[:20]:  # Sample 20 markets
    cps = detect_multiple_changepoints_pelt(market['price_history'])
    if len(cps) > 0:
        print(f"{market['title']}: {len(cps)} change points detected")
```

### Advanced Analysis (If Time Permits)

**3. Panel Data Regression** (4 hours)
```python
import statsmodels.formula.api as smf

# Multi-level model accounting for market clustering
model = smf.mixedlm(
    "brier_score ~ days_to_close + C(category) + log(volume)",
    data=dataset,
    groups=dataset["market_id"]
)
results = model.fit()
print(results.summary())
```

**4. Changepoint-Driven Recency Bias Analysis** (3 hours)
- Detect changepoints in all markets
- Compare calibration post-changepoint vs stable periods
- Visualize magnitude of overreaction

---

## Summary Table: When to Use Each Approach

| Method | Best For | Time Required | Insights |
|--------|----------|---------------|----------|
| **Fixed Time Windows** | Quick analysis, testing temporal hypotheses | 2-3 hours | "Calibration improves near resolution" |
| **Percentage Normalization** | Markets with vastly different durations | 3-4 hours | "Early predictions vs late predictions" |
| **Multi-Sample Panel** | Maximum statistical power | 4-5 hours | "Accounting for within-market correlation" |
| **CUSUM** | Single major shift detection | 1-2 hours | "Did this market react to news?" |
| **PELT** | Complex markets with multiple events | 2-3 hours | "How many times did beliefs shift?" |
| **Bayesian CP** | Uncertainty quantification | 4-6 hours | "How confident are we in changepoints?" |

---

## Code Implementation Priority

### Must Have (Day 2)
1. ✅ Fixed time window sampling
2. ✅ Multi-sample dataset creation
3. ✅ Calibration calculation at each window

### Should Have (Day 3)
4. ✅ PELT changepoint detection
5. ✅ Percentage normalization (for short markets)
6. ✅ Basic visualization of calibration over time

### Nice to Have (If ahead of schedule)
7. ⭐ Panel data regression with clustering
8. ⭐ Changepoint-categorization by event type
9. ⭐ Recency bias analysis post-changepoint

---

## Key Takeaways

1. **Time normalization is essential** - use "days to close" not calendar dates
2. **Multi-sampling is valid and powerful** - each time point is a different forecast
3. **Account for autocorrelation** - use panel data methods, not naive aggregation
4. **Change points reveal market psychology** - overreaction patterns show recency bias
5. **Start simple, add complexity** - fixed windows first, then advanced methods
6. **Research shows** - calibration improves near resolution (consistent across studies)

**Good luck with your timeseries analysis!**
