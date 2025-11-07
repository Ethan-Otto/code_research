# Prediction Markets Calibration Analysis

A research project for analyzing the forecasting calibration of betting markets (Kalshi and Polymarket) to identify systematic biases and error rates across different market categories.

## Overview

This project measures how well prediction markets are calibrated - that is, whether events predicted to occur with X% probability actually occur X% of the time. The analysis investigates:

- Overall calibration quality of each platform
- Systematic biases (favorite-longshot, optimism, recency)
- Category-specific accuracy (politics, sports, economics, etc.)
- Volume and liquidity effects on calibration
- Time-dependent calibration changes

## Project Files

- **`prediction_markets_calibration_plan.md`**: Comprehensive research plan with methodology, metrics, and analysis strategy
- **`api_implementation_examples.py`**: Python code for fetching data from Kalshi and Polymarket APIs
- **`visualization_examples.py`**: Visualization functions for calibration plots and analysis
- **`requirements.txt`**: Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

```python
from api_implementation_examples import KalshiClient, PolymarketClient

# Fetch Kalshi data
kalshi = KalshiClient()
kalshi_markets = kalshi.get_all_settled_markets(max_markets=1000)

# Fetch Polymarket data
polymarket = PolymarketClient()
poly_markets = polymarket.get_all_closed_markets(max_markets=1000)
```

### 3. Analyze Calibration

```python
from api_implementation_examples import calculate_brier_score, calculate_ece

# Extract predictions and outcomes
predictions = [m["predicted_probability"] for m in kalshi_markets]
outcomes = [m["actual_outcome"] for m in kalshi_markets]

# Calculate metrics
brier = calculate_brier_score(predictions, outcomes)
ece = calculate_ece(predictions, outcomes)

print(f"Brier Score: {brier:.4f}")
print(f"Expected Calibration Error: {ece:.4f}")
```

### 4. Visualize Results

```python
from visualization_examples import plot_calibration_curve, plot_favorite_longshot_bias

# Create calibration plot
plot_calibration_curve(predictions, outcomes, title="Kalshi Calibration")

# Analyze favorite-longshot bias
plot_favorite_longshot_bias(predictions, outcomes)
```

## Key Metrics

### Brier Score
Measures probabilistic forecast accuracy. Lower is better (0 = perfect, 1 = worst).

```
Brier Score = (1/N) Σ (forecast_probability - actual_outcome)²
```

### Expected Calibration Error (ECE)
Quantifies deviation from perfect calibration across probability bins.

### Calibration Plot (Reliability Diagram)
Visual assessment comparing predicted probabilities to observed frequencies. Perfect calibration follows the diagonal line (y = x).

## API Documentation

### Kalshi API

- **Base URL**: `https://api.elections.kalshi.com/trade-api/v2`
- **Key Endpoint**: `GET /markets`
- **Filters**: status, min_close_ts, max_close_ts, event_ticker
- **Authentication**: Public endpoints available without API key

### Polymarket API

- **Gamma API** (market data): `https://gamma-api.polymarket.com`
- **CLOB API** (price history): `https://clob.polymarket.com`
- **Key Endpoints**:
  - `GET /markets` - All markets
  - `GET /events` - Events (more efficient)
  - `GET /prices-history` - Historical prices

## Research Questions

1. **Calibration Quality**: How well-calibrated are prediction markets on each platform?
2. **Systematic Bias**: Are certain categories systematically over/underconfident?
3. **Platform Comparison**: Do Kalshi and Polymarket differ in calibration?
4. **Time Effects**: Does calibration improve as markets approach resolution?
5. **Volume Effects**: Do higher-volume markets show better calibration?

## Market Categories

- **Politics**: Elections, policy outcomes, approval ratings
- **Sports**: Team sports, individual competitions, championships
- **Economics**: GDP, unemployment, stock milestones, crypto
- **Weather**: Temperature records, storms, climate events
- **Entertainment**: Awards, box office, celebrity events
- **Science/Tech**: Product releases, discoveries, AI milestones
- **Current Events**: News, geopolitics, public health

## Known Systematic Biases

### Favorite-Longshot Bias
Markets tend to:
- Overvalue low-probability events (<20%)
- Undervalue high-probability events (>80%)

### Optimism Bias
Markets systematically overestimate positive outcomes

### Recency Bias
Over-weighting recent information leads to overshooting

### Volume Effects
Low-volume markets show worse calibration than high-volume markets

## Hackathon Timeline

### Day 1: Data Collection
- Set up API clients
- Fetch 1000+ settled markets from each platform
- Validate and clean data
- Implement categorization

### Day 2: Analysis
- Calculate calibration metrics (Brier, ECE)
- Generate calibration plots
- Test for systematic biases
- Category-specific analysis

### Day 3: Visualization & Reporting
- Create comprehensive visualizations
- Build comparison dashboards
- Statistical significance testing
- Prepare final presentation

## Expected Findings (Hypotheses)

1. Politics markets may show optimism bias toward challengers
2. Sports markets likely exhibit favorite-longshot bias
3. High-volume markets should have better calibration
4. Calibration improves in final week before resolution
5. Kalshi may show better calibration (regulatory oversight)
6. Polymarket may have deeper liquidity in crypto markets

## References

### Academic Papers
- "Prediction Markets for Economic Forecasting" (Brookings, NBER)
- "Stable reliability diagrams for probabilistic classifiers" (PNAS 2021)

### Existing Research
- CW Data Solutions: Kalshi calibration analysis
- Alex McCullough: Polymarket accuracy study (90% reported)
- Various favorite-longshot bias studies in sports betting

### Tools & Libraries
- scikit-learn calibration module: https://scikit-learn.org/stable/modules/calibration.html
- Brier score reference: https://en.wikipedia.org/wiki/Brier_score

## Data Structure

### Kalshi Market Data
```python
{
    "ticker": "MARKET-TICKER",
    "title": "Will event occur?",
    "last_price": 65,  # cents (0-100)
    "result": "yes",  # or "no"
    "volume": 10000,
    "close_time": "2025-01-01T00:00:00Z",
    "predicted_probability": 0.65,
    "actual_outcome": 1
}
```

### Polymarket Market Data
```python
{
    "id": "market-id",
    "question": "Will event occur?",
    "outcomes": ["Yes", "No"],
    "outcomePrices": [0.65, 0.35],
    "closed": true,
    "volume": 50000,
    "predicted_probability": 0.65,
    "actual_outcome": 1  # (needs resolution data)
}
```

## Tips for Success

1. **Start Early**: API rate limits may slow data collection
2. **Cache Data**: Save raw responses to avoid re-fetching
3. **Validate Quality**: Check for missing outcomes, invalid prices
4. **Automate**: Use existing libraries (scikit-learn) for standard metrics
5. **Focus**: Prioritize core analyses over stretch goals
6. **Document**: Keep notes on findings and assumptions

## Troubleshooting

### API Rate Limits
- Implement exponential backoff (2s, 4s, 8s, 16s)
- Add sleep between requests (0.5-1s)
- Cache responses locally

### Missing Resolution Data
- For Polymarket, may need to check on-chain data via UMA Oracle
- Filter to markets with clear binary outcomes
- Cross-reference with news sources if needed

### Small Sample Sizes
- Combine related categories
- Report confidence intervals
- Focus on categories with sufficient data

### Categorization Challenges
- Use keyword matching on titles
- Manual review of edge cases
- Multi-label classification when ambiguous

## Next Steps

After the hackathon, consider:

1. **Expand Dataset**: Collect more historical data
2. **Real-Time Tracking**: Monitor ongoing markets
3. **Predictive Modeling**: Build calibrated forecasting models
4. **Trading Strategy**: Exploit systematic biases
5. **Academic Paper**: Publish comprehensive findings
6. **Interactive Dashboard**: Deploy web-based exploration tool

## License

This project is for educational and research purposes.

---

**Good luck with your analysis!** 🎲📊
