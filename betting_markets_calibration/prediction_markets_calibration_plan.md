# Prediction Markets Calibration Analysis - Hackathon Project Plan

## Executive Summary

This project aims to measure the forecasting calibration of betting markets (Kalshi and Polymarket) to identify systematic biases and error rates across different market categories. The analysis will use historical price data, volume, and resolution outcomes to evaluate whether certain market types consistently deviate from perfect calibration.

## Research Questions

1. **Calibration Quality**: How well-calibrated are prediction markets on each platform?
2. **Systematic Bias Detection**: Are certain market categories (sports, politics, economics) systematically over/underconfident?
3. **Platform Comparison**: Do Kalshi and Polymarket exhibit different calibration characteristics?
4. **Time-Dependent Effects**: Does calibration improve or degrade as markets approach their resolution date?
5. **Volume Correlation**: Do higher-volume markets show better calibration than low-volume markets?

---

## Data Collection Strategy

### Kalshi API Access

**Base URL**: `https://api.elections.kalshi.com/trade-api/v2`

**Key Endpoints**:
- `GET /markets` - Retrieve market data with filtering options
  - Filter by `status`: 'unopened', 'open', 'closed', 'settled'
  - Use `min_close_ts` and `max_close_ts` for date ranges
  - Pagination: `limit` (max 1000), `cursor` for subsequent pages

**Critical Data Fields**:
- **Market Info**: ticker, event_ticker, title, market_type
- **Pricing**: yes_bid, yes_ask, no_bid, no_ask, last_price
- **Timing**: open_time, close_time, expiration_time
- **Volume**: volume, volume_24h, open_interest, liquidity
- **Resolution**: result (yes/no), settlement_value, status

**Historical Price Data**:
- Candlestick/OHLC data available via market history endpoints
- Provides time-series price evolution

**Authentication**:
- Public endpoints available without API keys
- May need authentication for rate limit increases

**Data Collection Approach**:
1. Fetch all settled markets: `GET /markets?status=settled&limit=1000`
2. Iterate through pages using cursor pagination
3. For markets of interest, retrieve historical price data
4. Store: final price before resolution, actual outcome, market category, volume

---

### Polymarket API Access

**Base URL**: `https://gamma-api.polymarket.com` (Gamma Markets API)

**Key Endpoints**:
- `GET /markets` - Retrieve all markets with resolution data
- `GET /events` - Most efficient for retrieving active markets
- `GET /markets/slug/{slug}` - Individual market by slug
- `GET /prices-history` (CLOB API: `https://clob.polymarket.com/`)

**Historical Price Data** (CLOB API):
```
GET /prices-history
Parameters:
- market (required): CLOB token ID
- startTs, endTs: Unix timestamps for time range
- interval: '1m', '1w', '1d', '6h', '1h', 'max'
- fidelity: Resolution in minutes
```

**Response Format**:
```json
{
  "history": [
    {"t": timestamp, "p": price},
    ...
  ]
}
```

**Critical Data Fields**:
- **Market Info**: id, question, conditionId, slug, outcomes
- **Prices**: outcomePrices (array for each outcome)
- **Status**: active, closed, closedTime
- **Resolution**: resolutionSource, resolution data (from UMA Oracle)
- **Volume/Liquidity**: volume, liquidity metrics
- **Timing**: startDate, endDate

**Data Collection Approach**:
1. Fetch resolved markets via Gamma API
2. For each market, get historical prices from CLOB API
3. Extract final price distributions and actual outcomes
4. Store with market metadata and category information

---

## Calibration Methodology

### 1. **Brier Score**

The primary metric for measuring probabilistic forecast accuracy.

**Formula**:
```
Brier Score = (1/N) Σ (forecast_probability - actual_outcome)²
```

Where:
- forecast_probability: Market price (0-1) before resolution
- actual_outcome: 1 if event occurred, 0 otherwise
- Lower scores = better calibration (0 = perfect, 1 = worst)

**Decomposition**:
```
Brier Score = Reliability - Resolution + Uncertainty
```
- **Reliability (Calibration)**: Measures systematic bias
- **Resolution**: Measures ability to distinguish outcomes
- **Uncertainty**: Baseline difficulty of prediction task

### 2. **Calibration Plots (Reliability Diagrams)**

Visual assessment of calibration quality.

**Traditional Binning Approach**:
1. Divide predictions into bins (e.g., 0-10%, 10-20%, ..., 90-100%)
2. For each bin, calculate:
   - Mean predicted probability
   - Observed frequency of positive outcomes
3. Plot observed frequency vs. predicted probability
4. Perfect calibration = diagonal line (y = x)

**Bins**: Recommend 10-20 bins, adjustable based on sample size

**Advanced Approach**:
- Use **CORP (Consistent, Optimally binned, Reproducible)** method
- Employs isotonic regression via Pool-Adjacent-Violators (PAV) algorithm
- Provides statistically consistent binning without arbitrary choices

### 3. **Expected Calibration Error (ECE)**

Quantitative measure of calibration quality.

**Formula**:
```
ECE = Σ (n_b / N) |observed_freq_b - predicted_prob_b|
```

Where:
- n_b: Number of predictions in bin b
- N: Total number of predictions
- observed_freq_b: Proportion of positive outcomes in bin b
- predicted_prob_b: Average predicted probability in bin b

### 4. **Brier Skill Score (BSS)**

Measures improvement over a reference forecast (typically the base rate).

**Formula**:
```
BSS = 1 - (Brier Score_model / Brier Score_reference)
```

**Interpretation**:
- BSS > 0: Model beats reference
- BSS = 0: Model equals reference
- BSS < 0: Model worse than reference

**Time-Series Analysis**: Track BSS as markets approach resolution to measure information accumulation

---

## Market Categorization

### Primary Categories

Based on existing prediction market taxonomies:

1. **Politics**
   - Elections (presidential, congressional, local)
   - Policy outcomes (legislation passage)
   - Approval ratings
   - International relations

2. **Sports**
   - Team sports (NFL, NBA, MLB, NHL, soccer)
   - Individual sports (tennis, golf, boxing)
   - Championships and tournaments
   - Player performance props

3. **Economics/Finance**
   - Economic indicators (GDP, unemployment, inflation)
   - Stock market milestones
   - Cryptocurrency prices
   - Interest rate decisions

4. **Weather/Climate**
   - Temperature records
   - Precipitation events
   - Storm occurrences
   - Climate milestones

5. **Entertainment/Pop Culture**
   - Award shows (Oscars, Grammys)
   - TV show outcomes
   - Celebrity events
   - Box office performance

6. **Science/Technology**
   - Product releases
   - Scientific discoveries
   - Space missions
   - AI milestones

7. **Current Events**
   - News events
   - Geopolitical developments
   - Public health

### Secondary Attributes

Additional categorization dimensions for deeper analysis:

- **Time Horizon**: Short-term (<1 week), Medium (1 week - 1 month), Long-term (>1 month)
- **Market Type**: Binary (yes/no) vs. Scalar/Range markets
- **Volume Tier**: High (top quartile), Medium, Low
- **Liquidity Level**: Based on bid-ask spread
- **Market Maturity**: Days between market creation and resolution

### Categorization Implementation

**Kalshi**:
- Use `event_ticker` and `series_ticker` for grouping
- Parse market titles for category keywords
- May have built-in category tags

**Polymarket**:
- Use tags/categories if available in API
- Parse question text and slug
- Group by `resolutionSource` patterns

---

## Known Systematic Biases to Investigate

### 1. **Favorite-Longshot Bias**

**Definition**: Tendency to overvalue low-probability events (longshots) and undervalue high-probability events (favorites)

**Hypothesis**:
- Predictions < 20% may be systematically too high
- Predictions > 80% may be systematically too low

**Analysis**: Create calibration plots specifically for extreme probabilities

### 2. **Optimism Bias**

**Definition**: Markets systematically overestimate likelihood of positive outcomes

**Hypothesis**:
- Sports home team advantages over-estimated
- Positive economic outcomes over-predicted
- Political challenger chances inflated

**Analysis**: Tag "optimistic" vs. "pessimistic" framing and compare calibration

### 3. **Recency Bias**

**Definition**: Over-weighting recent information when updating beliefs

**Hypothesis**: Markets overreact to recent news, leading to overshooting

**Analysis**:
- Track price volatility near resolution
- Compare markets with late-breaking news to stable markets

### 4. **Partition Dependence**

**Definition**: Probability assignments affected by how outcomes are categorized

**Hypothesis**: Multi-outcome markets show different calibration than binary markets

**Analysis**: Compare binary market calibration to scalar/range markets

### 5. **Volume/Liquidity Effects**

**Definition**: Low-volume markets have worse calibration due to fewer informed traders

**Hypothesis**:
- High-volume markets exhibit better calibration
- Low-volume markets show higher variance and bias

**Analysis**: Stratify calibration metrics by volume quartiles

### 6. **Time-to-Resolution Effects**

**Definition**: Calibration improves as resolution approaches (information accumulation)

**Hypothesis**:
- Early market prices less calibrated
- Calibration improves exponentially near resolution

**Analysis**:
- Sample prices at multiple time points (e.g., 30 days, 7 days, 1 day before)
- Plot calibration metrics vs. time-to-resolution

---

## Analysis Pipeline

### Phase 1: Data Collection (Days 1-2)

1. **Setup**
   - Create API client functions for both platforms
   - Implement pagination and rate limiting
   - Set up data storage (SQLite, CSV, or JSON)

2. **Historical Data Fetch**
   - Retrieve all settled markets from both platforms
   - For each market, get:
     - Final price snapshot (close to resolution)
     - Historical price series
     - Volume/liquidity metrics
     - Actual outcome
     - Market metadata

3. **Data Validation**
   - Check for missing values
   - Verify outcome coding (0/1 for binary)
   - Ensure price ranges are valid (0-1)

**Target**: 1000+ resolved markets per platform

### Phase 2: Categorization (Day 2)

1. **Automated Categorization**
   - Keyword matching on titles/questions
   - Pattern recognition for tickers/slugs
   - Manual review of ambiguous cases

2. **Feature Engineering**
   - Calculate time-to-resolution
   - Determine volume tier
   - Extract market characteristics

### Phase 3: Calibration Analysis (Days 2-3)

1. **Overall Calibration**
   - Calculate aggregate Brier Score for each platform
   - Generate calibration plots
   - Compute ECE

2. **Category-Specific Analysis**
   - Repeat metrics for each category
   - Compare across categories
   - Statistical significance testing (t-tests, ANOVA)

3. **Bias Detection**
   - Test for favorite-longshot bias (extreme probability bins)
   - Analyze by market attributes (volume, time horizon)
   - Temporal analysis (price evolution)

4. **Comparative Analysis**
   - Kalshi vs. Polymarket head-to-head
   - Identify relative strengths/weaknesses
   - Volume-adjusted comparisons

### Phase 4: Visualization & Reporting (Day 3)

1. **Visualizations**
   - Calibration plots by category
   - Brier Score distributions
   - Bias detection charts
   - Time-series calibration evolution
   - Platform comparison dashboards

2. **Statistical Summary**
   - Summary tables by category
   - Confidence intervals for metrics
   - Hypothesis test results

3. **Insights & Recommendations**
   - Key findings
   - Actionable insights for traders
   - Suggestions for platform improvements

---

## Technical Stack Recommendations

### Languages & Libraries

**Python** (recommended):
- `requests` or `httpx`: API calls
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`/`seaborn`: Visualizations
- `scikit-learn`: Calibration metrics and plots (has built-in calibration_curve)
- `scipy`: Statistical tests
- `sqlite3` or `sqlalchemy`: Data storage

**JavaScript/TypeScript** (alternative):
- `axios`: API calls
- `d3.js`: Visualizations
- `@nivo/core`: React-based charts

### Data Storage

- **SQLite**: Local, lightweight, good for <1M records
- **PostgreSQL**: If scaling to larger datasets
- **CSV/Parquet**: Simple flat file storage

### Deployment

- **Jupyter Notebooks**: Exploratory analysis and visualization
- **Streamlit/Dash**: Interactive dashboards
- **GitHub Pages**: Static report hosting

---

## Implementation Checklist

### Pre-Hackathon Setup
- [ ] Register for Kalshi API access (if authentication required)
- [ ] Test API endpoints for both platforms
- [ ] Set up development environment with required libraries
- [ ] Create data schemas/database structure

### Hackathon Day 1
- [ ] Implement Kalshi data fetcher
- [ ] Implement Polymarket data fetcher
- [ ] Collect 500+ settled markets from each platform
- [ ] Validate data quality

### Hackathon Day 2
- [ ] Complete data collection (1000+ markets each)
- [ ] Implement categorization pipeline
- [ ] Calculate basic calibration metrics (Brier Score, ECE)
- [ ] Generate initial calibration plots

### Hackathon Day 3
- [ ] Complete category-specific analyses
- [ ] Test for systematic biases
- [ ] Create visualizations and dashboards
- [ ] Prepare final presentation/report
- [ ] Document findings and code

---

## Expected Outcomes

### Deliverables

1. **Dataset**: Cleaned, categorized historical market data
2. **Analysis Scripts**: Reusable code for calibration analysis
3. **Visualizations**: Calibration plots, bias detection charts
4. **Report**: Findings on systematic biases by category and platform
5. **Dashboard** (stretch goal): Interactive tool for exploring calibration

### Key Metrics to Report

- Overall Brier Score by platform
- ECE by category
- Favorite-longshot bias magnitude
- Calibration improvement rate over time
- Volume-calibration correlation
- Platform ranking by category

### Potential Findings (Hypotheses)

1. Politics markets may show optimism bias toward challengers
2. Sports markets likely exhibit favorite-longshot bias
3. High-volume markets should have better calibration
4. Calibration improves significantly in final week before resolution
5. Kalshi may show better calibration due to regulatory oversight
6. Polymarket may have deeper liquidity in crypto-related markets

---

## Additional Resources

### Academic Papers
- "Prediction Markets for Economic Forecasting" (Brookings, NBER)
- "Stable reliability diagrams for probabilistic classifiers" (PNAS 2021)
- Studies on favorite-longshot bias in betting markets

### Existing Analyses
- CW Data Solutions: "Calibration and Skill of the Kalshi Prediction Markets"
- Alex McCullough's Polymarket accuracy study (90% reported)

### Tools & Libraries
- scikit-learn calibration module: https://scikit-learn.org/stable/modules/calibration.html
- Reliability diagram implementations: https://github.com/topics/calibration-plot

---

## Risk Mitigation

### Potential Issues

1. **API Rate Limits**:
   - Solution: Implement exponential backoff, respect rate limits, cache data

2. **Insufficient Historical Data**:
   - Solution: Focus on recent markets (2024-2025), combine platforms if needed

3. **Missing Resolution Data**:
   - Solution: Cross-reference with external sources, filter to clean subset

4. **Category Ambiguity**:
   - Solution: Multi-label classification, manual review of edge cases

5. **Small Sample Sizes in Subcategories**:
   - Solution: Aggregate related categories, report confidence intervals

6. **Time Constraints**:
   - Solution: Prioritize core analyses, automate repetitive tasks, use existing libraries

---

## Conclusion

This project offers a rigorous, data-driven approach to evaluating prediction market calibration. By combining comprehensive data collection from both major platforms (Kalshi and Polymarket) with established statistical methodologies (Brier Score, calibration plots, ECE), the analysis will provide valuable insights into:

1. Which market categories are most/least reliable for forecasting
2. Whether systematic biases (favorite-longshot, optimism, etc.) exist
3. How the two platforms compare in prediction accuracy
4. Practical guidance for traders and market designers

The findings could inform trading strategies, market design improvements, and broader understanding of collective forecasting accuracy in decentralized prediction markets.

**Good luck at the hackathon!**
