"""
Prediction Markets Data Collection - API Implementation Examples
For Kalshi and Polymarket calibration analysis

This module provides example code for fetching historical market data
from both platforms.
"""

import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json


# =============================================================================
# KALSHI API CLIENT
# =============================================================================

class KalshiClient:
    """Client for fetching market data from Kalshi API"""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Kalshi client

        Args:
            api_key: Optional API key for authenticated requests
                    (public endpoints work without authentication)
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_markets(
        self,
        status: str = "settled",
        limit: int = 1000,
        cursor: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None
    ) -> Dict:
        """
        Fetch markets from Kalshi API

        Args:
            status: Market status filter ('unopened', 'open', 'closed', 'settled')
            limit: Number of results per page (max 1000)
            cursor: Pagination cursor
            min_close_ts: Minimum close timestamp (Unix)
            max_close_ts: Maximum close timestamp (Unix)
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker

        Returns:
            API response with markets data
        """
        params = {
            "status": status,
            "limit": limit
        }

        if cursor:
            params["cursor"] = cursor
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker

        response = self.session.get(f"{self.BASE_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    def get_all_settled_markets(
        self,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        max_markets: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch all settled markets using pagination

        Args:
            min_close_ts: Minimum close timestamp (Unix)
            max_close_ts: Maximum close timestamp (Unix)
            max_markets: Maximum number of markets to fetch

        Returns:
            List of all settled markets
        """
        all_markets = []
        cursor = None

        while True:
            response = self.get_markets(
                status="settled",
                cursor=cursor,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts
            )

            markets = response.get("markets", [])
            all_markets.extend(markets)

            print(f"Fetched {len(markets)} markets (total: {len(all_markets)})")

            # Check if we've reached the max
            if max_markets and len(all_markets) >= max_markets:
                all_markets = all_markets[:max_markets]
                break

            # Check for next page
            cursor = response.get("cursor")
            if not cursor:
                break

            # Rate limiting - be nice to the API
            time.sleep(0.5)

        return all_markets

    def extract_market_data(self, market: Dict) -> Dict:
        """
        Extract relevant fields from market data for calibration analysis

        Args:
            market: Raw market data from API

        Returns:
            Cleaned market data with key fields
        """
        return {
            # Identifiers
            "ticker": market.get("ticker"),
            "event_ticker": market.get("event_ticker"),
            "title": market.get("title"),

            # Market type and question
            "market_type": market.get("market_type"),
            "subtitle": market.get("subtitle"),

            # Pricing (convert cents to probabilities if needed)
            "last_price": market.get("last_price"),  # In cents (0-100)
            "yes_ask": market.get("yes_ask"),
            "yes_bid": market.get("yes_bid"),

            # Resolution
            "result": market.get("result"),  # 'yes', 'no', or empty
            "status": market.get("status"),

            # Volume and liquidity
            "volume": market.get("volume"),
            "volume_24h": market.get("volume_24h"),
            "open_interest": market.get("open_interest"),
            "liquidity": market.get("liquidity"),

            # Timing
            "open_time": market.get("open_time"),
            "close_time": market.get("close_time"),
            "expiration_time": market.get("expiration_time"),

            # For probability conversion
            "predicted_probability": market.get("last_price", 0) / 100.0 if market.get("last_price") else None,
            "actual_outcome": 1 if market.get("result") == "yes" else 0 if market.get("result") == "no" else None
        }


# =============================================================================
# POLYMARKET API CLIENT
# =============================================================================

class PolymarketClient:
    """Client for fetching market data from Polymarket APIs"""

    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
    CLOB_BASE_URL = "https://clob.polymarket.com"

    def __init__(self):
        """Initialize Polymarket client"""
        self.session = requests.Session()

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: Optional[bool] = None,
        order: str = "id",
        ascending: bool = False
    ) -> List[Dict]:
        """
        Fetch markets from Polymarket Gamma API

        Args:
            limit: Number of results per page
            offset: Offset for pagination
            closed: Filter for closed markets (True/False/None)
            order: Field to order by
            ascending: Sort order

        Returns:
            List of markets
        """
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower()
        }

        if closed is not None:
            params["closed"] = str(closed).lower()

        response = self.session.get(f"{self.GAMMA_BASE_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: Optional[bool] = None,
        order: str = "id",
        ascending: bool = False
    ) -> List[Dict]:
        """
        Fetch events (which contain markets) from Polymarket Gamma API
        This is often more efficient than fetching markets directly

        Args:
            limit: Number of results per page
            offset: Offset for pagination
            closed: Filter for closed events (True/False/None)
            order: Field to order by
            ascending: Sort order

        Returns:
            List of events
        """
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower()
        }

        if closed is not None:
            params["closed"] = str(closed).lower()

        response = self.session.get(f"{self.GAMMA_BASE_URL}/events", params=params)
        response.raise_for_status()
        return response.json()

    def get_market_by_slug(self, slug: str) -> Dict:
        """
        Fetch a specific market by its slug

        Args:
            slug: Market slug identifier

        Returns:
            Market data
        """
        response = self.session.get(f"{self.GAMMA_BASE_URL}/markets/{slug}")
        response.raise_for_status()
        return response.json()

    def get_price_history(
        self,
        market: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: Optional[str] = None,
        fidelity: Optional[int] = None
    ) -> Dict:
        """
        Fetch historical price data for a market from CLOB API

        Args:
            market: CLOB token ID (from market data)
            start_ts: Start timestamp (Unix, UTC)
            end_ts: End timestamp (Unix, UTC)
            interval: Duration string ('1m', '1w', '1d', '6h', '1h', 'max')
                     Mutually exclusive with start_ts/end_ts
            fidelity: Resolution in minutes

        Returns:
            Historical price data
        """
        params = {"market": market}

        if interval:
            params["interval"] = interval
        else:
            if start_ts:
                params["startTs"] = start_ts
            if end_ts:
                params["endTs"] = end_ts

        if fidelity:
            params["fidelity"] = fidelity

        response = self.session.get(f"{self.CLOB_BASE_URL}/prices-history", params=params)
        response.raise_for_status()
        return response.json()

    def get_all_closed_markets(self, max_markets: Optional[int] = None) -> List[Dict]:
        """
        Fetch all closed markets using pagination

        Args:
            max_markets: Maximum number of markets to fetch

        Returns:
            List of closed markets
        """
        all_markets = []
        offset = 0
        limit = 100

        while True:
            markets = self.get_markets(
                limit=limit,
                offset=offset,
                closed=True
            )

            if not markets:
                break

            all_markets.extend(markets)
            print(f"Fetched {len(markets)} markets (total: {len(all_markets)})")

            if max_markets and len(all_markets) >= max_markets:
                all_markets = all_markets[:max_markets]
                break

            if len(markets) < limit:
                # No more markets available
                break

            offset += limit
            time.sleep(0.5)  # Rate limiting

        return all_markets

    def extract_market_data(self, market: Dict) -> Dict:
        """
        Extract relevant fields from Polymarket data for calibration analysis

        Args:
            market: Raw market data from API

        Returns:
            Cleaned market data with key fields
        """
        # For binary markets, typically have two outcomes
        outcomes = market.get("outcomes", [])
        outcome_prices = market.get("outcomePrices", [])

        # Assuming first outcome is "Yes" for binary markets
        predicted_prob = float(outcome_prices[0]) if outcome_prices else None

        return {
            # Identifiers
            "id": market.get("id"),
            "slug": market.get("slug"),
            "condition_id": market.get("conditionId"),

            # Question
            "question": market.get("question"),

            # Outcomes
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,

            # Status
            "active": market.get("active"),
            "closed": market.get("closed"),
            "closed_time": market.get("closedTime"),

            # Resolution
            "resolution_source": market.get("resolutionSource"),
            # Note: Actual resolution outcome needs to be determined from
            # on-chain data or additional API calls

            # Timing
            "start_date": market.get("startDate"),
            "end_date": market.get("endDate"),

            # Volume/liquidity (if available)
            "volume": market.get("volume"),
            "liquidity": market.get("liquidity"),

            # For calibration analysis
            "predicted_probability": predicted_prob,
            # actual_outcome needs to be determined from resolution data
        }


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def calculate_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Calculate Brier Score for probabilistic predictions

    Args:
        predictions: List of predicted probabilities (0-1)
        outcomes: List of actual outcomes (0 or 1)

    Returns:
        Brier Score (lower is better, 0 = perfect)
    """
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")

    squared_errors = [(pred - out) ** 2 for pred, out in zip(predictions, outcomes)]
    return sum(squared_errors) / len(squared_errors)


def calculate_calibration_data(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> Dict:
    """
    Calculate data for calibration plot

    Args:
        predictions: List of predicted probabilities (0-1)
        outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of bins for grouping predictions

    Returns:
        Dictionary with bin data for plotting
    """
    import numpy as np

    predictions = np.array(predictions)
    outcomes = np.array(outcomes)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges[:-1]) - 1

    bin_data = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_predicted = np.mean(predictions[mask])
            mean_observed = np.mean(outcomes[mask])
            count = np.sum(mask)

            bin_data.append({
                "bin_index": i,
                "mean_predicted": mean_predicted,
                "mean_observed": mean_observed,
                "count": count,
                "bin_range": (bin_edges[i], bin_edges[i + 1])
            })

    return {
        "bins": bin_data,
        "n_predictions": len(predictions)
    }


def calculate_ece(predictions: List[float], outcomes: List[int], n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error

    Args:
        predictions: List of predicted probabilities (0-1)
        outcomes: List of actual outcomes (0 or 1)
        n_bins: Number of bins

    Returns:
        Expected Calibration Error
    """
    import numpy as np

    calibration_data = calculate_calibration_data(predictions, outcomes, n_bins)

    total = len(predictions)
    ece = 0.0

    for bin_info in calibration_data["bins"]:
        weight = bin_info["count"] / total
        calibration_error = abs(bin_info["mean_observed"] - bin_info["mean_predicted"])
        ece += weight * calibration_error

    return ece


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_kalshi_usage():
    """Example: Fetch and analyze Kalshi data"""

    # Initialize client
    client = KalshiClient()

    # Fetch settled markets from last 6 months
    six_months_ago = int((datetime.now() - timedelta(days=180)).timestamp())
    markets = client.get_all_settled_markets(
        min_close_ts=six_months_ago,
        max_markets=1000
    )

    print(f"Fetched {len(markets)} settled markets")

    # Extract and clean data
    cleaned_data = [client.extract_market_data(m) for m in markets]

    # Filter to valid binary outcomes
    valid_markets = [
        m for m in cleaned_data
        if m["predicted_probability"] is not None and m["actual_outcome"] is not None
    ]

    print(f"Valid markets for analysis: {len(valid_markets)}")

    # Calculate overall Brier Score
    predictions = [m["predicted_probability"] for m in valid_markets]
    outcomes = [m["actual_outcome"] for m in valid_markets]

    brier_score = calculate_brier_score(predictions, outcomes)
    ece = calculate_ece(predictions, outcomes)

    print(f"Brier Score: {brier_score:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")

    # Save data
    with open("kalshi_markets.json", "w") as f:
        json.dump(valid_markets, f, indent=2)


def example_polymarket_usage():
    """Example: Fetch and analyze Polymarket data"""

    # Initialize client
    client = PolymarketClient()

    # Fetch closed markets
    markets = client.get_all_closed_markets(max_markets=1000)

    print(f"Fetched {len(markets)} closed markets")

    # Extract and clean data
    cleaned_data = [client.extract_market_data(m) for m in markets]

    # Note: You'll need to fetch resolution data separately
    # This might involve checking on-chain data or additional API endpoints

    # Save data
    with open("polymarket_markets.json", "w") as f:
        json.dump(cleaned_data, f, indent=2)

    print("Data saved. Resolution outcomes need to be added for calibration analysis.")


def example_combined_analysis():
    """Example: Compare Kalshi vs Polymarket calibration"""

    # Load data (assuming you've already collected it)
    with open("kalshi_markets.json", "r") as f:
        kalshi_data = json.load(f)

    with open("polymarket_markets.json", "r") as f:
        polymarket_data = json.load(f)

    # Calculate metrics for each platform
    kalshi_predictions = [m["predicted_probability"] for m in kalshi_data]
    kalshi_outcomes = [m["actual_outcome"] for m in kalshi_data]

    # (Polymarket would need resolution data added)

    kalshi_brier = calculate_brier_score(kalshi_predictions, kalshi_outcomes)
    kalshi_ece = calculate_ece(kalshi_predictions, kalshi_outcomes)

    print("=== Platform Comparison ===")
    print(f"Kalshi - Brier Score: {kalshi_brier:.4f}, ECE: {kalshi_ece:.4f}")
    # print(f"Polymarket - Brier Score: {poly_brier:.4f}, ECE: {poly_ece:.4f}")


if __name__ == "__main__":
    print("Prediction Markets Calibration Analysis - API Examples")
    print("=" * 60)

    # Uncomment to run examples
    # example_kalshi_usage()
    # example_polymarket_usage()

    print("\nReady to start data collection!")
    print("\nNext steps:")
    print("1. Run example_kalshi_usage() to fetch Kalshi data")
    print("2. Run example_polymarket_usage() to fetch Polymarket data")
    print("3. Add resolution data to Polymarket markets")
    print("4. Run calibration analysis and create visualizations")
