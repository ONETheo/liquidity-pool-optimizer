"""
CoinGecko API Module

Provides functionality to fetch token price data from CoinGecko.
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# Import local modules
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from simulation import config
from simulation.config import DEFAULT_CONFIG

# Set up logger
logger = logging.getLogger(__name__)


class CoinGeckoAPI:
    """
    CoinGecko API wrapper for fetching token price data.
    """
    
    def __init__(self, api_config: Dict[str, Any] = None, api_key: str = None):
        """
        Initialize CoinGecko API wrapper.
        
        Args:
            api_config: Configuration for the API
            api_key: CoinGecko API key (optional)
        """
        # Initialize with default config if not provided
        self.api_config = api_config or DEFAULT_CONFIG["api_config"]["coingecko"]
        
        # Set up base URL
        self.base_url = self.api_config.get("base_url", "https://api.coingecko.com/api/v3")
        
        # Set up token IDs mapping
        self.token_ids = self.api_config.get("token_ids", {})
        
        # Set up rate limiting
        self.rate_limit = self.api_config.get("rate_limit", {
            "calls_per_minute": 10,
            "retry_after": 60
        })
        
        # Set up API key if provided
        self.api_key = api_key
        if not self.api_key and "api_key" in self.api_config:
            self.api_key = self.api_config["api_key"]
            
        # Track API calls for rate limiting
        self.api_calls = []
        
        logger.info(f"Initialized CoinGecko API with base URL: {self.base_url}")
        logger.info(f"API Key configured: {'Yes' if self.api_key else 'No'}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the CoinGecko API with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        # Implement rate limiting
        self._enforce_rate_limit()
        
        # Prepare URL and params
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        
        # Add API key if available
        headers = {}
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key
        
        # Make request
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Track this API call
            self.api_calls.append(time.time())
            
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            if response.status_code == 429:
                logger.warning("Rate limit hit, enforcing cooldown")
                time.sleep(self.rate_limit["retry_after"])
                return self._make_request(endpoint, params)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error occurred: {e}")
            raise
    
    def get_token_price_history(
        self,
        token_id: str,
        vs_currency: str = "usd",
        from_date: Union[str, datetime] = None,
        to_date: Union[str, datetime] = None,
        interval: str = "daily"
    ) -> pd.DataFrame:
        """
        Get historical price data for a cryptocurrency.
        
        Args:
            token_id (str): CoinGecko token ID or symbol
            vs_currency (str, optional): Currency to get prices in. Defaults to "usd".
            from_date (Union[str, datetime], optional): Start date. Defaults to 90 days ago.
            to_date (Union[str, datetime], optional): End date. Defaults to today.
            interval (str, optional): Data interval. Defaults to "daily".
            
        Returns:
            pd.DataFrame: DataFrame with timestamp and price columns
        """
        # Get token ID from symbol if necessary
        if token_id in self.token_ids:
            token_id = self.token_ids[token_id]
        
        # Convert date parameters
        from_timestamp, to_timestamp = self._parse_date_params(from_date, to_date)
        
        # Determine the appropriate API endpoint and parameters based on interval
        if interval == "daily":
            endpoint = f"/coins/{token_id}/market_chart/range"
            params = {
                "vs_currency": vs_currency,
                "from": from_timestamp,
                "to": to_timestamp
            }
        else:
            # For hourly data, we need a different approach
            # CoinGecko's free API only provides hourly data for last 90 days
            days = min(90, (to_timestamp - from_timestamp) // 86400 + 1)
            endpoint = f"/coins/{token_id}/market_chart"
            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": "hourly" if interval in ["hourly", "1h"] else interval
            }
        
        # Make the API request
        try:
            response_data = self._make_request(endpoint, params)
            
            # Process the response
            price_data = self._process_price_response(response_data, interval)
            
            # Filter to requested date range
            if interval != "daily":
                price_data = self._filter_date_range(price_data, from_timestamp, to_timestamp)
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching price data for {token_id}: {e}")
            # Return empty DataFrame
            return pd.DataFrame(columns=["timestamp", "price"])
    
    def get_price_data(
        self,
        base_token: str,
        quote_token: str = "USDC",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Get historical price data for a token pair.
        
        This method calculates the relative price between two tokens by
        fetching USD prices for both and computing the ratio.
        
        Args:
            base_token (str): Base token symbol (e.g., "USD*")
            quote_token (str): Quote token symbol (e.g., "USDC")
            start_date (Union[str, datetime], optional): Start date
            end_date (Union[str, datetime], optional): End date
            interval (str, optional): Data interval ("1h", "daily")
            
        Returns:
            pd.DataFrame: DataFrame with timestamp and price columns
        """
        # Convert interval to CoinGecko format
        cg_interval = "hourly" if interval in ["1h", "hourly"] else "daily"
        
        # Get price data for both tokens
        base_prices = self.get_token_price_history(
            base_token, "usd", start_date, end_date, cg_interval)
        
        quote_prices = self.get_token_price_history(
            quote_token, "usd", start_date, end_date, cg_interval)
        
        # Check if we have data
        if base_prices.empty or quote_prices.empty:
            logger.warning(f"Missing price data for {base_token} or {quote_token}")
            return pd.DataFrame(columns=["timestamp", "price"])
        
        # Merge price data
        merged = pd.merge(
            base_prices, 
            quote_prices, 
            on="timestamp", 
            suffixes=("_base", "_quote")
        )
        
        # Calculate the price ratio
        merged["price"] = merged["price_base"] / merged["price_quote"]
        
        # Keep only timestamp and price columns
        result = merged[["timestamp", "price"]].copy()
        
        return result
    
    def _parse_date_params(
        self, 
        from_date: Union[str, datetime, None], 
        to_date: Union[str, datetime, None]
    ) -> Tuple[int, int]:
        """
        Parse and convert date parameters to Unix timestamps.
        
        Args:
            from_date (Union[str, datetime, None]): Start date
            to_date (Union[str, datetime, None]): End date
            
        Returns:
            Tuple[int, int]: (from_timestamp, to_timestamp)
        """
        # Handle default dates
        if from_date is None:
            from_date = datetime.now() - timedelta(days=90)
        if to_date is None:
            to_date = datetime.now()
        
        # Convert string dates to datetime
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, "%Y-%m-%d")
        if isinstance(to_date, str):
            to_date = datetime.strptime(to_date, "%Y-%m-%d")
        
        # Convert to Unix timestamps (seconds)
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        return from_timestamp, to_timestamp
    
    def _process_price_response(
        self, 
        response_data: Dict[str, Any],
        interval: str
    ) -> pd.DataFrame:
        """
        Process CoinGecko API response into a DataFrame.
        
        Args:
            response_data (Dict[str, Any]): API response data
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: DataFrame with timestamp and price columns
        """
        if "prices" not in response_data:
            logger.error("Invalid API response: 'prices' field missing")
            return pd.DataFrame(columns=["timestamp", "price"])
        
        # Extract price data
        prices = response_data["prices"]
        
        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        
        # Convert millisecond timestamps to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Round timestamps based on interval for consistency
        if interval == "daily":
            df["timestamp"] = df["timestamp"].dt.floor("D")
        elif interval in ["hourly", "1h"]:
            df["timestamp"] = df["timestamp"].dt.floor("H")
        
        # Remove duplicates (in case of rounding)
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        
        return df
    
    def _filter_date_range(
        self, 
        df: pd.DataFrame, 
        from_timestamp: int, 
        to_timestamp: int
    ) -> pd.DataFrame:
        """
        Filter DataFrame to specified date range.
        
        Args:
            df (pd.DataFrame): DataFrame with timestamp column
            from_timestamp (int): Start timestamp
            to_timestamp (int): End timestamp
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Convert Unix timestamps to datetime for comparison
        from_dt = pd.to_datetime(from_timestamp, unit="s")
        to_dt = pd.to_datetime(to_timestamp, unit="s")
        
        # Filter DataFrame
        mask = (df["timestamp"] >= from_dt) & (df["timestamp"] <= to_dt)
        return df[mask].reset_index(drop=True)


# Function to get a shared API instance
_api_instance = None

def get_api_instance() -> CoinGeckoAPI:
    """
    Get a shared CoinGecko API instance.
    
    Returns:
        CoinGeckoAPI: Shared API instance
    """
    global _api_instance
    if _api_instance is None:
        _api_instance = CoinGeckoAPI()
    return _api_instance


def get_price_data(
    token_pair: str,
    start_date: Union[str, datetime] = None,
    end_date: Union[str, datetime] = None,
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Get historical price data for a token pair.
    
    Args:
        token_pair (str): Token pair (e.g., "USD*/USDC")
        start_date (Union[str, datetime], optional): Start date
        end_date (Union[str, datetime], optional): End date
        interval (str, optional): Data interval
        
    Returns:
        pd.DataFrame: DataFrame with timestamp and price columns
    """
    # Parse token pair
    tokens = token_pair.split("/")
    if len(tokens) != 2:
        logger.error(f"Invalid token pair format: {token_pair}. Expected format: 'BASE/QUOTE'")
        return pd.DataFrame(columns=["timestamp", "price"])
    
    base_token, quote_token = tokens
    
    # Get API instance
    api = get_api_instance()
    
    # Get price data
    return api.get_price_data(
        base_token, quote_token, start_date, end_date, interval)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the API
    api = CoinGeckoAPI()
    
    # Test getting price history for a token
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    
    print(f"Fetching daily price data for USD* from {start_date.date()} to {end_date.date()}")
    df = api.get_token_price_history("USD*", from_date=start_date, to_date=end_date)
    print("Results:")
    print(f"- Shape: {df.shape}")
    print(df.head())
    
    # Test getting price data for a token pair
    print("\nFetching hourly price data for USD*/USDC")
    pair_df = get_price_data("USD*/USDC", start_date, end_date, "1h")
    print("Results:")
    print(f"- Shape: {pair_df.shape}")
    print(pair_df.head()) 