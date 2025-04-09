#!/usr/bin/env python3
"""
Simple Volatility Tracking Script

This script fetches historical price data for Bitcoin (BTC) and Sonic (S) from CoinGecko
and exports the data to CSV files that can be imported to Google Sheets.
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# CoinGecko API constants
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINS = [
    {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    {"id": "sonic-3", "symbol": "S", "name": "Sonic"}  # Updated to use latest CoinGecko ID for Sonic
]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "volatility")


class SimpleCoinGeckoAPI:
    """
    A simplified CoinGecko API client that handles rate limiting and API key usage.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the API client with an optional API key.
        
        Args:
            api_key: CoinGecko API key
        """
        self.api_key = api_key
        self.base_url = COINGECKO_BASE_URL
        self.headers = {"Accept": "application/json"}
        
        if api_key:
            self.headers["X-CG-Pro-API-Key"] = api_key
            logger.info("Using CoinGecko API with API key")
        else:
            logger.warning("Using CoinGecko API without API key (rate limits apply)")
    
    def _make_request(self, endpoint, params=None):
        """
        Make a request to the CoinGecko API with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Add a small delay to avoid hitting rate limits
            time.sleep(1.5)
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_coin_market_data(self, coin_id, vs_currency="usd", days=30):
        """
        Get historical market data for a coin.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency (default: USD)
            days: Number of days to fetch (max: 90)
            
        Returns:
            JSON data with prices, market caps, and volumes
        """
        endpoint = f"coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily"
        }
        
        return self._make_request(endpoint, params)
    
    def get_coin_hourly_data(self, coin_id, vs_currency="usd", days=2):
        """
        Get hourly market data for a coin.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency (default: USD)
            days: Number of days to fetch (should be small for hourly data)
            
        Returns:
            JSON data with hourly prices
        """
        endpoint = f"coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "hourly"
        }
        
        return self._make_request(endpoint, params)
    
    def verify_coin_id(self, coin_id):
        """
        Verify if a coin ID exists on CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID to verify
            
        Returns:
            True if valid, False otherwise
        """
        endpoint = f"coins/{coin_id}"
        data = self._make_request(endpoint)
        return data is not None


class PriceDataCSVExporter:
    """
    Exports cryptocurrency price data to CSV files.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the exporter with an API client.
        
        Args:
            api_key: CoinGecko API key
        """
        self.api = SimpleCoinGeckoAPI(api_key=api_key)
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def process_daily_data(self, coin_id, symbol):
        """
        Process and save daily price data.
        
        Args:
            coin_id: CoinGecko coin ID
            symbol: Coin symbol (for filename)
            
        Returns:
            Path to saved CSV file
        """
        logger.info(f"Fetching daily data for {symbol}")
        
        # Get daily data for the last 30 days
        data = self.api.get_coin_market_data(coin_id, days=30)
        
        if not data or "prices" not in data:
            logger.error(f"Failed to fetch daily data for {symbol}")
            return None
        
        # Convert to DataFrame
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Sort by timestamp in descending order and take the last 30 days
        df = df.sort_values(by="timestamp", ascending=False).head(30)
        
        # Format the timestamp
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        
        # Select and rename columns
        result_df = df[["date", "price"]].copy()
        result_df.columns = ["Date", "Price (USD)"]
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, f"{symbol}_daily.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved daily data to {output_path}")
        
        return output_path
    
    def process_hourly_data(self, coin_id, symbol):
        """
        Process and save hourly price data.
        
        Args:
            coin_id: CoinGecko coin ID
            symbol: Coin symbol (for filename)
            
        Returns:
            Path to saved CSV file
        """
        logger.info(f"Fetching hourly data for {symbol}")
        
        # Get hourly data for the last 2 days
        data = self.api.get_coin_hourly_data(coin_id, days=2)
        
        if not data or "prices" not in data:
            logger.error(f"Failed to fetch hourly data for {symbol}")
            return None
        
        # Convert to DataFrame
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Sort by timestamp in descending order and take the last 30 hours
        df = df.sort_values(by="timestamp", ascending=False).head(30)
        
        # Format the timestamp
        df["datetime"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:00")
        
        # Select and rename columns
        result_df = df[["datetime", "price"]].copy()
        result_df.columns = ["DateTime", "Price (USD)"]
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, f"{symbol}_hourly.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved hourly data to {output_path}")
        
        return output_path
    
    def verify_coin_ids(self):
        """
        Verify if the coin IDs are valid.
        
        Returns:
            Dictionary of coin IDs and their validity
        """
        results = {}
        
        for coin in COINS:
            logger.info(f"Verifying ID for {coin['name']}: {coin['id']}")
            valid = self.api.verify_coin_id(coin["id"])
            results[coin["id"]] = valid
            
            if valid:
                logger.info(f"✅ ID '{coin['id']}' is valid for {coin['name']}")
            else:
                logger.error(f"❌ ID '{coin['id']}' is invalid for {coin['name']}")
        
        return results
    
    def export_all_data(self):
        """
        Export data for all configured coins.
        
        Returns:
            Dictionary of output paths
        """
        results = {'daily': {}, 'hourly': {}}
        
        # First, verify coin IDs
        valid_ids = self.verify_coin_ids()
        
        for coin in COINS:
            if valid_ids.get(coin["id"], False):
                # Process daily data
                daily_path = self.process_daily_data(coin["id"], coin["symbol"])
                if daily_path:
                    results['daily'][coin["symbol"]] = daily_path
                
                # Process hourly data
                hourly_path = self.process_hourly_data(coin["id"], coin["symbol"])
                if hourly_path:
                    results['hourly'][coin["symbol"]] = hourly_path
            else:
                logger.warning(f"Skipping {coin['name']} due to invalid ID")
        
        return results


def main():
    """Main entry point for the script."""
    logger.info("Starting simple price data tracking")
    
    # Use API key from environment if available
    api_key = COINGECKO_API_KEY
    
    # Initialize exporter
    exporter = PriceDataCSVExporter(api_key=api_key)
    
    # Export all data
    results = exporter.export_all_data()
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    
    if results['daily'] or results['hourly']:
        print("\nFiles exported:")
        
        if results['daily']:
            print("\nDaily data:")
            for symbol, path in results['daily'].items():
                print(f"  - {symbol}: {path}")
        
        if results['hourly']:
            print("\nHourly data:")
            for symbol, path in results['hourly'].items():
                print(f"  - {symbol}: {path}")
        
        print("\nTo import into Google Sheets:")
        print("1. Open a new Google Sheet")
        print("2. Click on File > Import")
        print("3. Upload the CSV files")
        print("4. Choose 'Replace current sheet' or 'Insert new sheet(s)'")
    else:
        print("\nNo data was exported. Check the logs for errors.")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 