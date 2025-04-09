#!/usr/bin/env python3
"""
Test script for the volatility tracking functionality.

This script tests the price data fetching functionality without requiring Google Sheets API.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import requests

# Set up path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.coingecko import CoinGeckoAPI
from simulation.config import get_config

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Tokens to test
COINS = [
    {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    {"id": "sonic-3", "symbol": "S", "name": "Sonic"}  # Updated to use latest CoinGecko ID
]


def test_price_data_fetching():
    """Test fetching price data from CoinGecko API."""
    # Get configuration
    config = get_config()
    
    # Initialize CoinGecko API
    cg_api = CoinGeckoAPI(api_config=config["api_config"]["coingecko"])
    
    for coin in COINS:
        logger.info(f"Testing price data fetching for {coin['name']} ({coin['symbol']})")
        
        # Test daily data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        daily_df = cg_api.get_token_price_history(
            coin["id"],
            vs_currency="usd",
            from_date=start_date,
            to_date=end_date,
            interval="daily"
        )
        
        logger.info(f"Daily data for {coin['symbol']}: {len(daily_df)} records")
        if not daily_df.empty:
            print(f"\nDaily price data for {coin['name']} ({coin['symbol']}):")
            print(daily_df.head(5))
        else:
            logger.error(f"No daily data found for {coin['symbol']}")
        
        # Test hourly data (last 30 hours)
        start_date = end_date - timedelta(days=2)  # Get 2 days to ensure we have 30 hours
        
        hourly_df = cg_api.get_token_price_history(
            coin["id"],
            vs_currency="usd",
            from_date=start_date,
            to_date=end_date,
            interval="hourly"
        )
        
        hourly_df = hourly_df.sort_values(by="timestamp", ascending=False).head(30)
        
        logger.info(f"Hourly data for {coin['symbol']}: {len(hourly_df)} records")
        if not hourly_df.empty:
            print(f"\nHourly price data for {coin['name']} ({coin['symbol']}):")
            print(hourly_df.head(5))
        else:
            logger.error(f"No hourly data found for {coin['symbol']}")
        
        print("\n" + "-" * 50 + "\n")


def verify_sonic_id():
    """Verify the correct CoinGecko ID for Sonic token."""
    # Create CoinGecko API session
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    
    # Try different potential IDs for Sonic
    potential_ids = ["sonic-3", "sonic", "sonic-token"]
    
    print("\nVerifying Sonic token ID:")
    for token_id in potential_ids:
        try:
            logger.info(f"Testing token ID: {token_id}")
            df = cg_api.get_token_price_history(
                token_id,
                vs_currency="usd",
                from_date=datetime.now() - timedelta(days=2),
                to_date=datetime.now(),
                interval="daily"
            )
            
            if not df.empty:
                print(f"✅ ID '{token_id}' is valid for Sonic token")
                print(f"Sample data: {df.head(1)}")
            else:
                print(f"❌ ID '{token_id}' returned empty data")
                
        except Exception as e:
            print(f"❌ ID '{token_id}' error: {e}")
    
    print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("VOLATILITY TRACKING TEST")
    print("=" * 50 + "\n")
    
    # First verify the Sonic token ID
    verify_sonic_id()
    
    # Then test price data fetching
    test_price_data_fetching()
    
    print("Test completed") 