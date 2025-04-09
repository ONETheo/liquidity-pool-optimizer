#!/usr/bin/env python3
"""
Basic Volatility Tracker

This script fetches historical price data for Bitcoin (BTC), Ethereum (ETH), and Sonic (S) from CoinGecko API.
It processes the data and exports it to CSV files for further analysis.
"""

import os
import sys
import time
import json
import requests
import csv
from datetime import datetime, timedelta
import random

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "volatility")
COIN_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "S": "sonic-3",  # Updated to use new CoinGecko ID for Sonic
    "wS": "wrapped-sonic",  # Wrapped Sonic token
    "USDC.e": "sonic-bridged-usdc-e-sonic",  # Sonic bridged USDC-e
    "scUSD": "rings-scusd"  # Rings scUSD
}


def log_message(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def make_request(url, params=None, max_retries=3, retry_delay=1):
    """
    Make an API request with retries.
    
    Args:
        url: API endpoint URL
        params: Query parameters
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        API response as JSON or None if failed
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "USD* Volatility Tracker/1.0"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", retry_delay * 2))
                log_message(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            # Handle other errors
            if response.status_code != 200:
                log_message(f"Error {response.status_code}: {response.text}")
                time.sleep(retry_delay)
                continue
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            log_message(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(retry_delay)
    
    log_message(f"Failed to get data after {max_retries} attempts")
    return None


def get_coin_market_data(coin_id, vs_currency="usd", days=30, interval="daily"):
    """
    Get market data for a specific coin.
    
    Args:
        coin_id: CoinGecko coin ID
        vs_currency: Base currency (default: usd)
        days: Number of days of data to retrieve
        interval: Data interval (daily or hourly)
        
    Returns:
        Market data or None if failed
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days
    }
    
    # Only add interval parameter for daily data
    # For hourly data, CoinGecko recommends not specifying interval
    # and using days=2 or days=3 to get hourly data automatically
    if interval == "daily":
        params["interval"] = interval
    
    log_message(f"Fetching {interval} price data for {coin_id} (last {days} days)")
    return make_request(url, params)


def find_coin_id(symbol):
    """
    Find the CoinGecko coin ID for a given symbol.
    
    Args:
        symbol: Coin symbol (e.g., "BTC", "S")
        
    Returns:
        Coin ID or None if not found
    """
    # First, try the predefined mapping
    if symbol in COIN_MAP:
        return COIN_MAP[symbol]
    
    # If not in the mapping, search in the list of all coins
    log_message(f"Symbol {symbol} not found in predefined mappings, searching in all coins...")
    
    url = "https://api.coingecko.com/api/v3/coins/list"
    coins = make_request(url)
    
    if coins:
        for coin in coins:
            if coin["symbol"].upper() == symbol.upper():
                log_message(f"Found match: {coin['id']} for symbol {symbol}")
                return coin["id"]
    
    log_message(f"Could not find coin ID for symbol {symbol}")
    return None


def process_price_data(data, time_key="timestamp", price_key="price"):
    """
    Process price data from CoinGecko response.
    
    Args:
        data: CoinGecko price data response
        time_key: Key to use for timestamp in output
        price_key: Key to use for price in output
        
    Returns:
        Processed data as a list of dictionaries with timestamp, price, 
        percent_change_24h, and volatility fields
    """
    if not data or "prices" not in data:
        return []
    
    processed_data = []
    price_points = data["prices"]  # Format: [[timestamp, price], ...]
    
    # Sort by timestamp (ascending)
    price_points.sort(key=lambda x: x[0])
    
    for i, (timestamp, price) in enumerate(price_points):
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(timestamp / 1000)  # CoinGecko timestamps are in milliseconds
        
        # Calculate percent change from previous day (if available)
        percent_change_24h = None
        if i > 0:
            prev_price = price_points[i-1][1]
            percent_change_24h = ((price - prev_price) / prev_price) * 100
        
        # Add to processed data
        processed_data.append({
            time_key: dt.strftime("%Y-%m-%d %H:%M:%S"),
            price_key: round(price, 6),
            "percent_change_24h": round(percent_change_24h, 2) if percent_change_24h is not None else None,
            "volatility": abs(round(percent_change_24h, 2)) if percent_change_24h is not None else None
        })
    
    return processed_data


def ensure_data_dir():
    """Ensure the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)
    log_message(f"Ensuring data directory exists: {DATA_DIR}")


def save_to_csv(data, filename, headers=None):
    """
    Save data to a CSV file.
    
    Args:
        data: List of dictionaries with data to save
        filename: Output filename
        headers: Column headers (if None, use keys from first data item)
        
    Returns:
        Path to the saved file
    """
    if not data:
        log_message(f"No data to save to {filename}")
        return None
    
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, filename)
    
    # Determine headers if not provided
    if headers is None and data:
        headers = list(data[0].keys())
    
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        
        log_message(f"Saved {len(data)} rows to {filepath}")
        return filepath
    
    except Exception as e:
        log_message(f"Error saving to CSV: {e}")
        return None


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print(" VOLATILITY DATA COLLECTION ")
    print("=" * 60 + "\n")
    
    # Ensure data directory exists
    ensure_data_dir()
    
    # Collect data for each coin
    coin_symbols = ["BTC", "ETH", "S", "wS", "USDC.e", "scUSD"]  # Added Sonic Protocol assets
    output_files = []
    
    for symbol in coin_symbols:
        log_message(f"Processing {symbol}...")
        
        # Get coin ID
        coin_id = find_coin_id(symbol)
        if not coin_id:
            log_message(f"Error: Could not find coin ID for {symbol}")
            continue
        
        # Get daily data (different settings for BTC, ETH, and S)
        if symbol in ["BTC", "ETH"]:
            # Get 100 days for BTC and ETH
            daily_data = get_coin_market_data(coin_id, days=100, interval="daily")
        else:
            # Keep 30 days for other tokens
            daily_data = get_coin_market_data(coin_id, days=30, interval="daily")
            
        if daily_data:
            processed_daily = process_price_data(daily_data)
            daily_file = save_to_csv(processed_daily, f"{symbol}_daily.csv")
            if daily_file:
                output_files.append(daily_file)
        
        # Get hourly data (different settings for BTC, ETH, and S)
        if symbol in ["BTC", "ETH"]:
            # For BTC and ETH, we need 100 hours, which requires about 5 days of data
            hourly_data = get_coin_market_data(coin_id, days=5, interval="hourly")
            if hourly_data:
                processed_hourly = process_price_data(hourly_data)
                # Extract only the last 100 hours if we have enough data
                if len(processed_hourly) > 100:
                    processed_hourly = processed_hourly[-100:]
                hourly_file = save_to_csv(processed_hourly, f"{symbol}_hourly.csv")
                if hourly_file:
                    output_files.append(hourly_file)
        else:
            # For other tokens, keep the existing 24 hours approach
            hourly_data = get_coin_market_data(coin_id, days=2, interval="hourly")
            if hourly_data:
                processed_hourly = process_price_data(hourly_data)
                # Extract only the last 24 hours if we have enough data
                if len(processed_hourly) > 24:
                    processed_hourly = processed_hourly[-24:]
                hourly_file = save_to_csv(processed_hourly, f"{symbol}_hourly.csv")
                if hourly_file:
                    output_files.append(hourly_file)
        
        # Avoid rate limiting between coins
        time.sleep(2)
    
    # Print summary
    print("\n" + "=" * 60)
    print(" DATA COLLECTION SUMMARY ")
    print("=" * 60)
    
    if output_files:
        print(f"\nSuccessfully exported volatility data to CSV files:")
        for f in output_files:
            print(f"  - {os.path.basename(f)}")
        
        print("\nOutput path: " + DATA_DIR)
        print("\nTo import into Google Sheets:")
        print("1. Open Google Sheets")
        print("2. Click File > Import > Upload")
        print("3. Select the CSV files")
    else:
        print("\nNo data was exported. Please check the logs for errors.")


if __name__ == "__main__":
    main() 