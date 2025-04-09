#!/usr/bin/env python3
"""
Custom Chart Pairings Generator

This script generates specific chart pairings for USD* pools visualization:
1. Price charts with price changes for multiple assets
2. Pairing chart for scUSD and USDC.e
3. S (Sonic) price with price changes
4. S and wS (Wrapped Sonic) pairing
5. Price change pairing of wS and USDC.e
6. BTC price and price change
7. ETH price and price change
8. BTC and ETH pairing

Usage:
    python custom_pairings.py --output <output_directory> --days <lookback_days>
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import requests
import time
import yaml
import json
import random

# Set matplotlib style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')

# Simple CoinGecko API implementation to avoid dependency issues
class CoinGeckoAPI:
    def __init__(self, config=None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = config.get('api_key') if config else None
        self.rate_limit_sleep = 3.0  # At least 2 seconds per call (30 calls per minute max)
        self.max_retries = 3
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        # Track last request time for rate limiting
        self.last_request_time = 0
    
    def get_coin_market_chart_by_id(self, coin_id, vs_currency='usd', days=30):
        """Get historical market data for a coin with caching"""
        cache_file = os.path.join(self.cache_dir, f"{coin_id}_{vs_currency}_{days}.json")
        
        # Check if we have cached data that's not too old
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            # Use cache if it's less than 6 hours old
            if file_age < 6 * 3600:
                try:
                    with open(cache_file, 'r') as f:
                        print(f"Using cached data for {coin_id}")
                        return json.load(f)
                except Exception as e:
                    print(f"Error reading cache for {coin_id}: {str(e)}")
        
        # If no cache or cache is too old, fetch from API
        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        headers = {}
        if self.api_key:
            headers['x-cg-pro-api-key'] = self.api_key
        
        response = self._make_request('GET', endpoint, params=params, headers=headers)
        
        # Cache the successful response
        if response and 'prices' in response:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(response, f)
            except Exception as e:
                print(f"Error caching data for {coin_id}: {str(e)}")
                
        return response
    
    def _make_request(self, method, endpoint, params=None, headers=None):
        """Make API request with rate limiting and error handling"""
        for attempt in range(self.max_retries):
            try:
                # Ensure we wait at least rate_limit_sleep seconds since last request
                current_time = time.time()
                elapsed = current_time - self.last_request_time
                if elapsed < self.rate_limit_sleep:
                    wait_time = self.rate_limit_sleep - elapsed
                    # Add jitter to avoid synchronized requests (5-15% variation)
                    jitter = wait_time * random.uniform(0.05, 0.15)
                    time.sleep(wait_time + jitter)
                
                print(f"Making API request to {endpoint.split('/')[-2]}")
                self.last_request_time = time.time()
                response = requests.request(method, endpoint, params=params, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    wait_time = min(30, 5 * (2 ** attempt))  # Exponential backoff with max 30 seconds
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"Error: API request failed with status code {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
            
            except Exception as e:
                print(f"Error making request: {str(e)}")
                time.sleep(2)
        
        print(f"Failed after {self.max_retries} attempts to access {endpoint}")
        return None

# Simple config loader
def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            print(f"Config file not found: {config_path}")
            return {}
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}

def setup_plot(title, figsize=(12, 6)):
    """Create and setup a new figure with proper styling"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    return fig, ax

def format_axis(ax, y_label):
    """Apply consistent formatting to axis"""
    ax.set_ylabel(y_label, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

def fetch_price_data(api, token_id, days):
    """Fetch historical price data for a token"""
    try:
        data = api.get_coin_market_chart_by_id(token_id, vs_currency='usd', days=days)
        if not data or 'prices' not in data:
            print(f"Error: No price data available for {token_id}")
            return None
            
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['daily_change'] = df['price'].pct_change() * 100
        return df
    except Exception as e:
        print(f"Error fetching data for {token_id}: {str(e)}")
        return None

def plot_price_chart(ax, df, token_name, color):
    """Plot price chart for a token"""
    dates = df['date']
    ax.plot(dates, df['price'], label=f"{token_name} Price", color=color, linewidth=2)
    
    # Add 7-day moving average
    ma7 = df['price'].rolling(window=7).mean()
    ax.plot(dates, ma7, label=f"{token_name} 7D MA", color=color, linestyle='--', alpha=0.7)
    
    # Format y-axis to show dollar values
    ax.yaxis.set_major_formatter('${x:,.2f}')
    
    return ax

def plot_price_change_chart(ax, df, token_name, color):
    """Plot price change chart for a token"""
    dates = df['date']
    ax.plot(dates, df['daily_change'], label=f"{token_name} Daily Change", color=color, linewidth=2)
    
    # Add horizontal line at 0%
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Format y-axis to show percentage
    ax.yaxis.set_major_formatter('{x:,.1f}%')
    
    return ax

def plot_pairing_chart(ax1, df1, df2, token1_name, token2_name, color1, color2):
    """Plot two tokens on the same chart with dual y-axis"""
    dates1 = df1['date']
    dates2 = df2['date']
    
    # Left y-axis for token1
    ax1.plot(dates1, df1['price'], label=f"{token1_name}", color=color1, linewidth=2)
    ax1.set_ylabel(f"{token1_name} Price (USD)", color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.yaxis.set_major_formatter('${x:,.4f}')
    
    # Right y-axis for token2
    ax2 = ax1.twinx()
    ax2.plot(dates2, df2['price'], label=f"{token2_name}", color=color2, linewidth=2)
    ax2.set_ylabel(f"{token2_name} Price (USD)", color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.yaxis.set_major_formatter('${x:,.4f}')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    return ax1, ax2

def generate_custom_charts(config, output_dir, days=30, cache_only=False):
    """Generate all the custom chart pairings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CoinGecko API
    api = CoinGeckoAPI(config.get('coingecko', {}))
    
    # Define tokens and their colors
    tokens = {
        'bitcoin': {'name': 'BTC', 'color': '#F7931A'},
        'ethereum': {'name': 'ETH', 'color': '#627EEA'},
        'sonic-3': {'name': 'S', 'color': '#3D58B0'},  # Updated to sonic-3
        'wrapped-sonic': {'name': 'wS', 'color': '#5D78D0'},  # Wrapped Sonic
        'usd-coin': {'name': 'USDC.e', 'color': '#2775CA'},
        'usd-coin-wormhole-from-ethereum': {'name': 'scUSD', 'color': '#16A34A'}  # Using closest match for scUSD
    }
    
    # Fetch data for all tokens with controlled rate limiting
    token_data = {}
    for token_id, info in tokens.items():
        print(f"Fetching data for {info['name']} ({token_id})...")
        
        # If cache_only is True, only use cached data
        if cache_only:
            cache_file = os.path.join(api.cache_dir, f"{token_id}_usd_{days}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        prices = data['prices']
                        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['daily_change'] = df['price'].pct_change() * 100
                        token_data[token_id] = {
                            'df': df,
                            'name': info['name'],
                            'color': info['color']
                        }
                        print(f"Using cached data for {info['name']}")
                except Exception as e:
                    print(f"Error reading cache for {token_id}: {str(e)}")
            else:
                print(f"No cached data available for {info['name']}")
        else:
            df = fetch_price_data(api, token_id, days)
            if df is not None:
                token_data[token_id] = {
                    'df': df,
                    'name': info['name'],
                    'color': info['color']
                }
    
    # Check if we have all the required data
    if len(token_data) < len(tokens):
        print(f"Warning: Could only fetch data for {len(token_data)} out of {len(tokens)} tokens")
    
    # Skip chart generation if no data is available
    if not token_data:
        print("No data available to generate charts. Please try again later or use cached data.")
        return
    
    # 1. Price charts with price changes for multiple assets (one combined chart)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot prices on top chart
    for token_id, data in token_data.items():
        plot_price_chart(ax1, data['df'], data['name'], data['color'])
    
    ax1.set_title("Asset Prices", fontsize=16)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot price changes on bottom chart
    for token_id, data in token_data.items():
        plot_price_change_chart(ax2, data['df'], data['name'], data['color'])
    
    ax2.set_title("Daily Price Changes (%)", fontsize=16)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Date", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_all_assets_combined.png"), dpi=300)
    plt.close()
    
    # 2. Pairing chart for scUSD and USDC.e
    scusd_key = 'usd-coin-wormhole-from-ethereum'  # Using as replacement for scUSD
    if scusd_key in token_data and 'usd-coin' in token_data:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        plot_pairing_chart(
            ax1, 
            token_data[scusd_key]['df'], 
            token_data['usd-coin']['df'],
            token_data[scusd_key]['name'],
            token_data['usd-coin']['name'],
            token_data[scusd_key]['color'],
            token_data['usd-coin']['color']
        )
        plt.title("scUSD vs USDC.e Price Comparison", fontsize=16, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "2_scusd_usdc_pairing.png"), dpi=300)
        plt.close()
    
    # 3. S (Sonic) price with price changes
    sonic_key = 'sonic-3'  # Updated to sonic-3
    if sonic_key in token_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        plot_price_chart(ax1, token_data[sonic_key]['df'], 'S', token_data[sonic_key]['color'])
        ax1.set_title("Sonic (S) Price", fontsize=16)
        ax1.legend(loc='upper left')
        
        plot_price_change_chart(ax2, token_data[sonic_key]['df'], 'S', token_data[sonic_key]['color'])
        ax2.set_title("Sonic (S) Daily Price Change (%)", fontsize=16)
        ax2.legend(loc='upper left')
        ax2.set_xlabel("Date", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_sonic_price_and_change.png"), dpi=300)
        plt.close()
    
    # 4. S and wS (Wrapped Sonic) pairing
    if sonic_key in token_data and 'wrapped-sonic' in token_data:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        plot_pairing_chart(
            ax1, 
            token_data[sonic_key]['df'], 
            token_data['wrapped-sonic']['df'],
            token_data[sonic_key]['name'],
            token_data['wrapped-sonic']['name'],
            token_data[sonic_key]['color'],
            token_data['wrapped-sonic']['color']
        )
        plt.title("Sonic (S) vs Wrapped Sonic (wS) Price Comparison", fontsize=16, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "4_sonic_wsonic_pairing.png"), dpi=300)
        plt.close()
    
    # 5. Price change pairing of wS and USDC.e
    if 'wrapped-sonic' in token_data and 'usd-coin' in token_data:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Use price change instead of price for these
        ws_df = token_data['wrapped-sonic']['df'].copy()
        usdc_df = token_data['usd-coin']['df'].copy()
        
        # Left y-axis for wS price change
        ax1.plot(ws_df['date'], ws_df['daily_change'], label=f"wS Change %", 
                 color=token_data['wrapped-sonic']['color'], linewidth=2)
        ax1.set_ylabel("wS Daily Change (%)", color=token_data['wrapped-sonic']['color'], fontsize=12)
        ax1.tick_params(axis='y', labelcolor=token_data['wrapped-sonic']['color'])
        ax1.yaxis.set_major_formatter('{x:,.1f}%')
        
        # Right y-axis for USDC.e price change
        ax2 = ax1.twinx()
        ax2.plot(usdc_df['date'], usdc_df['daily_change'], label=f"USDC.e Change %", 
                 color=token_data['usd-coin']['color'], linewidth=2)
        ax2.set_ylabel("USDC.e Daily Change (%)", color=token_data['usd-coin']['color'], fontsize=12)
        ax2.tick_params(axis='y', labelcolor=token_data['usd-coin']['color'])
        ax2.yaxis.set_major_formatter('{x:,.1f}%')
        
        # Add horizontal line at 0%
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title("Wrapped Sonic (wS) vs USDC.e Daily Price Changes", fontsize=16, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "5_wsonic_usdc_change_pairing.png"), dpi=300)
        plt.close()
    
    # 6. BTC price and price change
    if 'bitcoin' in token_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        plot_price_chart(ax1, token_data['bitcoin']['df'], 'BTC', token_data['bitcoin']['color'])
        ax1.set_title("Bitcoin (BTC) Price", fontsize=16)
        ax1.legend(loc='upper left')
        
        plot_price_change_chart(ax2, token_data['bitcoin']['df'], 'BTC', token_data['bitcoin']['color'])
        ax2.set_title("Bitcoin (BTC) Daily Price Change (%)", fontsize=16)
        ax2.legend(loc='upper left')
        ax2.set_xlabel("Date", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "6_btc_price_and_change.png"), dpi=300)
        plt.close()
    
    # 7. ETH price and price change
    if 'ethereum' in token_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        plot_price_chart(ax1, token_data['ethereum']['df'], 'ETH', token_data['ethereum']['color'])
        ax1.set_title("Ethereum (ETH) Price", fontsize=16)
        ax1.legend(loc='upper left')
        
        plot_price_change_chart(ax2, token_data['ethereum']['df'], 'ETH', token_data['ethereum']['color'])
        ax2.set_title("Ethereum (ETH) Daily Price Change (%)", fontsize=16)
        ax2.legend(loc='upper left')
        ax2.set_xlabel("Date", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "7_eth_price_and_change.png"), dpi=300)
        plt.close()
    
    # 8. BTC and ETH pairing
    if 'bitcoin' in token_data and 'ethereum' in token_data:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        plot_pairing_chart(
            ax1, 
            token_data['bitcoin']['df'], 
            token_data['ethereum']['df'],
            token_data['bitcoin']['name'],
            token_data['ethereum']['name'],
            token_data['bitcoin']['color'],
            token_data['ethereum']['color']
        )
        plt.title("Bitcoin (BTC) vs Ethereum (ETH) Price Comparison", fontsize=16, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "8_btc_eth_pairing.png"), dpi=300)
        plt.close()
    
    print(f"All charts generated successfully in {output_dir}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate custom chart pairings for USD* pools visualization')
    parser.add_argument('--output', type=str, default='./charts', help='Output directory for charts')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to fetch')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--cache', action='store_true', help='Use cached data only (no API calls)')
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        print(f"Generating custom chart pairings for the last {args.days} days...")
        generate_custom_charts(config, args.output, args.days, args.cache)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 