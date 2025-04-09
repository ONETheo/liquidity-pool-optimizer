#!/usr/bin/env python3
"""
vfat Strategy Simulator

This script provides a CLI interface for backtesting vfat automated liquidity providing strategies.
It allows testing different parameter combinations for width, buffer, and cutoff values 
and calculates potential returns, APR, and comparison to hodl strategy.

Usage:
    python vfat_simulator.py --pair USDC.e/scUSD --amount 1000 --days 30 --width 1.0 --buffer 0.2 --cutoff 0.02
"""

import os
import sys
import argparse
import json
import requests
import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
POOL_ADDRESS = {
    "USDC.e/scUSD": "0x2C13383855377faf5A562F1AeF47E4be7A0f12Ac",
    "S/wS": "0xb607E73aF53C045c220C1d27337913d5eeA12d72"
}

# Fee tiers for specific pools
FEE_TIERS = {
    "USDC.e/scUSD": 0.0001,  # 0.01%
    "S/wS": 0.0025,  # 0.25%
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants for API settings
API_BASE_URL = "https://api.geckoterminal.com/api/v2"
API_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Pool Networks
SONIC_NETWORK = "sonic"
ETHEREUM_NETWORK = "eth"
POLYGON_NETWORK = "polygon"

# Pool IDs (used for API lookups)
POOL_IDS = {
    "USDC.e/scUSD": "0x2C13383855377faf5A562F1AeF47E4be7A0f12Ac",
    "USDC/scUSD": "scusd-usdcd",
    "ETH/BTC": "eth-wbtc",
    "DAI/USDC": "dai-usdc",
    "ETH/USDC": "eth-usdc",
    "WBTC/ETH": "wbtc-eth",
    "WMATIC/USDC": "wmatic-usdc",
    "USDT/USDC": "usdt-usdc",
}

# Networks mapping to their default tokens
NETWORK_DEFAULT_TOKENS = {
    SONIC_NETWORK: {
        "scUSD": {
            "decimals": 6,
            "symbol": "scUSD"
        },
        "USDC.e": {
            "decimals": 6,
            "symbol": "USDC.e"
        },
        "USDC": {
            "decimals": 6,
            "symbol": "USDC"
        }
    },
    ETHEREUM_NETWORK: {
        "USDC": {
            "decimals": 6,
            "symbol": "USDC"
        },
        "ETH": {
            "decimals": 18,
            "symbol": "ETH"
        },
        "DAI": {
            "decimals": 18,
            "symbol": "DAI"
        },
        "WBTC": {
            "decimals": 8,
            "symbol": "WBTC"
        },
        "USDT": {
            "decimals": 6,
            "symbol": "USDT"
        }
    },
    POLYGON_NETWORK: {
        "USDC": {
            "decimals": 6,
            "symbol": "USDC"
        },
        "WMATIC": {
            "decimals": 18,
            "symbol": "WMATIC"
        }
    }
}

# Map pairs to their networks
POOL_NETWORKS = {
    "USDC.e/scUSD": SONIC_NETWORK,
    "USDC/scUSD": SONIC_NETWORK,
    "ETH/BTC": ETHEREUM_NETWORK,
    "DAI/USDC": ETHEREUM_NETWORK,
    "ETH/USDC": ETHEREUM_NETWORK,
    "WBTC/ETH": ETHEREUM_NETWORK,
    "WMATIC/USDC": POLYGON_NETWORK,
    "USDT/USDC": ETHEREUM_NETWORK,
}

class APIClient:
    """API client for fetching data from GeckoTerminal API"""
    
    def __init__(self):
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session
    
    def get_pool_info(self, network: str, pool_id: str) -> Dict:
        """Fetch pool information from the API"""
        url = f"{API_BASE_URL}/networks/{network}/pools/{pool_id}"
        
        try:
            response = self.session.get(url, headers=API_HEADERS, timeout=10)
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            print(f"Error fetching pool info: {e}")
            return {}
    
    def get_pool_ohlcv(self, network: str, pool_id: str, timeframe: str = "day", limit: int = 100) -> Dict:
        """
        Fetch pool OHLCV data from the API
        
        Args:
            network: Network identifier (e.g., "sonic", "eth", "polygon")
            pool_id: Pool identifier
            timeframe: Data timeframe ("day", "hour", "minute")
            limit: Number of results to return (max 1000)
            
        Returns:
            Dictionary with OHLCV data and token metadata
        """
        # Remove network prefix if present (e.g., "sonic_" from pool_id)
        if pool_id.startswith(f"{network}_"):
            clean_pool_id = pool_id[len(network) + 1:]
        else:
            clean_pool_id = pool_id
            
        url = f"{API_BASE_URL}/networks/{network}/pools/{clean_pool_id}/ohlcv/{timeframe}"
        params = {"limit": limit, "currency": "usd"}
        
        try:
            response = self.session.get(url, headers=API_HEADERS, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching pool OHLCV data: {e}")
            return {}

# vfat Strategy Simulator
class VfatSimulator:
    """
    Simulator for the vfat liquidity providing strategy.
    
    This class simulates the automated strategy with configurable parameters:
    - width: The price range width in percent (±width/2 from current price)
    - buffer: The buffer zone in percent before rebalancing
    - cutoff: The maximum percent deviation before forced rebalancing
    - max_dust_percent: Max percentage of dust allowed during rebalancing (default: 1%)
    - max_slippage: Maximum allowed slippage for swaps (default: 0.5%)
    - swap_max_price_impact: Maximum allowed price impact for swaps (default: 0.5%)
    """
    
    def __init__(
        self,
        pair: str,
        initial_amount: float = 1000,
        backtest_days: int = 30,
        width: float = 1.0,
        buffer: float = 0.2,
        cutoff: float = 5.0,
        max_dust_percent: float = 1.0,
        max_slippage: float = 0.5,
        swap_max_price_impact: float = 0.5,
        plot_results: bool = True,
    ):
        """
        Initialize the simulator with strategy parameters.
        
        Args:
            pair: Trading pair in format "TOKEN0/TOKEN1"
            initial_amount: Initial investment amount in USD
            backtest_days: Number of days to simulate
            width: Range width in percent (e.g., 1.0 = ±0.5% around the current price)
            buffer: Buffer zone in percent before rebalancing (acts as hysteresis)
            cutoff: Maximum price deviation allowed before forced rebalancing
            max_dust_percent: Maximum percentage of dust allowed during rebalancing
            max_slippage: Maximum slippage percentage allowed for swaps
            swap_max_price_impact: Maximum price impact percentage allowed for swaps
            plot_results: Whether to plot results
        """
        # Pair and tokens
        self.pair = pair
        tokens = pair.split("/")
        self.base_token = tokens[0]
        self.quote_token = tokens[1]
        self.network = POOL_NETWORKS.get(pair, "sonic_sonic")
        self.pool_id = POOL_IDS.get(pair)
        
        # Strategy parameters
        self.initial_amount = initial_amount
        self.backtest_days = backtest_days
        self.width = width
        self.buffer = buffer
        self.cutoff = cutoff
        self.max_dust_percent = max_dust_percent
        self.max_slippage = max_slippage
        self.swap_max_price_impact = swap_max_price_impact
        self.plot_results = plot_results
        
        # Initialize API client
        self.api_client = APIClient()
        
        # Results storage
        self.data = None
        self.results = None
        self.rebalance_events = []
        
        # Store real pool liquidity (from market data)
        # For USDC.e/scUSD pair
        if self.pair == "USDC.e/scUSD":
            self.real_liquidity = 3890000  # $3.89M from current pool data
        else:
            # For other pairs, we could fetch dynamically but use a default for now
            self.real_liquidity = 3890000  # Default to same value
    
    def fetch_pool_historical_data(self) -> pd.DataFrame:
        """
        Fetch historical data for the current pool
        
        Returns:
            pd.DataFrame: Historical data as a DataFrame
        """
        # Check if pool ID is available
        if not self.pool_id:
            logger.error(f"No pool ID available for {self.pair}. Cannot fetch historical data.")
            return pd.DataFrame()
        
        # Check for cached data
        cache_file = os.path.join(CACHE_DIR, f"{self.pair.replace('/', '_')}_{self.backtest_days}days_data.csv")
        
        # If cache exists and is recent, use it
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 86400:  # Less than 24 hours old
                logger.info(f"Using cached data for {self.pair} (cache age: {file_age/3600:.1f} hours)")
                return pd.read_csv(cache_file)
        
        logger.info(f"Fetching historical data for {self.pair} (pool_id: {self.pool_id})...")
        
        # Fetch data from API
        response_data = self.api_client.get_pool_ohlcv(
            self.network, 
            self.pool_id, 
            timeframe="day", 
            limit=self.backtest_days
        )
        
        # Check if we have valid OHLCV data
        if not response_data or 'data' not in response_data:
            logger.warning(f"Failed to fetch OHLCV data for {self.pair} from GeckoTerminal. Trying cached file...")
            
            # Try to use cached file even if older than 24 hours
            if os.path.exists(cache_file):
                logger.info(f"Using existing cache file for {self.pair}")
                return pd.read_csv(cache_file)
            
            # Check for parameter_optimizer cached file (which might use a different naming convention)
            alt_cache_file = os.path.join(CACHE_DIR, f"0x2C13383855377faf5A562F1AeF47E4be7A0f12Ac_historical_{self.backtest_days}.csv")
            if os.path.exists(alt_cache_file):
                logger.info(f"Using parameter_optimizer cache file")
                df = pd.read_csv(alt_cache_file)
                # Convert column names if necessary to match expected format
                if 'timestamp' in df.columns and 'date' not in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'])
                return df
            
            logger.error(f"No valid data available for {self.pair}. Please run parameter_optimizer first to generate cached data.")
            return pd.DataFrame()
            
        # Extract OHLCV list from the data
        try:
            ohlcv_list = response_data['data']['attributes']['ohlcv_list']
            if not ohlcv_list:
                logger.error(f"No OHLCV data found for {self.pair}")
                return pd.DataFrame()
        except KeyError:
            logger.error(f"Invalid OHLCV data structure for {self.pair}")
            return pd.DataFrame()
        
        # Process OHLCV entries
        entries = []
        for entry in ohlcv_list:
            try:
                # OHLCV format is [timestamp, open, high, low, close, volume]
                if len(entry) < 6:
                    logger.warning(f"Skipping incomplete OHLCV entry: {entry}")
                    continue
                
                # Extract data
                timestamp = datetime.fromtimestamp(entry[0])
                open_price = float(entry[1])
                high_price = float(entry[2])
                low_price = float(entry[3])
                close_price = float(entry[4])
                volume = float(entry[5])
                
                # Create an entry dictionary
                entry_dict = {
                    'date': timestamp,
                    'price': close_price,
                    'base_quote_ratio': close_price,  # Price is the base/quote ratio
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume_usd': volume
                }
                
                # Try to extract liquidity data if available
                liquidity_usd = 0.0
                if 'meta' in response_data and 'liquidity_usd' in response_data['meta']:
                    liquidity_usd = float(response_data['meta'].get('liquidity_usd', 0))
                
                entry_dict['liquidity_usd'] = liquidity_usd
                entries.append(entry_dict)
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error processing OHLCV entry: {e}")
                continue
        
        # Create DataFrame from entries
        if not entries:
            logger.error("No valid entries found in OHLCV data")
            return pd.DataFrame()
        
        df = pd.DataFrame(entries)
        
        # Sort by date
        df = df.sort_values(by='date')
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached {len(df)} days of historical data to {cache_file}")
        
        return df
    
    def calculate_hodl_value(self) -> float:
        """Calculate the value of a HODL strategy (buy and hold)"""
        if self.data is None or len(self.data) == 0:
            print("No data available for HODL calculation")
            return 0.0
        
        # Initialize results dictionary if it doesn't exist
        if self.results is None:
            self.results = {}
        
        # Get prices at the beginning and end of the period
        start_price_ratio = self.data.iloc[0]['base_quote_ratio']
        end_price_ratio = self.data.iloc[-1]['base_quote_ratio']
        
        # Calculate value of a simple buy and hold strategy
        # Start with 50/50 allocation at the beginning prices
        base_token_amount = self.initial_amount / 2
        quote_token_amount = self.initial_amount / 2 / start_price_ratio
        
        # Calculate final value (in USD)
        final_value = base_token_amount + quote_token_amount * end_price_ratio
        
        # Store in results and as instance attribute
        self.results['hodl_value'] = final_value
        self.hodl_value = final_value
        
        return final_value
    
    def simulate_vfat_strategy(self) -> Dict:
        """
        Simulate vfat automated liquidity providing strategy
        
        Returns:
            Dictionary with simulation results
        """
        if self.data is None or len(self.data) < 2:
            raise ValueError("No historical data available")
        
        # Reset tracking variables
        position_values = []
        position_base = []
        position_quote = []
        price_in_range = []
        rebalance_count = 0
        fees_earned = 0
        rebalance_costs = 0
        rebalance_events = []  # Track which days rebalances occur
        value_lost_on_rebalances = 0  # Track total value lost from rebalancing
        
        # Determine fee rate based on pool and exchange
        fee_rate = FEE_TIERS.get(self.pair, 0.003)  # Get fee tier for this specific pool
        
        # Get initial price
        start_price = self.data.iloc[0]['base_quote_ratio']
        current_price = start_price
        initial_price = start_price
        
        # Initialize position (50/50 allocation)
        base_amount = self.initial_amount / 2
        quote_amount = self.initial_amount / 2 / current_price
        position_value = self.initial_amount
        
        # Get max dust and slippage thresholds
        max_dust_threshold = self.max_dust_percent / 100
        max_slippage_threshold = self.max_slippage / 100
        max_price_impact = self.swap_max_price_impact / 100
        
        # Store initial position value
        position_values.append(position_value)
        position_base.append(base_amount * current_price)
        position_quote.append(quote_amount)
        
        # Iterate through historical prices for simulation
        for i, day in self.data.iloc[1:].iterrows():
            # Update current price
            current_price = day['base_quote_ratio']
            
            # Calculate normalized price (relative to initial position price)
            normalized_price = current_price / initial_price
            
            # Calculate price range based on width
            width_factor = self.width / 100
            lower_bound = 1 - width_factor
            upper_bound = 1 + width_factor
            
            # Check if price is in range for fee calculation
            in_range = lower_bound <= normalized_price <= upper_bound
            price_in_range.append(in_range)
            
            # Calculate buffer boundaries
            buffer_factor = self.buffer / 100
            lower_buffer = 1 - buffer_factor
            upper_buffer = 1 + buffer_factor
            
            # Calculate cutoff boundaries
            cutoff_factor = self.cutoff / 100
            cutoff_lower = 1 - cutoff_factor
            cutoff_upper = 1 + cutoff_factor
            
            # Track if rebalance is needed
            need_rebalance = False
            rebalance_reason = ""
            
            # Check price against buffer boundaries
            if normalized_price < lower_buffer or normalized_price > upper_buffer:
                need_rebalance = True
                rebalance_reason = "Buffer exceeded"
            
            # Check price against cutoff boundaries
            if normalized_price < cutoff_lower or normalized_price > cutoff_upper:
                need_rebalance = False  # Skip rebalance if outside cutoff (too expensive)
                rebalance_reason = "Cutoff exceeded - skipping rebalance"
            
            # Calculate current position value before any actions
            pre_action_value = base_amount * current_price + quote_amount
            
            # Calculate fees earned for the day based on direct volume data
            daily_volume = day.get('volume_usd', 0)
            
            # Only earn fees when price is in range
            if in_range and daily_volume > 0:
                # Calculate fees using actual pool liquidity
                # Use the real liquidity value of $3.89M for the pool
                pool_share = position_value / self.real_liquidity
                
                # Daily fees = Pool volume * Fee rate * Your pool share
                daily_fee = daily_volume * fee_rate * pool_share
                fees_earned += daily_fee
                
                # Add fees to position value
                position_value = pre_action_value + daily_fee
                
                # Distribute fee earnings between tokens (simplified)
                fee_split = daily_fee / 2
                base_amount += fee_split / current_price
                quote_amount += fee_split
            else:
                position_value = pre_action_value
            
            # Perform rebalancing if needed and not outside cutoff
            if need_rebalance:
                rebalance_count += 1
                rebalance_events.append(i)  # Record the day index when rebalance occurred
                
                # Check remaining position value after dust
                pre_rebalance_value = base_amount * current_price + quote_amount
                
                # Calculate a more realistic rebalance cost:
                # 1. Protocol fee (typically 0.1-0.3% of the value being swapped)
                # 2. Slippage (depends on liquidity and size of the swap)
                # 3. Price impact (depends on market volatility)
                
                # Assume we need to swap ~50% of position value (worst case)
                swap_value = pre_rebalance_value * 0.5
                
                # Protocol fee component (based on swap value)
                protocol_fee = swap_value * fee_rate
                
                # Slippage and price impact components
                slippage = swap_value * np.random.uniform(0, max_slippage_threshold)
                price_impact_cost = swap_value * np.random.uniform(0, max_price_impact)
                
                # Dust loss (value that becomes "stuck" and unusable)
                dust_value = pre_rebalance_value * np.random.uniform(0, max_dust_threshold)
                
                # Total rebalance cost
                rebalance_cost = protocol_fee + slippage + price_impact_cost + dust_value
                rebalance_costs += rebalance_cost
                
                # Track total value lost during rebalances
                value_lost_on_rebalances += rebalance_cost
                
                # Apply rebalancing cost to position
                position_value = pre_rebalance_value - rebalance_cost
                
                # Reset position to 50/50 after accounting for the costs
                base_amount = position_value / 2 / current_price
                quote_amount = position_value / 2
                
                # After rebalancing, update initial_price for relative price calculation
                initial_price = current_price
                
                # Reset normalized tracking after rebalance
                normalized_price = 1.0
            else:
                # If no rebalance, update position value from pre-action value
                position_value = pre_action_value
            
            # Store position values
            position_values.append(position_value)
            position_base.append(base_amount * current_price)
            position_quote.append(quote_amount)
        
        # Store rebalance events for use in other methods
        self.rebalance_events = rebalance_events
        
        # Calculate final position value
        final_position_value = position_values[-1]
        
        # Calculate ROI and APR
        position_roi = (final_position_value / self.initial_amount - 1) * 100
        days_passed = len(self.data) - 1  # Actual trading days in the simulation
        
        # Properly annualize the return - compound the 30-day return to a full year
        # (1 + r)^(365/days) - 1
        position_apr = ((1 + position_roi / 100) ** (365 / days_passed) - 1) * 100
        
        # Calculate fee APR (annualized)
        fee_roi = (fees_earned / self.initial_amount) * 100
        fee_apr = ((1 + fee_roi / 100) ** (365 / days_passed) - 1) * 100
        
        # Calculate percent of time in range
        percent_in_range = sum(price_in_range) / len(price_in_range) * 100 if price_in_range else 0
        
        # Calculate HODL APR with proper annualization
        hodl_roi = (self.hodl_value / self.initial_amount - 1) * 100
        hodl_apr = ((1 + hodl_roi / 100) ** (365 / days_passed) - 1) * 100
        
        # Store results
        self.results = {
            "initial_investment": self.initial_amount,
            "final_position_value": final_position_value,
            "position_roi": position_roi,
            "position_apr": position_apr,
            "hodl_value": self.hodl_value,
            "hodl_roi": hodl_roi,
            "hodl_apr": hodl_apr,
            "performance_vs_hodl": (final_position_value / self.hodl_value - 1) * 100,
            "position_values": position_values,
            "position_base": position_base,
            "position_quote": position_quote,
            "price_in_range": price_in_range,
            "rebalance_count": rebalance_count,
            "fees_earned": fees_earned,
            "fee_apr": fee_apr,
            "rebalance_costs": rebalance_costs,
            "value_lost_on_rebalances": value_lost_on_rebalances,
            "impermanent_gain": (final_position_value / self.hodl_value - 1) * 100,
            "percent_in_range": percent_in_range,
            "max_dust_percent": self.max_dust_percent,
            "max_slippage": self.max_slippage,
            "swap_max_price_impact": self.swap_max_price_impact
        }
        
        return self.results
    
    def run_simulation(self) -> Dict:
        """Run the complete simulation"""
        try:
            print(f"\nFetching historical data for {self.pair}...")
            self.data = self.fetch_pool_historical_data()
            
            print(f"Calculating hodl strategy value...")
            self.calculate_hodl_value()
            
            print(f"Simulating vfat strategy with width={self.width}%, buffer={self.buffer}%, cutoff={self.cutoff}%...")
            self.simulate_vfat_strategy()
            
            print("\nSimulation Results:")
            self.print_results()
            
            return self.results
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.results
    
    def print_results(self):
        """
        Print the results of the simulation
        """
        if not self.results:
            print("No results available. Run simulation first.")
            return
        
        # Get results data
        initial_amount = self.results.get('initial_investment', 0)
        final_value = self.results.get('final_position_value', 0)
        roi = self.results.get('position_roi', 0)
        apr = self.results.get('position_apr', 0)
        hodl_value = self.results.get('hodl_value', 0)
        hodl_roi = self.results.get('hodl_roi', 0)
        hodl_apr = self.results.get('hodl_apr', 0)
        vs_hodl = self.results.get('performance_vs_hodl', 0)
        rebalance_count = self.results.get('rebalance_count', 0)
        fees_earned = self.results.get('fees_earned', 0)
        rebalance_costs = self.results.get('rebalance_costs', 0)
        percent_in_range = self.results.get('percent_in_range', 0)
        fee_apr = self.results.get('fee_apr', 0)
        
        # Calculate average days between rebalances
        days_passed = len(self.data) - 1
        avg_days_between_rebalances = days_passed / rebalance_count if rebalance_count > 0 else 0
        
        # Calculate cost per rebalance
        cost_per_rebalance = rebalance_costs / rebalance_count if rebalance_count > 0 else 0
        
        # Calculate net value (fees earned - rebalance costs)
        net_value = fees_earned - rebalance_costs
        
        # Calculate net APR (annualized)
        net_roi = (net_value / initial_amount) * 100
        net_apr = ((1 + net_roi / 100) ** (365 / days_passed) - 1) * 100 if days_passed > 0 else 0
        
        # Print results
        print("\nVFAT STRATEGY RESULTS")
        print("=" * 60)
        print(f"Pair:                        {self.pair}")
        print(f"Initial Investment:          ${initial_amount:.2f}")
        print(f"Final Position Value:        ${final_value:.2f}")
        print(f"ROI:                         {roi:.2f}%")
        print(f"APR (annualized):            {apr:.2f}%")
        print(f"HODL Value:                  ${hodl_value:.2f}")
        print(f"HODL ROI:                    {hodl_roi:.2f}%")
        print(f"HODL APR (annualized):       {hodl_apr:.2f}%")
        print(f"Performance vs HODL:         {vs_hodl:.2f}%")
        print("-" * 60)
        print(f"Strategy Parameters:")
        print(f"Width:                       {self.width:.1f}%")
        print(f"Buffer:                      {self.buffer:.1f}%")
        print(f"Cutoff:                      {self.cutoff:.1f}%")
        print(f"Max Dust:                    {self.max_dust_percent:.1f}%")
        print(f"Max Slippage:                {self.max_slippage:.1f}%")
        print(f"Max Price Impact:            {self.swap_max_price_impact:.1f}%")
        print("-" * 60)
        print(f"Metrics:")
        print(f"Percent Time In Range:       {percent_in_range:.2f}%")
        print(f"Total Fees Earned:           ${fees_earned:.2f}")
        print(f"Fee APR (annualized):        {fee_apr:.2f}%")
        print(f"Rebalance Count:             {rebalance_count}")
        print(f"Avg Days Between Rebalances: {avg_days_between_rebalances:.1f}")
        print(f"Total Rebalance Costs:       ${rebalance_costs:.2f}")
        print(f"Avg Cost Per Rebalance:      ${cost_per_rebalance:.2f}")
        print(f"Net Value (Fees - Costs):    ${net_value:.2f}")
        print(f"Net Value APR (annualized):  {net_apr:.2f}%")
        print("=" * 60)
    
    def plot_results(self, output_file: str = None) -> None:
        """Plot simulation results with additional pool data"""
        if self.data is None or self.results is None or 'position_values' not in self.results:
            print("No data available for plotting")
            return
        
        # Get daily metrics for plotting
        daily_metrics = self.get_daily_metrics()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
        
        # Plot price ratio
        ax1.plot(daily_metrics['timestamp'], daily_metrics['base_quote_ratio'], 
                 label=f"{self.base_token}/{self.quote_token} Price Ratio", color='blue')
        ax1.set_title(f"{self.pair} Price Ratio", fontsize=16)
        ax1.set_ylabel("Price Ratio", fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot position value and HODL value
        position_values = self.results.get('position_values', [])
        if len(position_values) > len(daily_metrics):
            position_values = position_values[:len(daily_metrics)]
        elif len(position_values) < len(daily_metrics):
            pad_value = position_values[-1] if position_values else self.initial_amount
            position_values = position_values + [pad_value] * (len(daily_metrics) - len(position_values))
        
        ax2.plot(daily_metrics['timestamp'], position_values, 
                 label=f"vfat Position Value", color='green', linewidth=2)
        
        # Calculate hodl value over time
        start_price = daily_metrics.iloc[0]['base_quote_ratio']
        hodl_values = []
        for i, row in daily_metrics.iterrows():
            current_price = row['base_quote_ratio']
            # Simple 50/50 allocation for HODL
            hodl_value = self.initial_amount * (0.5 + 0.5 * (current_price / start_price))
            hodl_values.append(hodl_value)
        
        ax2.plot(daily_metrics['timestamp'], hodl_values, 
                 label=f"HODL Value", color='red', linestyle='--', linewidth=2)
        ax2.axhline(y=self.initial_amount, color='gray', linestyle=':', label='Initial Investment')
        
        ax2.set_title(f"Position Value Comparison", fontsize=16)
        ax2.set_ylabel("USD Value", fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot volume, TVL and fees
        ax3.bar(daily_metrics['timestamp'], daily_metrics['fees_earned'], 
                label='Daily Fees Earned', color='purple', alpha=0.7)
        ax3.set_title(f"Daily Fees Earned", fontsize=16)
        ax3.set_ylabel("USD", fontsize=12)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Add secondary axis for TVL
        ax4 = ax3.twinx()
        if 'liquidity_usd' in daily_metrics:
            ax4.plot(daily_metrics['timestamp'], daily_metrics['liquidity_usd'], 
                     label='Pool TVL', color='orange', linestyle='-.')
            ax4.set_ylabel("Pool TVL (USD)", fontsize=12, color='orange')
            ax4.tick_params(axis='y', labelcolor='orange')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
        
        # Also plot cumulative fees
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_metrics['timestamp'], daily_metrics['cumulative_fees'], 
                label='Cumulative Fees Earned', color='green', linewidth=2)
        ax.set_title(f"Cumulative Fees Earned ({self.pair})", fontsize=16)
        ax.set_ylabel("USD", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if output_file:
            fee_output = output_file.replace('.png', '_fees.png')
            plt.savefig(fee_output, dpi=300)
            print(f"Fees plot saved to {fee_output}")
        else:
            plt.show()
        
        plt.close()

    def get_daily_metrics(self, export_file: Optional[str] = None) -> pd.DataFrame:
        """
        Extract daily metrics from simulation results
        
        Args:
            export_file: Optional path to export metrics as CSV
            
        Returns:
            pandas.DataFrame: DataFrame with daily metrics
        """
        if self.results is None or self.data is None:
            return pd.DataFrame()
        
        # Get daily data and position values
        daily_df = self.data
        position_values = self.results.get('position_values', [])
        
        # Ensure daily_df has a date column
        if 'date' not in daily_df.columns and 'timestamp' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['timestamp'])
        
        # Adjust position_values to match daily_df length
        if len(position_values) < len(daily_df):
            # If position_values is shorter, pad with the last value
            last_value = position_values[-1] if position_values else self.initial_amount
            position_values.extend([last_value] * (len(daily_df) - len(position_values)))
        elif len(position_values) > len(daily_df):
            # If position_values is longer, truncate
            position_values = position_values[:len(daily_df)]
        
        # Create daily metrics dataframe
        metrics_df = daily_df.copy()
        metrics_df['position_value'] = position_values
        
        # Calculate the fee rate from actual pool data
        fee_rate = FEE_TIERS.get(self.pair, 0.003)
        
        # Track if price is in range for each day
        price_in_range = []
        
        # Get the lower and upper price bounds for the position
        for _, row in daily_df.iterrows():
            current_price = row['base_quote_ratio']
            # Check if we're in range using actual price data
            in_range = self.is_price_in_range(current_price, row['date'])
            price_in_range.append(in_range)
        
        metrics_df['price_in_range'] = price_in_range
        
        # Calculate fees earned based on volume
        fees_earned = []
        swap_volume = []
        
        for i, row in metrics_df.iterrows():
            # Get daily volume directly
            daily_volume = row['volume_usd']
            position_value = row['position_value']
            
            # Only earn fees when price is in range and there's volume
            if row['price_in_range'] and daily_volume > 0:
                # Use actual pool liquidity for calculating share
                pool_share = position_value / self.real_liquidity
                
                daily_fee = daily_volume * fee_rate * pool_share
                fees_earned.append(daily_fee)
                swap_volume.append(daily_volume)
            else:
                fees_earned.append(0)
                swap_volume.append(0)
        
        metrics_df['fees_earned'] = fees_earned
        metrics_df['swap_volume'] = swap_volume
        metrics_df['cumulative_fees'] = np.cumsum(fees_earned)
        
        # Calculate TVL share using the actual pool liquidity
        tvl_share = []
        for i, row in metrics_df.iterrows():
            tvl_share.append(row['position_value'] / self.real_liquidity)
        
        metrics_df['tvl_share'] = tvl_share
        
        # Add the real liquidity to all days
        metrics_df['liquidity_usd'] = self.real_liquidity
        
        # Track rebalance events if available
        if hasattr(self, 'rebalance_events') and self.rebalance_events:
            rebalanced = []
            
            # Handle integer-based rebalance events (day indices)
            if all(isinstance(event, (int, np.integer)) for event in self.rebalance_events):
                # Convert day indices to dates
                rebalance_dates = set()
                for day_index in self.rebalance_events:
                    if 0 <= day_index < len(self.data):
                        rebalance_date = self.data.iloc[day_index]['date'].date() if 'date' in self.data.columns else None
                        if rebalance_date:
                            rebalance_dates.add(rebalance_date)
            # Handle dictionary-based rebalance events
            else:
                try:
                    rebalance_dates = set(event['date'].date() for event in self.rebalance_events if isinstance(event, dict) and 'date' in event)
                except (TypeError, AttributeError):
                    # Fallback if there's an issue with the format
                    rebalance_dates = set()
                    
            # Mark days with rebalances
            for date in metrics_df['date']:
                if isinstance(date, pd.Timestamp):
                    rebalanced.append(date.date() in rebalance_dates)
                else:
                    rebalanced.append(False)
            
            metrics_df['rebalanced'] = rebalanced
        
        # Export if requested
        if export_file:
            metrics_df.to_csv(export_file, index=False)
        
        return metrics_df
    
    def export_detailed_metrics(self, output_file: str = None) -> None:
        """
        Export detailed daily metrics to a CSV file
        
        Args:
            output_file: Path to save the CSV file (default: auto-generated based on pair and parameters)
        """
        # Get daily metrics
        daily_metrics = self.get_daily_metrics()
        
        if daily_metrics.empty:
            print("No daily metrics available for export")
            return
        
        # Generate default filename if not provided
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"vfat_detailed_metrics_{self.pair.replace('/', '_')}_w{self.width}_b{self.buffer}_c{self.cutoff}_{timestamp}.csv"
        
        # Save to CSV
        daily_metrics.to_csv(output_file, index=False)
        print(f"Detailed metrics saved to {output_file}")
        
        return daily_metrics

    def is_price_in_range(self, current_price: float, date: Optional[datetime] = None) -> bool:
        """Check if the current price is within the position's range
        
        Args:
            current_price: Current price to check
            date: Optional date to determine which price range applies (for rebalances)
            
        Returns:
            bool: True if price is in range, False otherwise
        """
        # Get current data and results
        if not hasattr(self, 'results') or self.results is None:
            return False
        
        # Find the initial price for range calculation
        if hasattr(self, 'data') and not self.data.empty:
            initial_price = self.data.iloc[0]['base_quote_ratio']
        else:
            return False
        
        # Calculate normalized price
        normalized_price = current_price / initial_price
        
        # Get width from parameters
        width_factor = self.width / 100
        
        # Calculate default range
        lower_bound = 1 - width_factor
        upper_bound = 1 + width_factor
        
        # Handle rebalance events if date is provided
        if date is not None and hasattr(self, 'rebalance_events') and self.rebalance_events:
            # Convert rebalance_events to dates for comparison
            rebalance_dates = []
            
            # Try to extract dates from both list of dicts and list of date indices
            for event in self.rebalance_events:
                if isinstance(event, dict) and 'date' in event:
                    rebalance_dates.append((event['date'], event))
                elif isinstance(event, (int, np.integer)):
                    # Get the date from data DataFrame using index, with bounds checking
                    if 0 <= event < len(self.data):
                        try:
                            event_date = self.data.iloc[event]['date']
                            rebalance_dates.append((event_date, event))
                        except (KeyError, IndexError):
                            # Skip if the date column is missing or there's an index error
                            continue
                elif isinstance(event, (pd.Timestamp, datetime)):
                    rebalance_dates.append((event, event))
            
            # Sort by date
            rebalance_dates.sort(key=lambda x: x[0])
            
            # Find the most recent rebalance before the given date
            latest_rebalance = None
            latest_date = None
            
            for rebalance_date, event in rebalance_dates:
                if rebalance_date <= date:
                    latest_rebalance = event
                    latest_date = rebalance_date
            
            # If we found a rebalance, adjust the initial price to reset normalization
            if latest_rebalance is not None:
                # Get the price at the time of rebalance
                if isinstance(latest_rebalance, dict) and 'price' in latest_rebalance:
                    initial_price = latest_rebalance['price']
                elif isinstance(latest_rebalance, (int, np.integer)) and 0 <= latest_rebalance < len(self.data):
                    try:
                        initial_price = self.data.iloc[latest_rebalance]['base_quote_ratio']
                    except (KeyError, IndexError):
                        # Keep using the default initial price if there's an error
                        pass
                
                # Recalculate normalized price
                normalized_price = current_price / initial_price
        
        # Check if price is in range
        return lower_bound <= normalized_price <= upper_bound


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='vfat Strategy Simulator')
    
    # Required parameters
    parser.add_argument('--pair', type=str, choices=list(POOL_ADDRESS.keys()), 
                        default="USDC.e/scUSD", help='Trading pair to simulate')
    parser.add_argument('--amount', type=float, default=1000.0,
                        help='Initial investment amount in USD')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest')
    
    # vfat parameters
    parser.add_argument('--width', type=float, default=1.0,
                        help='Position width in percentage (e.g., 1.0 means ±0.5%)')
    parser.add_argument('--buffer', type=float, default=0.2,
                        help='Buffer percentage (e.g., 0.2 means 0.2% buffer)')
    parser.add_argument('--cutoff', type=float, default=5.0,
                        help='Cutoff percentage for extreme price moves (e.g., 5.0 means 5%)')
    
    # Output options
    parser.add_argument('--plot', action='store_true', help='Generate performance plot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for the plot')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        # Print simulation setup
        print("\nVFAT STRATEGY SIMULATOR")
        print("=" * 60)
        print(f"Pair: {args.pair}")
        print(f"Initial Investment: ${args.amount:.2f}")
        print(f"Backtest Period: {args.days} days")
        print(f"vfat Parameters:")
        print(f"  - Width: {args.width}% (±{args.width/2}% from current price)")
        print(f"  - Buffer: {args.buffer}%")
        print(f"  - Cutoff: {args.cutoff}%")
        print("=" * 60)
        
        # Confirm if user wants to proceed
        if not args.output:  # Only prompt for confirmation in interactive mode
            confirm = input("Proceed with simulation? (y/n): ").lower()
            if confirm != 'y':
                print("Simulation aborted")
                return 0
        
        # Run simulation
        simulator = VfatSimulator(
            pair=args.pair,
            initial_amount=args.amount,
            backtest_days=args.days,
            width=args.width,
            buffer=args.buffer,
            cutoff=args.cutoff
        )
        
        results = simulator.run_simulation()
        
        # Generate plot if requested
        if args.plot or args.output:
            output_file = args.output or f"vfat_simulation_{args.pair.replace('/', '_')}_{int(time.time())}.png"
            simulator.plot_results(output_file)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 