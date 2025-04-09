"""
Price Data Collection Module

This module handles fetching and processing historical price data
for USD* and related tokens from various API sources.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# Set up logger
logger = logging.getLogger(__name__)

# Import config
try:
    import config as cfg
except ImportError:
    # If called directly, try relative import
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    import config as cfg

def fetch_historical_prices(
    pool: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Fetch historical price data for the tokens in the specified pool.
    
    Args:
        pool (str): Pool identifier (e.g., "CL5-USDC.e/scUSD")
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        interval (str): Data interval (1h, 4h, 1d)
        
    Returns:
        pd.DataFrame: DataFrame with historical price data
    """
    logger.info(f"Fetching historical prices for {pool} from {start_date} to {end_date}")
    
    # Get pool configuration
    pool_config = cfg.POOL_ADDRESSES.get(pool)
    if not pool_config:
        raise ValueError(f"Pool {pool} not found in configuration")
    
    # Parse token pair from pool name
    tokens = pool.split("-")[1].split("/")
    if len(tokens) != 2:
        raise ValueError(f"Invalid pool format: {pool}")
    
    token1, token2 = tokens
    
    # Check cache first
    cached_data = _check_price_cache(pool, start_date, end_date, interval)
    if cached_data is not None:
        logger.info(f"Using cached price data for {pool}")
        return cached_data
    
    # Try different data sources in order of preference
    data = None
    
    # Try DexScreener
    try:
        data = _fetch_from_dexscreener(pool_config["pool"], start_date, end_date, interval)
    except Exception as e:
        logger.warning(f"Error fetching from DexScreener: {e}")
    
    # If DexScreener failed, try CoinGecko
    if data is None:
        try:
            data = _fetch_from_coingecko(token1, token2, start_date, end_date, interval)
        except Exception as e:
            logger.warning(f"Error fetching from CoinGecko: {e}")
    
    # If all API sources failed, try to use on-chain data via Sonicscan
    if data is None:
        try:
            data = _fetch_from_sonicscan(pool_config["pool"], start_date, end_date)
        except Exception as e:
            logger.warning(f"Error fetching from Sonicscan: {e}")
            raise ValueError(f"Failed to fetch price data from all sources for {pool}")
    
    # Cache the data for future use
    _cache_price_data(pool, data, start_date, end_date, interval)
    
    return data

def _check_price_cache(
    pool: str,
    start_date: datetime,
    end_date: datetime,
    interval: str
) -> Optional[pd.DataFrame]:
    """Check if we have cached price data for this request."""
    cache_dir = cfg.DATA_PATHS["cache_dir"]
    cache_file = os.path.join(
        cache_dir, 
        f"prices_{pool.replace('/', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.csv"
    )
    
    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            return data
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")
    
    return None

def _cache_price_data(
    pool: str,
    data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    interval: str
) -> None:
    """Cache price data for future use."""
    cache_dir = cfg.DATA_PATHS["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(
        cache_dir, 
        f"prices_{pool.replace('/', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.csv"
    )
    
    try:
        data.to_csv(cache_file, index=False)
        logger.info(f"Cached price data to {cache_file}")
    except Exception as e:
        logger.warning(f"Error caching price data: {e}")

def _fetch_from_dexscreener(
    pool_address: str,
    start_date: datetime,
    end_date: datetime,
    interval: str
) -> Optional[pd.DataFrame]:
    """Fetch historical price data from DexScreener API."""
    logger.info(f"Fetching price data from DexScreener for pool {pool_address}")
    
    # Convert interval to DexScreener format
    interval_mapping = {
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"
    }
    dex_interval = interval_mapping.get(interval, "1h")
    
    # Calculate Unix timestamps
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    # DexScreener API endpoint
    url = f"{cfg.API_ENDPOINTS['dexscreener']}/charts/sonic/{pool_address}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            logger.warning(f"DexScreener API returned status code {response.status_code}")
            return None
        
        data = response.json()
        
        # Check if we have price data
        if 'pairs' not in data or not data['pairs'] or 'priceUsd' not in data['pairs'][0]:
            logger.warning("No price data found in DexScreener response")
            return None
        
        # Extract price data
        pair_data = data['pairs'][0]
        price_data = pair_data['priceUsd']
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        
        # Filter by date range
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    except Exception as e:
        logger.warning(f"Error fetching from DexScreener: {e}")
        return None

def _fetch_from_coingecko(
    token1: str,
    token2: str,
    start_date: datetime,
    end_date: datetime,
    interval: str
) -> Optional[pd.DataFrame]:
    """Fetch historical price data from CoinGecko API."""
    logger.info(f"Fetching price data from CoinGecko for {token1} and {token2}")
    
    # Map token symbols to CoinGecko IDs
    token_mapping = {
        "USDC.e": "sonic-bridged-usdc-e-sonic",
        "scUSD": "scusd-shadow",
        "USDT": "tether",
        "S": "wrapped-sonic",
        "x33": "shadow-2"
    }
    
    token1_id = token_mapping.get(token1)
    token2_id = token_mapping.get(token2)
    
    if not token1_id or not token2_id:
        logger.warning(f"Could not map tokens {token1} or {token2} to CoinGecko IDs")
        return None
    
    # Calculate Unix timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    
    # CoinGecko API has rate limits, so we need to be careful
    # Fetch data for each token
    data1 = _fetch_coingecko_price(token1_id, start_ts, end_ts)
    time.sleep(1.5)  # Avoid rate limiting
    data2 = _fetch_coingecko_price(token2_id, start_ts, end_ts)
    
    if data1 is None or data2 is None:
        return None
    
    # Merge data
    df1 = pd.DataFrame(data1, columns=['timestamp', f'{token1}_price'])
    df2 = pd.DataFrame(data2, columns=['timestamp', f'{token2}_price'])
    
    df = pd.merge(df1, df2, on='timestamp', how='outer')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Calculate the price ratio (token1/token2)
    df['price'] = df[f'{token1}_price'] / df[f'{token2}_price']
    
    # Resample to desired interval if needed
    if interval == '1h' and len(df) > 24:
        df = df.set_index('timestamp').resample('1H').mean().reset_index()
    elif interval == '4h' and len(df) > 6:
        df = df.set_index('timestamp').resample('4H').mean().reset_index()
    
    return df

def _fetch_coingecko_price(
    token_id: str,
    start_ts: int,
    end_ts: int
) -> Optional[List[List[Union[int, float]]]]:
    """Fetch historical price data for a single token from CoinGecko."""
    url = f"{cfg.API_ENDPOINTS['coingecko']}/coins/{token_id}/market_chart/range"
    
    params = {
        'vs_currency': 'usd',
        'from': start_ts,
        'to': end_ts,
        'x_cg_demo_api_key': cfg.API_KEYS['coingecko']
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.warning(f"CoinGecko API returned status code {response.status_code}")
            return None
        
        data = response.json()
        
        if 'prices' not in data or not data['prices']:
            logger.warning(f"No price data found for {token_id}")
            return None
        
        return data['prices']
    
    except Exception as e:
        logger.warning(f"Error fetching from CoinGecko: {e}")
        return None

def _fetch_from_sonicscan(
    pool_address: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """Fetch historical price data from Sonicscan API."""
    logger.info(f"Fetching price data from Sonicscan for pool {pool_address}")
    
    # Calculate block numbers (approximate)
    # This is a simplified approach - in a real implementation, we would query the actual blocks
    avg_block_time = 1.5  # seconds per block (approx)
    current_time = datetime.now().timestamp()
    current_block = 15000000  # Approximate current block number - should be queried dynamically
    
    seconds_per_day = 86400
    blocks_per_day = seconds_per_day / avg_block_time
    
    days_from_end = (current_time - end_date.timestamp()) / seconds_per_day
    days_from_start = (current_time - start_date.timestamp()) / seconds_per_day
    
    end_block = int(current_block - (days_from_end * blocks_per_day))
    start_block = int(current_block - (days_from_start * blocks_per_day))
    
    # Sonicscan API endpoint
    url = f"{cfg.API_ENDPOINTS['sonicscan']}"
    
    # Parameters for the API request
    params = {
        "module": "account",
        "action": "tokentx",
        "address": pool_address,
        "startblock": start_block,
        "endblock": end_block,
        "sort": "asc",
        "apikey": cfg.API_KEYS["sonicscan"]
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.warning(f"Sonicscan API returned status code {response.status_code}")
            return None
        
        data = response.json()
        
        if 'result' not in data or not data['result']:
            logger.warning("No transaction data found in Sonicscan response")
            return None
        
        # Process the transaction data to estimate prices
        # This is a complex process and would require analyzing swap events
        # For this example, we'll return a simplified dataframe
        
        transactions = data['result']
        processed_data = []
        
        for tx in transactions:
            timestamp = datetime.fromtimestamp(int(tx['timeStamp']))
            if 'value' in tx and tx['tokenSymbol']:
                processed_data.append({
                    'timestamp': timestamp,
                    'token': tx['tokenSymbol'],
                    'value': float(tx['value']) / (10 ** int(tx['tokenDecimal']))
                })
        
        if not processed_data:
            logger.warning("No usable price data found in Sonicscan transactions")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # This would need additional processing to derive actual price data
        # For now, we're returning a simplified version
        
        return df
    
    except Exception as e:
        logger.warning(f"Error fetching from Sonicscan: {e}")
        return None

if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    try:
        data = fetch_historical_prices(
            pool="CL5-USDC.e/scUSD",
            start_date=start_date,
            end_date=end_date,
            interval="1h"
        )
        
        print(f"Fetched {len(data)} data points")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}") 