"""
Pool Data Collection Module

This module handles fetching and processing historical pool configuration data
including width settings, buffer margins, and cutoff values.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

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

def fetch_pool_data(
    pool: str,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    Fetch historical pool configuration and performance data.
    
    Args:
        pool (str): Pool identifier (e.g., "CL5-USDC.e/scUSD")
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        Dict[str, Any]: Dictionary with pool data including:
            - width_history: Historical position width settings
            - buffer_history: Historical buffer margin settings
            - cutoff_history: Historical cutoff values
            - volume_history: Historical trading volume
            - fee_history: Historical fee rates
            - tvl_history: Historical total value locked
    """
    logger.info(f"Fetching pool data for {pool} from {start_date} to {end_date}")
    
    # Get pool configuration
    pool_config = cfg.POOL_ADDRESSES.get(pool)
    if not pool_config:
        raise ValueError(f"Pool {pool} not found in configuration")
    
    # Check cache first
    cached_data = _check_pool_cache(pool, start_date, end_date)
    if cached_data is not None:
        logger.info(f"Using cached pool data for {pool}")
        return cached_data
    
    # Collect data from different sources
    pool_data = {}
    
    # Fetch pool configuration history (width, buffer, cutoff)
    try:
        config_history = _fetch_pool_config_history(pool_config, start_date, end_date)
        if config_history:
            pool_data.update(config_history)
    except Exception as e:
        logger.warning(f"Error fetching pool configuration history: {e}")
        # Use default/fallback values
        pool_data["width_history"] = _generate_default_width_history(start_date, end_date)
        pool_data["buffer_history"] = _generate_default_buffer_history(start_date, end_date)
        pool_data["cutoff_history"] = _generate_default_cutoff_history(start_date, end_date)
    
    # Fetch volume history
    try:
        volume_history = _fetch_volume_history(pool_config, start_date, end_date)
        if volume_history is not None:
            pool_data["volume_history"] = volume_history
    except Exception as e:
        logger.warning(f"Error fetching volume history: {e}")
        pool_data["volume_history"] = _generate_default_volume_history(start_date, end_date)
    
    # Fetch fee history
    try:
        fee_history = _fetch_fee_history(pool_config, start_date, end_date)
        if fee_history is not None:
            pool_data["fee_history"] = fee_history
    except Exception as e:
        logger.warning(f"Error fetching fee history: {e}")
        pool_data["fee_history"] = _generate_default_fee_history(start_date, end_date)
    
    # Fetch TVL history
    try:
        tvl_history = _fetch_tvl_history(pool_config, start_date, end_date)
        if tvl_history is not None:
            pool_data["tvl_history"] = tvl_history
    except Exception as e:
        logger.warning(f"Error fetching TVL history: {e}")
        pool_data["tvl_history"] = _generate_default_tvl_history(start_date, end_date)
    
    # Cache the data for future use
    _cache_pool_data(pool, pool_data, start_date, end_date)
    
    return pool_data

def _check_pool_cache(
    pool: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[Dict[str, Any]]:
    """Check if we have cached pool data for this request."""
    cache_dir = cfg.DATA_PATHS["cache_dir"]
    cache_file = os.path.join(
        cache_dir, 
        f"pool_{pool.replace('/', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    )
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Convert string timestamps back to datetime for DataFrames
            for key in ['width_history', 'buffer_history', 'cutoff_history', 
                       'volume_history', 'fee_history', 'tvl_history']:
                if key in data and isinstance(data[key], dict) and 'data' in data[key]:
                    df = pd.DataFrame(data[key]['data'])
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    data[key] = df
            
            return data
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")
    
    return None

def _cache_pool_data(
    pool: str,
    data: Dict[str, Any],
    start_date: datetime,
    end_date: datetime
) -> None:
    """Cache pool data for future use."""
    cache_dir = cfg.DATA_PATHS["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(
        cache_dir, 
        f"pool_{pool.replace('/', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    )
    
    try:
        # Convert DataFrames to serializable format
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[key] = {
                    'type': 'dataframe',
                    'data': value.to_dict(orient='records')
                }
            else:
                serializable_data[key] = value
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_data, f)
        
        logger.info(f"Cached pool data to {cache_file}")
    except Exception as e:
        logger.warning(f"Error caching pool data: {e}")

def _fetch_pool_config_history(
    pool_config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical pool configuration data (width, buffer, cutoff).
    
    In a real implementation, this would query historical configuration changes
    from the blockchain or a service like vfat.io.
    """
    logger.info(f"Fetching pool configuration history")
    
    # For this example, we'll use dummy data
    # In a real implementation, we would fetch this from an API or blockchain
    
    # Generate some sample data spanning the time range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Width history (simulate some changes over time)
    width_data = []
    current_width = 0.05  # Start with 0.05%
    
    for i, date in enumerate(date_range):
        # Change width occasionally
        if i > 0 and i % 5 == 0:
            # Alternate between a few common values
            widths = [0.01, 0.05, 0.1, 0.3, 0.5]
            current_width = widths[i % len(widths)]
        
        width_data.append({
            'timestamp': date,
            'width': current_width
        })
    
    # Buffer history
    buffer_data = []
    current_buffer_lower = 0.1
    current_buffer_upper = 0.1
    
    for i, date in enumerate(date_range):
        # Change buffer occasionally
        if i > 0 and i % 7 == 0:
            # Simulate different buffer configurations
            current_buffer_lower = round(0.1 * (i % 5), 1)
            current_buffer_upper = round(0.1 * ((i + 1) % 5), 1)
        
        buffer_data.append({
            'timestamp': date,
            'buffer_lower': current_buffer_lower,
            'buffer_upper': current_buffer_upper
        })
    
    # Cutoff history
    cutoff_data = []
    current_cutoff_lower = 0.98
    current_cutoff_upper = 1.02
    
    for i, date in enumerate(date_range):
        # Change cutoffs occasionally
        if i > 0 and i % 10 == 0:
            # Simulate different cutoff configurations
            current_cutoff_lower = round(0.95 + (0.01 * (i % 5)), 2)
            current_cutoff_upper = round(1.05 - (0.01 * (i % 5)), 2)
        
        cutoff_data.append({
            'timestamp': date,
            'cutoff_lower': current_cutoff_lower,
            'cutoff_upper': current_cutoff_upper
        })
    
    # Convert to DataFrames
    width_df = pd.DataFrame(width_data)
    buffer_df = pd.DataFrame(buffer_data)
    cutoff_df = pd.DataFrame(cutoff_data)
    
    return {
        'width_history': width_df,
        'buffer_history': buffer_df,
        'cutoff_history': cutoff_df
    }

def _fetch_volume_history(
    pool_config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical trading volume data for the pool.
    
    In a real implementation, this would query volume data
    from DexScreener or a similar source.
    """
    logger.info(f"Fetching volume history")
    
    # Try to fetch from DexScreener first
    try:
        volume_data = _fetch_volume_from_dexscreener(pool_config["pool"], start_date, end_date)
        if volume_data is not None:
            return volume_data
    except Exception as e:
        logger.warning(f"Error fetching volume from DexScreener: {e}")
    
    # Fallback to generating sample data
    return _generate_default_volume_history(start_date, end_date)

def _fetch_volume_from_dexscreener(
    pool_address: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """Fetch volume data from DexScreener."""
    # DexScreener API endpoint
    url = f"{cfg.API_ENDPOINTS['dexscreener']}/charts/sonic/{pool_address}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            logger.warning(f"DexScreener API returned status code {response.status_code}")
            return None
        
        data = response.json()
        
        # Check if we have volume data
        if 'pairs' not in data or not data['pairs'] or 'volumeUsd' not in data['pairs'][0]:
            logger.warning("No volume data found in DexScreener response")
            return None
        
        # Extract volume data
        pair_data = data['pairs'][0]
        volume_data = pair_data['volumeUsd']
        
        # Convert to DataFrame
        df = pd.DataFrame(volume_data)
        
        # Calculate Unix timestamps for filtering
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        # Filter by date range
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    except Exception as e:
        logger.warning(f"Error fetching volume from DexScreener: {e}")
        return None

def _fetch_fee_history(
    pool_config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical fee rate data for the pool.
    
    In a real implementation, this would query fee data
    from vfat.io or the blockchain.
    """
    logger.info(f"Fetching fee history")
    
    # Generate sample data
    return _generate_default_fee_history(start_date, end_date)

def _fetch_tvl_history(
    pool_config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical total value locked (TVL) data for the pool.
    
    In a real implementation, this would query TVL data
    from DexScreener or a similar source.
    """
    logger.info(f"Fetching TVL history")
    
    # Generate sample data
    return _generate_default_tvl_history(start_date, end_date)

def _generate_default_width_history(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate default width history if actual data is unavailable."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = [{'timestamp': date, 'width': 0.05} for date in date_range]
    return pd.DataFrame(data)

def _generate_default_buffer_history(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate default buffer history if actual data is unavailable."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = [{'timestamp': date, 'buffer_lower': 0.1, 'buffer_upper': 0.1} for date in date_range]
    return pd.DataFrame(data)

def _generate_default_cutoff_history(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate default cutoff history if actual data is unavailable."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = [{'timestamp': date, 'cutoff_lower': 0.98, 'cutoff_upper': 1.02} for date in date_range]
    return pd.DataFrame(data)

def _generate_default_volume_history(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate default volume history if actual data is unavailable."""
    import numpy as np
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate some random but realistic looking volume data
    base_volume = 100000  # Base volume in USD
    data = []
    
    for date in date_range:
        # Add daily and weekly patterns
        hour_factor = 1.0 + 0.5 * np.sin(date.hour * np.pi / 12)
        day_factor = 1.0 + 0.3 * np.sin(date.dayofweek * np.pi / 3.5)
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.2)
        
        volume = base_volume * hour_factor * day_factor * random_factor
        
        data.append({
            'timestamp': date,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def _generate_default_fee_history(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate default fee history if actual data is unavailable."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Start with a base fee
    base_fee = 0.003  # 0.3%
    data = []
    
    for i, date in enumerate(date_range):
        # Occasionally change the fee
        if i > 0 and i % 15 == 0:
            base_fee = round(0.001 * (1 + (i % 5)), 3)
        
        data.append({
            'timestamp': date,
            'fee_rate': base_fee
        })
    
    return pd.DataFrame(data)

def _generate_default_tvl_history(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate default TVL history if actual data is unavailable."""
    import numpy as np
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate some random but realistic looking TVL data
    base_tvl = 5000000  # Base TVL in USD
    data = []
    
    current_tvl = base_tvl
    
    for i, date in enumerate(date_range):
        # Add some randomness with a slight trend
        trend_factor = 1.0 + (0.002 * np.sin(i * np.pi / 7))
        random_factor = np.random.normal(1.0, 0.05)
        
        current_tvl = current_tvl * trend_factor * random_factor
        
        data.append({
            'timestamp': date,
            'tvl': current_tvl
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    try:
        data = fetch_pool_data(
            pool="CL5-USDC.e/scUSD",
            start_date=start_date,
            end_date=end_date
        )
        
        print("Pool data fetched successfully")
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print(f"\n{key}: {len(value)} entries")
                print(value.head())
            else:
                print(f"\n{key}: {value}")
    except Exception as e:
        print(f"Error: {e}") 