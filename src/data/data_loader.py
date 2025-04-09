"""
Data Loader Module

This module handles loading and preprocessing price and pool data for the
USD* pool backtesting engine.
"""

import os
import logging
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import requests
from time import sleep

# Set up logger
logger = logging.getLogger(__name__)

# Import local modules
try:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from utils.helpers import parse_date, date_range, interpolate_missing_values
    from utils.coingecko import get_price_data as fetch_coingecko_price
    import config as cfg
except ImportError:
    # If called directly, try relative import
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from utils.helpers import parse_date, date_range, interpolate_missing_values
    from utils.coingecko import get_price_data as fetch_coingecko_price
    import config as cfg


class DataLoader:
    """
    Load and preprocess price and pool data for backtesting.
    """
    
    def __init__(self, use_cache: bool = True, use_coingecko: bool = True):
        """
        Initialize the data loader.
        
        Args:
            use_cache (bool): Whether to use cached data
            use_coingecko (bool): Whether to use CoinGecko API for price data
        """
        # Get configuration
        config = cfg.get_config()
        
        # Set up cache settings
        self.use_cache = use_cache and config["cache"]["enabled"]
        self.cache_expiry_days = config["cache"]["expiry_days"]
        self.price_data_cache = config["cache"]["price_data_cache"]
        self.pool_data_cache = config["cache"]["pool_data_cache"]
        
        # Set up API configuration
        self.api_config = config["api_config"]
        
        # Set up default pool IDs
        self.default_pool_ids = config["default_pool_ids"]
        
        # Whether to use CoinGecko API
        self.use_coingecko = use_coingecko
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.price_data_cache), exist_ok=True)
        
        logger.info("Initialized DataLoader with cache " + 
                   ("enabled" if self.use_cache else "disabled") +
                   " and CoinGecko " +
                   ("enabled" if self.use_coingecko else "disabled"))
    
    def load_price_data(
        self,
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        token_pair: str = "USD*/USDC",
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Load historical price data for a token pair.
        
        Args:
            start_date (Union[str, datetime], optional): Start date
            end_date (Union[str, datetime], optional): End date
            token_pair (str, optional): Token pair (e.g., "USD*/USDC")
            interval (str, optional): Time interval (e.g., "1h", "1d")
            
        Returns:
            pd.DataFrame: DataFrame with timestamp and price columns
        """
        # Get default dates if not provided
        config = cfg.get_config()
        if start_date is None:
            start_date = config["dates"]["default_start_date"]
        if end_date is None:
            end_date = config["dates"]["default_end_date"]
        
        # Parse dates if needed
        if isinstance(start_date, str):
            start_date = parse_date(start_date)
        if isinstance(end_date, str):
            end_date = parse_date(end_date)
        
        # Create cache key
        cache_key = f"{token_pair}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Check cache first if enabled
        if self.use_cache:
            cached_data = self._load_from_cache(
                self.price_data_cache, 
                cache_key=cache_key,
                token_pair=token_pair, 
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if cached_data is not None:
                logger.info(f"Loaded price data for {token_pair} from cache")
                return cached_data
        
        # Fetch data from API or generate synthetic data
        logger.info(f"Fetching price data for {token_pair} from {start_date} to {end_date}")
        
        try:
            if self.use_coingecko:
                # Use CoinGecko API to fetch real price data
                price_data = self._fetch_from_coingecko(
                    token_pair, start_date, end_date, interval)
                
                # If we got data from CoinGecko, clean it
                if not price_data.empty:
                    logger.info(f"Successfully fetched {len(price_data)} price points from CoinGecko")
                    
                    # Ensure price data is sorted
                    price_data = price_data.sort_values('timestamp').reset_index(drop=True)
                    
                    # Interpolate missing values if needed
                    price_data = interpolate_missing_values(price_data)
                else:
                    logger.warning(f"No data returned from CoinGecko. Falling back to synthetic data.")
                    # Fall back to synthetic data
                    price_data = self._generate_synthetic_price_data(
                        start_date, end_date, interval, token_pair)
            else:
                # Generate synthetic data
                price_data = self._generate_synthetic_price_data(
                    start_date, end_date, interval, token_pair)
            
            # Cache the data if caching is enabled
            if self.use_cache and not price_data.empty:
                self._save_to_cache(
                    price_data, 
                    self.price_data_cache, 
                    cache_key=cache_key,
                    token_pair=token_pair, 
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=["timestamp", "price"])
    
    def load_pool_data(
        self,
        pool_id: str = None,
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None
    ) -> Dict[str, Any]:
        """
        Load pool data including volume, TVL, and fee history.
        
        Args:
            pool_id (str, optional): Pool ID to load data for
            start_date (Union[str, datetime], optional): Start date
            end_date (Union[str, datetime], optional): End date
            
        Returns:
            Dict[str, Any]: Dictionary with pool data
        """
        # Get default pool if not provided
        if pool_id is None and self.default_pool_ids:
            pool_id = self.default_pool_ids[0]
        
        # Get default dates if not provided
        config = cfg.get_config()
        if start_date is None:
            start_date = config["dates"]["default_start_date"]
        if end_date is None:
            end_date = config["dates"]["default_end_date"]
            
        # Parse dates if needed
        if isinstance(start_date, str):
            start_date = parse_date(start_date)
        if isinstance(end_date, str):
            end_date = parse_date(end_date)
        
        # Create cache key
        cache_key = f"{pool_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Check cache first if enabled
        if self.use_cache:
            cached_data = self._load_from_cache(
                self.pool_data_cache, 
                cache_key=cache_key,
                pool_id=pool_id, 
                start_date=start_date,
                end_date=end_date
            )
            
            if cached_data is not None:
                logger.info(f"Loaded pool data for {pool_id} from cache")
                return cached_data
        
        # Fetch data from API or generate synthetic data for testing
        logger.info(f"Fetching pool data for {pool_id} from {start_date} to {end_date}")
        
        try:
            # TODO: Implement actual API call for pool data
            # For now, always generate synthetic data
            pool_data = self._generate_synthetic_pool_data(
                start_date, end_date, pool_id)
            
            # Cache the data if caching is enabled
            if self.use_cache:
                self._save_to_cache(
                    pool_data, 
                    self.pool_data_cache, 
                    cache_key=cache_key,
                    pool_id=pool_id, 
                    start_date=start_date,
                    end_date=end_date
                )
            
            return pool_data
            
        except Exception as e:
            logger.error(f"Error loading pool data: {e}")
            # Return empty data as fallback
            return {
                'volume_history': pd.DataFrame(columns=['timestamp', 'volume']),
                'fee_history': pd.DataFrame(columns=['timestamp', 'fee_rate']),
                'tvl_history': pd.DataFrame(columns=['timestamp', 'tvl'])
            }
    
    def _fetch_from_coingecko(
        self, 
        token_pair: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch price data from CoinGecko API.
        
        Args:
            token_pair (str): Token pair (e.g., "USD*/USDC")
            start_date (datetime): Start date
            end_date (datetime): End date
            interval (str): Time interval
            
        Returns:
            pd.DataFrame: DataFrame with timestamp and price columns
        """
        try:
            # Convert dates to strings for the API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Call the CoinGecko API wrapper
            df = fetch_coingecko_price(
                token_pair=token_pair,
                start_date=start_str,
                end_date=end_str,
                interval=interval
            )
            
            # If we have data, return it
            if not df.empty:
                return df
            
            # Otherwise, log an error and return empty DataFrame
            logger.error(f"CoinGecko API returned no data for {token_pair}")
            return pd.DataFrame(columns=["timestamp", "price"])
            
        except Exception as e:
            logger.error(f"Error fetching data from CoinGecko: {e}")
            return pd.DataFrame(columns=["timestamp", "price"])
    
    def _load_from_cache(
        self,
        cache_file: str,
        cache_key: str,
        **kwargs
    ) -> Optional[Any]:
        """
        Load data from cache if it exists and is not expired.
        
        Args:
            cache_file (str): Path to the cache file
            cache_key (str): Unique key for this cache entry
            **kwargs: Additional cache metadata to check
            
        Returns:
            Optional[Any]: Cached data or None if not available/expired
        """
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Get cache file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            cache_age_days = (datetime.now() - mod_time).total_seconds() / 86400
            
            # Check if cache is expired
            if cache_age_days > self.cache_expiry_days:
                logger.info(f"Cache expired ({cache_age_days:.1f} days old)")
                return None
            
            # Load cache
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Find our specific cache entry
            if cache_key in cache_data:
                entry = cache_data[cache_key]
                
                # Verify cache metadata matches query parameters
                metadata = entry.get('metadata', {})
                for key, value in kwargs.items():
                    if key not in metadata or metadata[key] != value:
                        logger.info(f"Cache metadata mismatch for {key}")
                        return None
                
                # Return cached data
                return entry.get('data')
            
            return None
            
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None
    
    def _save_to_cache(
        self,
        data: Any,
        cache_file: str,
        cache_key: str,
        **kwargs
    ) -> bool:
        """
        Save data to cache with metadata.
        
        Args:
            data (Any): Data to cache
            cache_file (str): Path to the cache file
            cache_key (str): Unique key for this cache entry
            **kwargs: Additional cache metadata to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            
            # Create cache entry
            cache_entry = {
                'data': data,
                'metadata': kwargs,
                'cached_at': datetime.now()
            }
            
            # Load existing cache if it exists
            cache_data = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                except:
                    # If cache is corrupt, start fresh
                    cache_data = {}
            
            # Update cache with new entry
            cache_data[cache_key] = cache_entry
            
            # Save to file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved data to cache: {cache_file} with key {cache_key}")
            return True
            
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
            return False
    
    def _generate_synthetic_price_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
        token_pair: str = "USD*/USDC"
    ) -> pd.DataFrame:
        """
        Generate synthetic price data for testing purposes.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            interval (str): Time interval (e.g., "1h", "1d")
            token_pair (str): Token pair name
            
        Returns:
            pd.DataFrame: Synthetic price data
        """
        logger.info(f"Generating synthetic price data for {token_pair}")
        
        # Convert interval to timedelta
        if interval == "1h":
            delta = timedelta(hours=1)
        elif interval == "1d":
            delta = timedelta(days=1)
        elif interval == "15m":
            delta = timedelta(minutes=15)
        else:
            delta = timedelta(hours=1)
        
        # Generate timestamps
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += delta
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price depending on the token pair
        if "USD*/USDC" in token_pair or "USD*/USDT" in token_pair:
            base_price = 1.0
            volatility = 0.0005  # Low volatility for stablecoin pairs
        elif "USD*/DAI" in token_pair:
            base_price = 1.0
            volatility = 0.001   # Slightly higher volatility
        else:
            base_price = 1.0
            volatility = 0.002   # Higher volatility for unknown pairs
        
        # Generate price series with some randomness and mean reversion
        n = len(timestamps)
        random_changes = np.random.normal(0, volatility, n)
        
        # Add some mean reversion
        price_series = [base_price]
        for i in range(1, n):
            # Add mean reversion - pull back towards the base price
            mean_reversion = 0.1 * (base_price - price_series[-1])
            new_price = price_series[-1] * (1 + random_changes[i] + mean_reversion)
            price_series.append(new_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': price_series
        })
        
        return df
    
    def _generate_synthetic_pool_data(
        self,
        start_date: datetime,
        end_date: datetime,
        pool_id: str = "pool-123456"
    ) -> Dict[str, Any]:
        """
        Generate synthetic pool data for testing purposes.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            pool_id (str): Pool ID
            
        Returns:
            Dict[str, Any]: Synthetic pool data
        """
        logger.info(f"Generating synthetic pool data for {pool_id}")
        
        # Generate timestamps (hourly data)
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(hours=1)
        
        # Set random seed for reproducibility
        np.random.seed(int(pool_id.split('-')[-1]) % 1000)
        
        # Generate synthetic volume data with daily and weekly patterns
        n = len(timestamps)
        
        # Base volume depending on pool ID (just for variety in test data)
        pool_idx = int(pool_id.split('-')[-1]) % 1000
        base_volume = 50000 + pool_idx * 1000
        
        # Create hourly, daily, and weekly patterns
        hour_pattern = np.array([0.8 + 0.4 * np.sin(h * np.pi / 12) for h in range(24)])
        day_pattern = np.array([0.9 + 0.2 * np.sin(d * np.pi / 3.5) for d in range(7)])
        
        # Generate volume with patterns and randomness
        volumes = []
        for i, ts in enumerate(timestamps):
            hour_factor = hour_pattern[ts.hour]
            day_factor = day_pattern[ts.weekday()]
            
            # Base volume with patterns
            volume = base_volume * hour_factor * day_factor
            
            # Add randomness (around 20%)
            volume *= (0.8 + 0.4 * np.random.random())
            
            volumes.append(volume)
        
        # Generate synthetic TVL data with slower changes
        base_tvl = 5000000 + pool_idx * 100000
        
        # Create a slow-moving random walk for TVL
        tvl_changes = np.random.normal(0, 0.005, n)
        tvl_walk = np.cumprod(1 + tvl_changes)
        tvls = base_tvl * tvl_walk
        
        # Generate fee rates (mostly constant with occasional changes)
        fee_rates = [0.003] * n  # Default fee rate of 0.3%
        
        # Occasionally change fee rates
        change_points = np.random.choice(n, 3, replace=False)
        for cp in change_points:
            new_fee = np.random.choice([0.001, 0.002, 0.003, 0.005])
            for i in range(cp, min(cp + 24 * 7, n)):  # Change for about a week
                fee_rates[i] = new_fee
        
        # Create DataFrames
        volume_df = pd.DataFrame({
            'timestamp': timestamps,
            'volume': volumes
        })
        
        fee_df = pd.DataFrame({
            'timestamp': timestamps,
            'fee_rate': fee_rates
        })
        
        tvl_df = pd.DataFrame({
            'timestamp': timestamps,
            'tvl': tvls
        })
        
        # Return as dictionary
        return {
            'pool_id': pool_id,
            'volume_history': volume_df,
            'fee_history': fee_df,
            'tvl_history': tvl_df
        }


def load_test_data(
    start_date: Union[str, datetime] = None,
    end_date: Union[str, datetime] = None,
    token_pair: str = "USD*/USDC",
    pool_id: str = None,
    use_coingecko: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to load test data for backtesting.
    
    Args:
        start_date (Union[str, datetime], optional): Start date
        end_date (Union[str, datetime], optional): End date
        token_pair (str, optional): Token pair (e.g., "USD*/USDC")
        pool_id (str, optional): Pool ID
        use_coingecko (bool, optional): Whether to use CoinGecko API
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Price data and pool data
    """
    loader = DataLoader(use_cache=True, use_coingecko=use_coingecko)
    
    # Load price data
    price_data = loader.load_price_data(
        start_date=start_date,
        end_date=end_date,
        token_pair=token_pair
    )
    
    # Load pool data (use matching pool ID if not specified)
    if pool_id is None:
        # Map token pair to pool ID
        if "USD*/USDC" in token_pair:
            pool_id = "pool-123456"
        elif "USD*/USDT" in token_pair:
            pool_id = "pool-234567"
        elif "USD*/DAI" in token_pair:
            pool_id = "pool-345678"
        else:
            pool_id = "pool-123456"
    
    pool_data = loader.load_pool_data(
        pool_id=pool_id,
        start_date=start_date,
        end_date=end_date
    )
    
    return price_data, pool_data


if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO)
    
    # Create a data loader with CoinGecko enabled
    loader = DataLoader(use_cache=True, use_coingecko=True)
    
    # Test loading price data
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    print("\nTesting price data loading with CoinGecko:")
    price_data = loader.load_price_data(
        start_date=start_date,
        end_date=end_date,
        token_pair="USD*/USDC",
        interval="1h"
    )
    
    print("Price Data:")
    print(f"Shape: {price_data.shape}")
    print(price_data.head())
    
    # Test loading pool data
    pool_data = loader.load_pool_data(
        pool_id="pool-123456",
        start_date=start_date,
        end_date=end_date
    )
    
    print("\nPool Data:")
    print("Volume History:")
    print(pool_data['volume_history'].head())
    
    print("\nFee History:")
    print(pool_data['fee_history'].head())
    
    print("\nTVL History:")
    print(pool_data['tvl_history'].head())
    
    # Test the convenience function
    print("\nTesting convenience function:")
    price_data, pool_data = load_test_data(
        start_date=start_date,
        end_date=end_date,
        use_coingecko=True
    )
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Pool ID: {pool_data.get('pool_id')}")
    
    # Try loading data for a different token pair
    print("\nTesting with a different token pair:")
    price_data = loader.load_price_data(
        start_date=start_date,
        end_date=end_date,
        token_pair="USD*/USDT",
        interval="1h"
    )
    
    print(f"USD*/USDT Price Data Shape: {price_data.shape}")
    print(price_data.head()) 