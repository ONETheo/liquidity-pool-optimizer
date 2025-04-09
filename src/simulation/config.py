"""
Configuration Module

This module provides configuration settings for the USD* Rewards Simulator
and Parameter Optimizer.
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # API configuration
    "api_config": {
        "coingecko": {
            "base_url": "https://api.coingecko.com/api/v3",
            "token_ids": {
                "USD*": "usd-star",
                "USDC": "usd-coin",
                "USDT": "tether",
                "DAI": "dai",
                "S": "sonic-3",
                "wS": "wrapped-sonic",
                "USDC.e": "sonic-bridged-usdc-e-sonic",
                "scUSD": "rings-scusd"
            },
            "rate_limit": {
                "calls_per_minute": 10,
                "retry_after": 60
            },
            "api_key": os.getenv("COINGECKO_API_KEY", "")
        },
        "sonic": {
            "base_url": "https://sonic-api.example.com/v1",
            "api_key_env_var": "SONIC_API_KEY"
        }
    },
    
    # Default date ranges
    "dates": {
        "default_start_date": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        "default_end_date": datetime.now().strftime("%Y-%m-%d")
    },
    
    # Cache settings
    "cache": {
        "enabled": True,
        "expiry_days": 1,
        "price_data_cache": "usd-rewards/data/cache/price_data.pkl",
        "pool_data_cache": "usd-rewards/data/cache/pool_data.pkl"
    },
    
    # Gas and transaction cost settings
    "gas_settings": {
        "avg_gas_price": 15.0,  # in Gwei
        "rebalance_gas_cost": 150000,  # gas units for rebalance transaction
        "gas_price_multiplier": 1.1,  # safety buffer for gas price
        "icp_usd_price": 7.5  # USD price of ICP for gas cost calculation
    },
    
    # Default pool IDs
    "default_pool_ids": [
        "pool-123456",  # USD*/USDC
        "pool-234567",  # USD*/USDT
        "pool-345678"   # USD*/DAI
    ],
    
    # Simulation settings
    "simulation": {
        "default_width": 0.5,
        "default_buffer_lower": 0.1,
        "default_buffer_upper": 0.1,
        "default_cutoff_lower": 0.95,
        "default_cutoff_upper": 1.05,
        "default_investment": 10000.0,
        "include_transaction_costs": True,
        "max_rebalances_per_day": 48,  # Limit to prevent extreme rebalancing
        "min_time_between_rebalances": 30  # Minimum minutes between rebalances
    },
    
    # Optimizer settings
    "optimizer": {
        "default_strategy": "latin",  # grid, random, latin
        "default_sample_size": 100,
        "max_combinations": 1000,
        "parallel_jobs": 4,
        "optimization_metrics": [
            "net_apr",
            "time_in_position_pct",
            "rebalance_efficiency",
            "score"
        ]
    },
    
    # Results settings
    "results_dir": "usd-rewards/results",
    
    # Logging settings
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file": "usd-rewards/logs/simulation.log"
    }
}


def get_config() -> Dict[str, Any]:
    """
    Get the configuration settings, merging default with custom settings if available.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Check for custom config file
    config_file = "usd-rewards/config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
            
            # Merge custom config with defaults
            _merge_configs(config, custom_config)
            
            logger.info("Loaded custom configuration from config.json")
        except Exception as e:
            logger.warning(f"Error loading custom config: {e}")
    
    # Ensure directories exist
    _ensure_directories(config)
    
    return config


def _merge_configs(base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> None:
    """
    Recursively merge custom configuration into base configuration.
    
    Args:
        base_config (Dict[str, Any]): Base configuration to update
        custom_config (Dict[str, Any]): Custom configuration to merge
    """
    for key, value in custom_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value


def _ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure that required directories exist.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    directories = [
        os.path.dirname(config["cache"]["price_data_cache"]),
        os.path.dirname(config["cache"]["pool_data_cache"]),
        config["results_dir"]
    ]
    
    if "logging" in config and "log_file" in config["logging"]:
        directories.append(os.path.dirname(config["logging"]["log_file"]))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_config(config: Dict[str, Any], filepath: str = "usd-rewards/config.json") -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration to save
        filepath (str, optional): Path to save the configuration file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = get_config()
    
    # Print current configuration
    print("Current Configuration:")
    for section, settings in config.items():
        print(f"\n[{section}]")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {settings}")
    
    # Create example custom config file if it doesn't exist
    if not os.path.exists("usd-rewards/config.json"):
        example_config = {
            "api_config": {
                "coingecko": {
                    "api_key": "YOUR_API_KEY_HERE"  # Optional: for higher rate limits
                }
            },
            "simulation": {
                "default_investment": 50000.0,  # Higher investment amount
                "include_transaction_costs": True
            },
            "optimizer": {
                "parallel_jobs": 8  # Use more CPU cores
            }
        }
        
        save_config(example_config)
        print("\nCreated example custom configuration at usd-rewards/config.json")
        print("You can edit this file to customize your simulation settings.") 