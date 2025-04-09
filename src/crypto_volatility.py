#!/usr/bin/env python3
"""
Crypto Volatility Analysis

This script collects price data for BTC, ETH, and S (Sonic), then calculates and visualizes
volatility indices for BTC and ETH.
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70))
    print("=" * 70 + "\n")


def run_command(command, description):
    """
    Run a Python script as a subprocess.
    
    Args:
        command: List with the command and arguments
        description: Description of the command
        
    Returns:
        True if successful, False otherwise
    """
    print_header(description)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False


def collect_price_data():
    """Collect price data for BTC, ETH, and S."""
    data_script = os.path.join(SCRIPT_DIR, "basic_volatility_tracker.py")
    return run_command(
        [sys.executable, data_script], 
        "COLLECTING PRICE DATA"
    )


def calculate_btc_volatility():
    """Calculate and visualize BTC volatility index."""
    btc_script = os.path.join(SCRIPT_DIR, "btc_volatility_index.py")
    return run_command(
        [sys.executable, btc_script],
        "CALCULATING BTC VOLATILITY INDEX"
    )


def calculate_eth_volatility():
    """Calculate and visualize ETH volatility index."""
    eth_script = os.path.join(SCRIPT_DIR, "eth_volatility_index.py")
    return run_command(
        [sys.executable, eth_script],
        "CALCULATING ETH VOLATILITY INDEX"
    )


def print_summary(data_success, btc_success, eth_success):
    """Print a summary of the operations."""
    print_header("SUMMARY")
    
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data collection: {'✅ Success' if data_success else '❌ Failed'}")
    print(f"BTC volatility index: {'✅ Success' if btc_success else '❌ Failed'}")
    print(f"ETH volatility index: {'✅ Success' if eth_success else '❌ Failed'}")
    
    # Print data locations
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data", "volatility")
    volatility_index_dir = os.path.join(root_dir, "data", "volatility_index")
    
    print("\nData Locations:")
    print(f"Raw CSV data: {data_dir}")
    print(f"Volatility indices: {volatility_index_dir}")
    
    print("\nKey Files:")
    if data_success:
        print("Raw Data Files:")
        for coin in ["BTC", "ETH", "S"]:
            daily_file = os.path.join(data_dir, f"{coin}_daily.csv")
            hourly_file = os.path.join(data_dir, f"{coin}_hourly.csv")
            
            if os.path.exists(daily_file):
                print(f"  - {coin} Daily: {daily_file}")
            if os.path.exists(hourly_file):
                print(f"  - {coin} Hourly: {hourly_file}")
    
    if btc_success or eth_success:
        print("\nVolatility Index Files:")
        if btc_success:
            btc_daily = os.path.join(volatility_index_dir, "BTC_daily_volatility.csv")
            btc_hourly = os.path.join(volatility_index_dir, "BTC_hourly_volatility.csv")
            btc_daily_plot = os.path.join(volatility_index_dir, "BTC_daily_volatility.png")
            btc_hourly_plot = os.path.join(volatility_index_dir, "BTC_hourly_volatility.png")
            
            print(f"  - BTC Daily Index: {btc_daily}")
            print(f"  - BTC Hourly Index: {btc_hourly}")
            print(f"  - BTC Daily Plot: {btc_daily_plot}")
            print(f"  - BTC Hourly Plot: {btc_hourly_plot}")
            
        if eth_success:
            eth_daily = os.path.join(volatility_index_dir, "ETH_daily_volatility.csv")
            eth_hourly = os.path.join(volatility_index_dir, "ETH_hourly_volatility.csv")
            eth_daily_plot = os.path.join(volatility_index_dir, "ETH_daily_volatility.png")
            eth_hourly_plot = os.path.join(volatility_index_dir, "ETH_hourly_volatility.png")
            
            print(f"  - ETH Daily Index: {eth_daily}")
            print(f"  - ETH Hourly Index: {eth_hourly}")
            print(f"  - ETH Daily Plot: {eth_daily_plot}")
            print(f"  - ETH Hourly Plot: {eth_hourly_plot}")


def main():
    """Main entry point for the script."""
    start_time = time.time()
    
    print_header("CRYPTO VOLATILITY ANALYSIS")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Collect price data
    data_success = collect_price_data()
    
    # Calculate volatility indices if data collection was successful
    btc_success = False
    eth_success = False
    
    if data_success:
        btc_success = calculate_btc_volatility()
        eth_success = calculate_eth_volatility()
    
    # Print summary
    print_summary(data_success, btc_success, eth_success)
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    
    # Return appropriate exit code
    if data_success and (btc_success or eth_success):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main()) 