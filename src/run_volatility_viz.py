#!/usr/bin/env python3
"""
Run Volatility Visualization

This script:
1. Collects price data for BTC, ETH, and S using basic_volatility_tracker.py
2. Calculates volatility indices for BTC and ETH
3. Generates visualization charts including a volatility pairing chart
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70))
    print("=" * 70 + "\n")


def run_script(script_name, description):
    """
    Run a Python script as a subprocess.
    
    Args:
        script_name: Name of the script to run
        description: Description of what the script does
        
    Returns:
        True if successful, False otherwise
    """
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    print_header(description)
    print(f"Running: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
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
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False


def main():
    """Main entry point for the script."""
    start_time = time.time()
    
    print_header("CRYPTO VOLATILITY VISUALIZATION")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Collect price data
    data_success = run_script(
        "basic_volatility_tracker.py",
        "STEP 1: COLLECTING PRICE DATA"
    )
    
    if not data_success:
        print("❌ Failed to collect price data. Exiting.")
        return 1
    
    # Step 2: Calculate BTC volatility
    btc_success = run_script(
        "btc_volatility_index.py",
        "STEP 2: CALCULATING BTC VOLATILITY INDEX"
    )
    
    # Step 3: Calculate ETH volatility
    eth_success = run_script(
        "eth_volatility_index.py",
        "STEP 3: CALCULATING ETH VOLATILITY INDEX"
    )
    
    if not btc_success or not eth_success:
        print("⚠️ Warning: Some volatility calculations failed, but continuing with available data.")
    
    # Step 4: Generate visualizations
    viz_success = run_script(
        "volatility_visualization.py",
        "STEP 4: GENERATING VOLATILITY VISUALIZATIONS"
    )
    
    # Print summary
    print_header("EXECUTION SUMMARY")
    
    print(f"▶ Data collection: {'✅ Success' if data_success else '❌ Failed'}")
    print(f"▶ BTC volatility index: {'✅ Success' if btc_success else '❌ Failed'}")
    print(f"▶ ETH volatility index: {'✅ Success' if eth_success else '❌ Failed'}")
    print(f"▶ Visualizations: {'✅ Success' if viz_success else '❌ Failed'}")
    
    # Display execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    
    # Output directories
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data", "volatility")
    volatility_index_dir = os.path.join(root_dir, "data", "volatility_index")
    viz_dir = os.path.join(root_dir, "data", "visualizations")
    
    print("\nOutput Directories:")
    print(f"▶ Raw price data: {data_dir}")
    print(f"▶ Volatility indices: {volatility_index_dir}")
    print(f"▶ Visualizations: {viz_dir}")
    
    # Return success or failure
    if data_success and btc_success and eth_success and viz_success:
        print("\n✅ All steps completed successfully!")
        return 0
    else:
        print("\n⚠️ Some steps failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 