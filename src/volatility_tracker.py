#!/usr/bin/env python3
"""
Volatility Tracker - Combined Script

This script fetches historical price data for Bitcoin (BTC) and Sonic (S) tokens,
generates CSV files, and updates a Google Sheet with the data.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Set the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70))
    print("=" * 70 + "\n")


def run_data_collection():
    """Run the data collection script."""
    print_header("STEP 1: COLLECTING PRICE DATA")
    
    data_script = os.path.join(SCRIPT_DIR, "basic_volatility_tracker.py")
    
    try:
        # Run the data collection script
        result = subprocess.run(
            [sys.executable, data_script],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n✅ Data collection completed successfully!")
            return True
        else:
            print(f"\n❌ Data collection failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running data collection script: {e}")
        return False


def run_sheets_update():
    """Run the Google Sheets update script."""
    print_header("STEP 2: UPDATING GOOGLE SHEETS")
    
    sheets_script = os.path.join(SCRIPT_DIR, "update_google_sheets.py")
    
    try:
        # Check if valid credentials exist
        creds_file = os.path.join(ROOT_DIR, "credentials.json")
        if not os.path.exists(creds_file):
            print(f"❌ Google credentials file not found: {creds_file}")
            print("Skipping Google Sheets update. Data is still available in CSV files.")
            return False
        
        # Check if credentials file contains placeholder values
        with open(creds_file, 'r') as f:
            contents = f.read()
            if "placeholder" in contents:
                print(f"❌ Google credentials file contains placeholder values: {creds_file}")
                print("Please replace the placeholder values with actual Google Service Account credentials.")
                print("See README-volatility-tracker.md for instructions on setting up Google Sheets API.")
                print("Skipping Google Sheets update. Data is still available in CSV files.")
                return False
        
        # Run the sheets update script
        result = subprocess.run(
            [sys.executable, sheets_script],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n✅ Google Sheets update completed successfully!")
            return True
        else:
            print(f"\n❌ Google Sheets update failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running Google Sheets update script: {e}")
        return False


def print_summary(data_success, sheets_success):
    """Print a summary of the operations."""
    print_header("SUMMARY")
    
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data collection: {'✅ Success' if data_success else '❌ Failed'}")
    print(f"Google Sheets update: {'✅ Success' if sheets_success else '❌ Failed'}")
    
    # Print data locations
    print("\nData Locations:")
    data_dir = os.path.join(ROOT_DIR, "data", "volatility")
    
    if data_success:
        print(f"CSV files: {data_dir}")
        for coin in ["BTC", "S", "wS", "USDC.e", "scUSD"]:
            daily_file = os.path.join(data_dir, f"{coin}_daily.csv")
            hourly_file = os.path.join(data_dir, f"{coin}_hourly.csv")
            
            if os.path.exists(daily_file):
                print(f"  - {coin} Daily: {daily_file}")
            if os.path.exists(hourly_file):
                print(f"  - {coin} Hourly: {hourly_file}")
    
    print("\nTo import into Google Sheets manually:")
    print("1. Open a new Google Sheet")
    print("2. Click on File > Import")
    print("3. Upload the CSV files")
    print("4. Choose 'Replace current sheet' or 'Insert new sheet(s)'")
    print("5. Share the sheet with your-email@example.com")
    
    print("\nTo run this script automatically:")
    print("Add to crontab to run daily:")
    print("0 1 * * * cd /path/to/usd-rewards && python src/volatility_tracker.py")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Track volatility for cryptocurrencies and update Google Sheets."
    )
    parser.add_argument(
        "--data-only", 
        action="store_true",
        help="Only collect data, don't update Google Sheets"
    )
    parser.add_argument(
        "--sheets-only", 
        action="store_true",
        help="Only update Google Sheets using existing data, don't fetch new data"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print_header("VOLATILITY TRACKER - SONIC PROTOCOL ASSETS")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_success = False
    sheets_success = False
    
    # Run data collection if needed
    if not args.sheets_only:
        data_success = run_data_collection()
    else:
        print("Skipping data collection (--sheets-only flag used)")
        data_success = True  # Assume data exists
    
    # Run sheets update if needed
    if not args.data_only and data_success:
        sheets_success = run_sheets_update()
    elif args.data_only:
        print("Skipping Google Sheets update (--data-only flag used)")
    elif not data_success:
        print("Skipping Google Sheets update (data collection failed)")
    
    # Print summary
    print_summary(data_success, sheets_success)
    
    # Return appropriate exit code
    if args.data_only and data_success:
        return 0
    elif args.sheets_only and sheets_success:
        return 0
    elif data_success and sheets_success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main()) 