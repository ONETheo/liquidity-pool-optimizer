#!/usr/bin/env python3
"""
Google Sheets Updater for Volatility Data

This script uploads volatility data from CSV files to a Google Sheet.
It requires a Google Service Account credentials file for authentication.
"""

import os
import sys
import csv
import time
import datetime
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import SpreadsheetNotFound

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(ROOT_DIR, "credentials.json")
DATA_DIR = os.path.join(ROOT_DIR, "data", "volatility")
SPREADSHEET_NAME = "Crypto Price Tracking"
TIMEOUT_SECONDS = 300  # 5 minutes

# Scopes required for Google Sheets API
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


def log_message(message):
    """Print a timestamped log message."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_credentials():
    """
    Get Google API credentials from the service account credentials file.
    
    Returns:
        Credentials object or None if the credentials file is missing
    """
    if not os.path.exists(CREDENTIALS_FILE):
        log_message(f"Error: Credentials file not found at {CREDENTIALS_FILE}")
        return None
    
    try:
        return Credentials.from_service_account_file(
            CREDENTIALS_FILE, 
            scopes=SCOPES
        )
    except Exception as e:
        log_message(f"Error loading credentials: {e}")
        return None


def create_spreadsheet(client):
    """
    Create a new spreadsheet for tracking crypto prices.
    
    Args:
        client: Authenticated gspread client
        
    Returns:
        Spreadsheet object or None if creation failed
    """
    try:
        # Check if the spreadsheet already exists
        try:
            spreadsheet = client.open(SPREADSHEET_NAME)
            log_message(f"Using existing spreadsheet: {SPREADSHEET_NAME}")
            return spreadsheet
        except SpreadsheetNotFound:
            # Create a new spreadsheet if it doesn't exist
            log_message(f"Creating new spreadsheet: {SPREADSHEET_NAME}")
            spreadsheet = client.create(SPREADSHEET_NAME)
            
            # Share with a specific user
            spreadsheet.share("your-email@example.com", perm_type="user", role="writer")
            
            # Initialize worksheets
            worksheet = spreadsheet.sheet1
            worksheet.update_title("Overview")
            
            # Create worksheets for each coin
            spreadsheet.add_worksheet(title="BTC", rows=100, cols=20)
            spreadsheet.add_worksheet(title="S", rows=100, cols=20)
            spreadsheet.add_worksheet(title="wS", rows=100, cols=20)
            spreadsheet.add_worksheet(title="USDC.e", rows=100, cols=20)
            spreadsheet.add_worksheet(title="scUSD", rows=100, cols=20)
            
            # Set up overview sheet
            overview = spreadsheet.worksheet("Overview")
            overview.update("A1:B3", [
                ["Crypto Price Tracking", ""],
                ["Last Updated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["", ""]
            ])
            overview.update("A5:C5", [["Coin", "Last Price", "24h Change %"]])
            overview.update("A6:A10", [["BTC"], ["S"], ["wS"], ["USDC.e"], ["scUSD"]])
            
            # Format header
            overview.format("A1:B1", {
                "textFormat": {"bold": True, "fontSize": 14}
            })
            overview.format("A5:C5", {
                "textFormat": {"bold": True}
            })
            
            return spreadsheet
    except Exception as e:
        log_message(f"Error creating spreadsheet: {e}")
        return None


def read_csv_data(file_path):
    """
    Read data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (headers, data_rows) or (None, None) if the file cannot be read
    """
    if not os.path.exists(file_path):
        log_message(f"Error: CSV file not found at {file_path}")
        return None, None
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Read the first row as headers
            data_rows = list(reader)
            return headers, data_rows
    except Exception as e:
        log_message(f"Error reading CSV file: {e}")
        return None, None


def update_worksheet_for_coin(sheet, symbol):
    """
    Update the worksheet for a specific coin with price data.
    
    Args:
        sheet: Spreadsheet object
        symbol: Coin symbol (e.g., "BTC", "S")
        
    Returns:
        True if update was successful, False otherwise
    """
    log_message(f"Updating worksheet for {symbol}...")
    
    try:
        worksheet = sheet.worksheet(symbol)
        overview = sheet.worksheet("Overview")
        
        # Clear existing data
        worksheet.clear()
        
        # Set up headers and sections
        worksheet.update("A1:B1", [[f"{symbol} Price Data", ""]])
        worksheet.format("A1:B1", {
            "textFormat": {"bold": True, "fontSize": 14}
        })
        
        worksheet.update("A3:B3", [["Daily Price Data (Last 30 Days)", ""]])
        worksheet.format("A3:B3", {
            "textFormat": {"bold": True}
        })
        
        # Add daily data
        daily_file = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
        daily_headers, daily_data = read_csv_data(daily_file)
        
        if daily_headers and daily_data:
            # Update headers at A4
            worksheet.update("A4:D4", [daily_headers])
            
            # Update data starting at A5
            if daily_data:
                worksheet.update(f"A5:D{4 + len(daily_data)}", daily_data)
                
                # Update overview with the most recent price and change %
                last_daily = daily_data[0]  # Assuming data is in reverse chronological order
                if len(last_daily) >= 2:
                    price = last_daily[1]
                    row_index = 6 if symbol == "BTC" else 7
                    overview.update(f"B{row_index}", [[price]])
                
                if len(last_daily) >= 3 and last_daily[2]:
                    change = last_daily[2]
                    row_index = 6 if symbol == "BTC" else 7
                    overview.update(f"C{row_index}", [[change]])
                    
                    # Format cell color based on change value
                    try:
                        change_value = float(change)
                        color = {"red": 0.8, "green": 0.3, "blue": 0.3} if change_value < 0 else {"red": 0.3, "green": 0.8, "blue": 0.3}
                        
                        overview.format(f"C{row_index}", {
                            "backgroundColor": color
                        })
                    except (ValueError, TypeError):
                        pass  # Skip formatting if change is not a valid number
        else:
            log_message(f"No daily data found for {symbol}")
            worksheet.update("A4", [["No daily data available"]])
        
        # Add separation between daily and hourly data
        worksheet.update("A30:B30", [["Hourly Price Data (Last 24 Hours)", ""]])
        worksheet.format("A30:B30", {
            "textFormat": {"bold": True}
        })
        
        # Add hourly data
        hourly_file = os.path.join(DATA_DIR, f"{symbol}_hourly.csv")
        hourly_headers, hourly_data = read_csv_data(hourly_file)
        
        if hourly_headers and hourly_data:
            # Update headers at A31
            worksheet.update("A31:D31", [hourly_headers])
            
            # Update data starting at A32
            if hourly_data:
                worksheet.update(f"A32:D{31 + len(hourly_data)}", hourly_data)
        else:
            log_message(f"No hourly data found for {symbol}")
            worksheet.update("A31", [["No hourly data available"]])
        
        # Update the last updated time in the overview sheet
        overview.update("B2", [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]])
        
        log_message(f"Successfully updated {symbol} worksheet")
        return True
        
    except Exception as e:
        log_message(f"Error updating {symbol} worksheet: {e}")
        return False


def main():
    """Main entry point for the script."""
    print("\n" + "=" * 60)
    print(" GOOGLE SHEETS UPDATER ")
    print("=" * 60 + "\n")
    
    # Check for credentials file
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: Google credentials file not found at {CREDENTIALS_FILE}")
        print("Please download a service account credentials file from the Google Cloud Console")
        print("and place it at the specified location.")
        return False
    
    # Check for CSV files
    coin_symbols = ["BTC", "S", "wS", "USDC.e", "scUSD"]
    all_files_exist = True
    
    for symbol in coin_symbols:
        daily_file = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
        hourly_file = os.path.join(DATA_DIR, f"{symbol}_hourly.csv")
        
        if not os.path.exists(daily_file):
            print(f"Error: {symbol}_daily.csv not found")
            all_files_exist = False
        
        if not os.path.exists(hourly_file):
            print(f"Error: {symbol}_hourly.csv not found")
            all_files_exist = False
    
    if not all_files_exist:
        print("\nPlease run the data collection script first to generate the CSV files.")
        return False
    
    # Initialize Google Sheets client
    log_message("Authenticating with Google Sheets API...")
    credentials = get_credentials()
    
    if not credentials:
        print("Failed to obtain Google credentials. Check if the credentials file is valid.")
        return False
    
    try:
        client = gspread.authorize(credentials)
        log_message("Successfully authenticated with Google Sheets API")
    except Exception as e:
        log_message(f"Error authenticating with Google Sheets: {e}")
        return False
    
    # Create or open spreadsheet
    spreadsheet = create_spreadsheet(client)
    
    if not spreadsheet:
        log_message("Failed to create or open spreadsheet")
        return False
    
    # Update each coin worksheet
    success = True
    for symbol in coin_symbols:
        if not update_worksheet_for_coin(spreadsheet, symbol):
            success = False
    
    # Print summary
    print("\n" + "=" * 60)
    print(" UPDATE SUMMARY ")
    print("=" * 60)
    
    if success:
        print(f"\nSuccessfully updated Google Sheet: {SPREADSHEET_NAME}")
        print(f"Spreadsheet URL: {spreadsheet.url}")
    else:
        print("\nCompleted with some errors. Check the log messages above.")
    
    return success


if __name__ == "__main__":
    start_time = time.time()
    
    # Run with timeout
    success = main()
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nExecution completed in {elapsed_time:.2f} seconds")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 
    main() 