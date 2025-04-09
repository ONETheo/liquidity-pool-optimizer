#!/usr/bin/env python3
"""
Volatility Tracking Script

This script fetches historical price data for Bitcoin (BTC) and Sonic (S) from CoinGecko,
calculates daily and hourly price information, and exports the data to a Google Sheet.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# Google Sheets imports
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Import local modules
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.coingecko import CoinGeckoAPI
from simulation.config import get_config

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CREDENTIALS_FILE = "usd-rewards/credentials.json"
SPREADSHEET_NAME = "Crypto Price Tracking"
RECIPIENT_EMAIL = "your-email@example.com"
COINS = [
    {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    {"id": "sonic-3", "symbol": "S", "name": "Sonic"}  # Updated to use latest CoinGecko ID
]


class PriceDataTracker:
    """
    Tracks price data for cryptocurrencies and exports to Google Sheets.
    """
    
    def __init__(self):
        """Initialize the tracker with API connections."""
        # Get configuration
        self.config = get_config()
        
        # Initialize CoinGecko API
        self.cg_api = CoinGeckoAPI(
            api_config=self.config["api_config"]["coingecko"]
        )
        
        # Initialize Google Sheets API
        try:
            self.gs_client = self._init_google_sheets()
            logger.info("Successfully connected to Google Sheets API")
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets API: {e}")
            self.gs_client = None
    
    def _init_google_sheets(self) -> gspread.Client:
        """
        Initialize Google Sheets API client.
        
        Returns:
            gspread.Client: Google Sheets client
        """
        # Define the scope
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Authenticate with Google Sheets API
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            CREDENTIALS_FILE, scope
        )
        
        return gspread.authorize(creds)
    
    def fetch_historical_data(
        self, 
        coin_id: str, 
        days: int = 30, 
        interval: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko ID of the cryptocurrency
            days: Number of days of historical data
            interval: Data interval ("daily" or "hourly")
            
        Returns:
            DataFrame with timestamp and price columns
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        df = self.cg_api.get_token_price_history(
            coin_id,
            vs_currency="usd",
            from_date=start_date,
            to_date=end_date,
            interval=interval
        )
        
        return df
    
    def create_spreadsheet(self) -> Optional[gspread.Spreadsheet]:
        """
        Create a new Google Sheet for price tracking.
        
        Returns:
            Google Sheets spreadsheet object or None if creation failed
        """
        if not self.gs_client:
            logger.error("Google Sheets client not initialized")
            return None
        
        try:
            # Create a new spreadsheet
            sheet = self.gs_client.create(SPREADSHEET_NAME)
            
            # Share with recipient
            sheet.share(RECIPIENT_EMAIL, perm_type="user", role="writer")
            
            # Create worksheets for each coin
            for coin in COINS:
                sheet.add_worksheet(title=coin["symbol"], rows=100, cols=20)
            
            # Remove the default Sheet1
            sheet.del_worksheet(sheet.get_worksheet(0))
            
            logger.info(f"Created spreadsheet: {sheet.url}")
            return sheet
            
        except Exception as e:
            logger.error(f"Failed to create spreadsheet: {e}")
            return None
    
    def prepare_daily_data(self, coin_id: str) -> pd.DataFrame:
        """
        Prepare daily price data for the last 30 days.
        
        Args:
            coin_id: CoinGecko ID of the cryptocurrency
            
        Returns:
            DataFrame with daily price data
        """
        # Fetch data for the last 30 days
        df = self.fetch_historical_data(coin_id, days=30, interval="daily")
        
        # Format data
        if not df.empty:
            df = df.sort_values(by="timestamp", ascending=False)
            df = df.head(30)  # Ensure we only have 30 days
            df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
            df = df[["date", "price"]]
            df.columns = ["Date", "Price (USD)"]
        
        return df
    
    def prepare_hourly_data(self, coin_id: str) -> pd.DataFrame:
        """
        Prepare hourly price data for the last 30 hours.
        
        Args:
            coin_id: CoinGecko ID of the cryptocurrency
            
        Returns:
            DataFrame with hourly price data
        """
        # For hourly data, we'll fetch data for the last 2 days to ensure we have 30 hours
        df = self.fetch_historical_data(coin_id, days=2, interval="hourly")
        
        # Format data
        if not df.empty:
            df = df.sort_values(by="timestamp", ascending=False)
            df = df.head(30)  # Ensure we only have 30 hours
            df["datetime"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:00")
            df = df[["datetime", "price"]]
            df.columns = ["DateTime", "Price (USD)"]
        
        return df
    
    def update_sheet_for_coin(
        self, 
        sheet: gspread.Spreadsheet, 
        coin: Dict[str, str]
    ) -> None:
        """
        Update a sheet with price data for a specific coin.
        
        Args:
            sheet: Google Sheets spreadsheet
            coin: Coin information (id, symbol, name)
        """
        try:
            # Get the worksheet for this coin
            worksheet = sheet.worksheet(coin["symbol"])
            
            # Clear the worksheet
            worksheet.clear()
            
            # Add header with coin information
            header = [
                [f"{coin['name']} ({coin['symbol']}) Price Data"],
                [],
                ["Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
            worksheet.update("A1", header)
            
            # Format header
            worksheet.format("A1", {
                "textFormat": {"bold": True, "fontSize": 14}
            })
            
            # Prepare data
            daily_data = self.prepare_daily_data(coin["id"])
            hourly_data = self.prepare_hourly_data(coin["id"])
            
            # Add daily data
            if not daily_data.empty:
                worksheet.update("A5", [["Daily Price Data (Last 30 Days)"]])
                worksheet.format("A5", {"textFormat": {"bold": True}})
                
                # Add column headers and data
                worksheet.update("A6", [daily_data.columns.tolist()])
                worksheet.update("A7", daily_data.values.tolist())
            
            # Add hourly data
            if not hourly_data.empty:
                # Determine row for hourly data (below daily data)
                start_row = 7 + len(daily_data) + 2
                
                worksheet.update(f"A{start_row}", [["Hourly Price Data (Last 30 Hours)"]])
                worksheet.format(f"A{start_row}", {"textFormat": {"bold": True}})
                
                # Add column headers and data
                worksheet.update(f"A{start_row + 1}", [hourly_data.columns.tolist()])
                worksheet.update(f"A{start_row + 2}", hourly_data.values.tolist())
            
            logger.info(f"Updated worksheet for {coin['symbol']}")
            
        except Exception as e:
            logger.error(f"Failed to update sheet for {coin['symbol']}: {e}")
    
    def run(self) -> str:
        """
        Run the price data tracking process.
        
        Returns:
            URL of the created spreadsheet or error message
        """
        try:
            # Create spreadsheet
            sheet = self.create_spreadsheet()
            if not sheet:
                return "Failed to create spreadsheet"
            
            # Update each coin's worksheet
            for coin in COINS:
                logger.info(f"Processing {coin['name']} ({coin['symbol']})")
                self.update_sheet_for_coin(sheet, coin)
            
            return f"Successfully created and updated spreadsheet: {sheet.url}"
            
        except Exception as e:
            logger.error(f"Error during tracking process: {e}")
            return f"Error: {str(e)}"


def main():
    """Main entry point for the script."""
    logger.info("Starting price data tracking")
    
    tracker = PriceDataTracker()
    result = tracker.run()
    
    logger.info(result)
    print(result)


if __name__ == "__main__":
    main() 