#!/usr/bin/env python3
"""
Ethereum Volatility Index

This script calculates and visualizes an Ethereum volatility index based on collected price data.
It uses multiple methodologies to quantify market volatility and outputs both CSV files and charts.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "volatility")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "volatility_index")
ETH_DAILY_FILE = os.path.join(DATA_DIR, "ETH_daily.csv")
ETH_HOURLY_FILE = os.path.join(DATA_DIR, "ETH_hourly.csv")


class EthereumVolatilityIndex:
    """
    Calculate and track Ethereum volatility using multiple methodologies.
    """
    
    def __init__(self):
        """Initialize the volatility index calculator."""
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load data
        self.daily_data = self._load_data(ETH_DAILY_FILE)
        self.hourly_data = self._load_data(ETH_HOURLY_FILE)
        
        # Initialize results dataframes
        self.daily_volatility = None
        self.hourly_volatility = None
    
    def _load_data(self, file_path):
        """
        Load price data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with price data
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None
    
    def calculate_daily_volatility(self):
        """
        Calculate daily volatility metrics.
        
        Returns:
            DataFrame with daily volatility metrics
        """
        if self.daily_data is None:
            logger.error("Daily data not loaded")
            return None
        
        # Create a copy of the data
        df = self.daily_data.copy()
        
        # Calculate log returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Calculate metrics
        # 1. Rolling standard deviation of returns (7, 14, 30 days)
        df['volatility_7d'] = df['log_return'].rolling(window=7).std() * 100
        df['volatility_14d'] = df['log_return'].rolling(window=14).std() * 100
        df['volatility_30d'] = df['log_return'].rolling(window=30).std() * 100
        
        # 2. Average absolute percent change (rolling windows)
        df['avg_abs_change_7d'] = df['volatility'].rolling(window=7).mean()
        df['avg_abs_change_14d'] = df['volatility'].rolling(window=14).mean()
        df['avg_abs_change_30d'] = df['volatility'].rolling(window=30).mean()
        
        # 3. Parkinson's volatility estimator (uses high-low range)
        # For this we'll approximate using the daily volatility values
        df['parkinson_7d'] = df['volatility'].rolling(window=7).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * np.log(2) * len(x)))
        )
        
        # 4. Create a composite volatility index
        # Equal weight to std deviation and average absolute change
        df['eth_volatility_index'] = (
            (df['volatility_14d'] / 2) + df['avg_abs_change_14d']
        )
        
        # Store result
        self.daily_volatility = df
        
        return df
    
    def calculate_hourly_volatility(self):
        """
        Calculate hourly volatility metrics.
        
        Returns:
            DataFrame with hourly volatility metrics
        """
        if self.hourly_data is None:
            logger.error("Hourly data not loaded")
            return None
        
        # Create a copy of the data
        df = self.hourly_data.copy()
        
        # Calculate log returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Calculate metrics
        # 1. Rolling standard deviation of returns (12, 24, 48 hours)
        df['volatility_12h'] = df['log_return'].rolling(window=12).std() * 100
        df['volatility_24h'] = df['log_return'].rolling(window=24).std() * 100
        df['volatility_48h'] = df['log_return'].rolling(window=48).std() * 100
        
        # 2. Average absolute percent change (rolling windows)
        df['avg_abs_change_12h'] = df['volatility'].rolling(window=12).mean()
        df['avg_abs_change_24h'] = df['volatility'].rolling(window=24).mean()
        
        # 3. Create a composite volatility index
        # Equal weight to std deviation and average absolute change
        df['eth_volatility_index_hourly'] = (
            (df['volatility_24h'] / 2) + df['avg_abs_change_24h']
        )
        
        # Store result
        self.hourly_volatility = df
        
        return df
    
    def save_volatility_data(self):
        """
        Save volatility data to CSV files.
        
        Returns:
            Dictionary with paths to output files
        """
        output_files = {}
        
        # Save daily volatility data
        if self.daily_volatility is not None:
            daily_file = os.path.join(OUTPUT_DIR, "ETH_daily_volatility.csv")
            self.daily_volatility.to_csv(daily_file)
            logger.info(f"Saved daily volatility data to {daily_file}")
            output_files['daily'] = daily_file
        
        # Save hourly volatility data
        if self.hourly_volatility is not None:
            hourly_file = os.path.join(OUTPUT_DIR, "ETH_hourly_volatility.csv")
            self.hourly_volatility.to_csv(hourly_file)
            logger.info(f"Saved hourly volatility data to {hourly_file}")
            output_files['hourly'] = hourly_file
        
        return output_files
    
    def plot_daily_volatility(self):
        """
        Create plots for daily volatility metrics.
        
        Returns:
            Path to saved plot file
        """
        if self.daily_volatility is None:
            logger.error("Daily volatility not calculated")
            return None
        
        # Set plot style
        plt.style.use('ggplot')
        sns.set_style("darkgrid")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot 1: Price and composite volatility index
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Plot price
        self.daily_volatility['price'].plot(ax=ax1, color='blue', label='ETH Price')
        
        # Plot volatility index
        self.daily_volatility['eth_volatility_index'].plot(
            ax=ax1_twin, color='red', label='ETH Volatility Index', linewidth=2
        )
        
        ax1.set_title('Ethereum Price and Volatility Index (Daily)', fontsize=14)
        ax1.set_ylabel('Price (USD)', color='blue')
        ax1_twin.set_ylabel('Volatility Index', color='red')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Different volatility metrics
        ax2 = axes[1]
        
        self.daily_volatility['volatility_7d'].plot(
            ax=ax2, label='7-Day StdDev', linewidth=1.5
        )
        self.daily_volatility['volatility_14d'].plot(
            ax=ax2, label='14-Day StdDev', linewidth=2
        )
        self.daily_volatility['volatility_30d'].plot(
            ax=ax2, label='30-Day StdDev', linewidth=2.5
        )
        
        ax2.set_title('Ethereum Return Volatility (Different Time Windows)', fontsize=14)
        ax2.set_ylabel('Volatility (%)')
        ax2.legend()
        
        # Plot 3: Average absolute change
        ax3 = axes[2]
        
        self.daily_volatility['avg_abs_change_7d'].plot(
            ax=ax3, label='7-Day Avg. Change', linewidth=1.5
        )
        self.daily_volatility['avg_abs_change_14d'].plot(
            ax=ax3, label='14-Day Avg. Change', linewidth=2
        )
        self.daily_volatility['avg_abs_change_30d'].plot(
            ax=ax3, label='30-Day Avg. Change', linewidth=2.5
        )
        
        ax3.set_title('Ethereum Average Daily Price Change (%)', fontsize=14)
        ax3.set_ylabel('Avg. Price Change (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        
        # Set tight layout
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, "ETH_daily_volatility.png")
        plt.savefig(plot_file, dpi=300)
        logger.info(f"Saved daily volatility plot to {plot_file}")
        
        return plot_file
    
    def plot_hourly_volatility(self):
        """
        Create plots for hourly volatility metrics.
        
        Returns:
            Path to saved plot file
        """
        if self.hourly_volatility is None:
            logger.error("Hourly volatility not calculated")
            return None
        
        # Set plot style
        plt.style.use('ggplot')
        sns.set_style("darkgrid")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Price and composite volatility index
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Plot price
        self.hourly_volatility['price'].plot(ax=ax1, color='blue', label='ETH Price')
        
        # Plot volatility index
        self.hourly_volatility['eth_volatility_index_hourly'].plot(
            ax=ax1_twin, color='red', label='ETH Hourly Volatility Index', linewidth=2
        )
        
        ax1.set_title('Ethereum Price and Volatility Index (Hourly)', fontsize=14)
        ax1.set_ylabel('Price (USD)', color='blue')
        ax1_twin.set_ylabel('Volatility Index', color='red')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Different volatility metrics
        ax2 = axes[1]
        
        self.hourly_volatility['volatility_12h'].plot(
            ax=ax2, label='12-Hour StdDev', linewidth=1.5
        )
        self.hourly_volatility['volatility_24h'].plot(
            ax=ax2, label='24-Hour StdDev', linewidth=2
        )
        self.hourly_volatility['volatility_48h'].plot(
            ax=ax2, label='48-Hour StdDev', linewidth=2.5, 
            alpha=0.7
        )
        
        ax2.set_title('Ethereum Return Volatility (Different Time Windows)', fontsize=14)
        ax2.set_ylabel('Volatility (%)')
        ax2.set_xlabel('Date/Time')
        ax2.legend()
        
        # Set tight layout
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, "ETH_hourly_volatility.png")
        plt.savefig(plot_file, dpi=300)
        logger.info(f"Saved hourly volatility plot to {plot_file}")
        
        return plot_file


def main():
    """Main entry point for the script."""
    print("\n" + "=" * 60)
    print(" ETHEREUM VOLATILITY INDEX ")
    print("=" * 60 + "\n")
    
    # Initialize the volatility index calculator
    eth_volatility = EthereumVolatilityIndex()
    
    # Calculate daily volatility
    print("Calculating daily volatility metrics...")
    eth_volatility.calculate_daily_volatility()
    
    # Calculate hourly volatility
    print("Calculating hourly volatility metrics...")
    eth_volatility.calculate_hourly_volatility()
    
    # Save data to CSV files
    print("Saving volatility data to CSV files...")
    output_files = eth_volatility.save_volatility_data()
    
    # Create plots
    print("Creating volatility plots...")
    daily_plot = eth_volatility.plot_daily_volatility()
    hourly_plot = eth_volatility.plot_hourly_volatility()
    
    # Print summary
    print("\n" + "=" * 60)
    print(" VOLATILITY INDEX SUMMARY ")
    print("=" * 60)
    
    print("\nVolatility data files:")
    for key, file_path in output_files.items():
        print(f"  - {key.capitalize()}: {file_path}")
    
    print("\nVolatility plots:")
    if daily_plot:
        print(f"  - Daily: {daily_plot}")
    if hourly_plot:
        print(f"  - Hourly: {hourly_plot}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main() 