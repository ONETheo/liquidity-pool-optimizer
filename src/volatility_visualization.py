#!/usr/bin/env python3
"""
Volatility Visualization

This script generates visualizations for BTC and ETH volatility data, including:
1. Individual volatility charts for BTC and ETH
2. A volatility pairing chart that shows the correlation between BTC and ETH volatility
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "volatility")
VOLATILITY_INDEX_DIR = os.path.join(ROOT_DIR, "data", "volatility_index")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "visualizations")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70))
    print("=" * 70 + "\n")


def load_volatility_data(symbol, timeframe="daily"):
    """
    Load volatility data for a specific symbol.
    
    Args:
        symbol: Token symbol (e.g., "BTC", "ETH")
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        DataFrame with volatility data
    """
    filename = f"{symbol}_{timeframe}_volatility.csv"
    file_path = os.path.join(VOLATILITY_INDEX_DIR, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Index is already timestamp
            df.index = pd.to_datetime(df.index)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def plot_volatility_comparison(btc_data, eth_data, timeframe="daily"):
    """
    Create a plot comparing BTC and ETH volatility.
    
    Args:
        btc_data: DataFrame with BTC volatility data
        eth_data: DataFrame with ETH volatility data
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved plot file
    """
    if btc_data is None or eth_data is None:
        logger.error(f"Missing data for volatility comparison")
        return None
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_style("darkgrid")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Price comparison
    ax1 = axes[0]
    ax1.set_title("BTC vs ETH Price", fontsize=14)
    
    # Normalize prices to 100 at the start for comparison
    btc_price = btc_data['price'] / btc_data['price'].iloc[0] * 100
    eth_price = eth_data['price'] / eth_data['price'].iloc[0] * 100
    
    btc_price.plot(ax=ax1, color='orange', label='BTC Price (normalized)')
    eth_price.plot(ax=ax1, color='blue', label='ETH Price (normalized)')
    
    ax1.set_ylabel("Normalized Price (100 = start)")
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot 2: Volatility index comparison
    ax2 = axes[1]
    ax2.set_title("BTC vs ETH Volatility Index", fontsize=14)
    
    if timeframe == "daily":
        btc_vol_col = 'btc_volatility_index'
        eth_vol_col = 'eth_volatility_index'
    else:
        btc_vol_col = 'btc_volatility_index_hourly'
        eth_vol_col = 'eth_volatility_index_hourly'
    
    btc_data[btc_vol_col].plot(ax=ax2, color='orange', label='BTC Volatility')
    eth_data[eth_vol_col].plot(ax=ax2, color='blue', label='ETH Volatility')
    
    ax2.set_ylabel("Volatility Index")
    ax2.set_xlabel("Date")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, f"BTC_ETH_{timeframe}_comparison.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved volatility comparison plot to {plot_file}")
    return plot_file


def create_volatility_pairing_chart(btc_data, eth_data, timeframe="daily"):
    """
    Create a scatterplot showing the relationship between BTC and ETH volatility.
    
    Args:
        btc_data: DataFrame with BTC volatility data
        eth_data: DataFrame with ETH volatility data
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved plot file
    """
    if btc_data is None or eth_data is None:
        logger.error(f"Missing data for volatility pairing chart")
        return None
    
    # Get the volatility columns based on timeframe
    if timeframe == "daily":
        btc_vol_col = 'btc_volatility_index'
        eth_vol_col = 'eth_volatility_index'
    else:
        btc_vol_col = 'btc_volatility_index_hourly'
        eth_vol_col = 'eth_volatility_index_hourly'
    
    # Join the dataframes on the index (date)
    if 'timestamp' in btc_data.columns:
        btc_data.set_index('timestamp', inplace=True)
    if 'timestamp' in eth_data.columns:
        eth_data.set_index('timestamp', inplace=True)
    
    merged_data = pd.DataFrame({
        'BTC_Volatility': btc_data[btc_vol_col],
        'ETH_Volatility': eth_data[eth_vol_col]
    })
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_style("darkgrid")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create scatterplot with regression line
    sns.regplot(
        x='BTC_Volatility', 
        y='ETH_Volatility', 
        data=merged_data,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'}
    )
    
    # Calculate correlation
    correlation = merged_data['BTC_Volatility'].corr(merged_data['ETH_Volatility'])
    
    plt.title(f"BTC vs ETH Volatility Correlation ({timeframe})\nCorrelation: {correlation:.3f}", fontsize=14)
    plt.xlabel("BTC Volatility Index")
    plt.ylabel("ETH Volatility Index")
    
    # Add annotation with correlation value
    plt.annotate(
        f"Correlation: {correlation:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Save plot
    plot_file = os.path.join(OUTPUT_DIR, f"BTC_ETH_{timeframe}_correlation.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved volatility pairing chart to {plot_file}")
    return plot_file


def create_volatility_heatmap(btc_data, eth_data, timeframe="daily"):
    """
    Create a heatmap showing the correlation between various volatility metrics.
    
    Args:
        btc_data: DataFrame with BTC volatility data
        eth_data: DataFrame with ETH volatility data
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved plot file
    """
    if btc_data is None or eth_data is None:
        logger.error(f"Missing data for volatility heatmap")
        return None
    
    # Set indices for merging
    if 'timestamp' in btc_data.columns:
        btc_data.set_index('timestamp', inplace=True)
    if 'timestamp' in eth_data.columns:
        eth_data.set_index('timestamp', inplace=True)
    
    # Select columns for correlation analysis based on timeframe
    if timeframe == "daily":
        btc_cols = ['price', 'volatility_7d', 'volatility_14d', 'volatility_30d', 'btc_volatility_index']
        eth_cols = ['price', 'volatility_7d', 'volatility_14d', 'volatility_30d', 'eth_volatility_index']
    else:
        btc_cols = ['price', 'volatility_12h', 'volatility_24h', 'volatility_48h', 'btc_volatility_index_hourly']
        eth_cols = ['price', 'volatility_12h', 'volatility_24h', 'volatility_48h', 'eth_volatility_index_hourly']
    
    # Create a merged dataframe with selected BTC and ETH metrics
    merged_data = pd.DataFrame()
    
    # Add BTC columns
    for col in btc_cols:
        if col in btc_data.columns:
            merged_data[f'BTC_{col}'] = btc_data[col]
    
    # Add ETH columns
    for col in eth_cols:
        if col in eth_data.columns:
            merged_data[f'ETH_{col}'] = eth_data[col]
    
    # Drop rows with missing values
    merged_data = merged_data.dropna()
    
    # Calculate correlation matrix
    corr_matrix = merged_data.corr()
    
    # Set plot style
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap,
        vmax=1.0, 
        vmin=-1.0,
        center=0,
        square=True, 
        linewidths=.5, 
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f"BTC & ETH Volatility Metrics Correlation ({timeframe})", fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(OUTPUT_DIR, f"BTC_ETH_{timeframe}_heatmap.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    logger.info(f"Saved volatility correlation heatmap to {plot_file}")
    return plot_file


def main():
    """Main entry point for the script."""
    print_header("VOLATILITY VISUALIZATION")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load daily volatility data
    print("Loading daily volatility data...")
    btc_daily = load_volatility_data("BTC", "daily")
    eth_daily = load_volatility_data("ETH", "daily")
    
    if btc_daily is not None and eth_daily is not None:
        # Create daily visualizations
        plot_volatility_comparison(btc_daily, eth_daily, "daily")
        create_volatility_pairing_chart(btc_daily, eth_daily, "daily")
        create_volatility_heatmap(btc_daily, eth_daily, "daily")
    else:
        logger.warning("Daily volatility data not available for both BTC and ETH")
    
    # Load hourly volatility data
    print("Loading hourly volatility data...")
    btc_hourly = load_volatility_data("BTC", "hourly")
    eth_hourly = load_volatility_data("ETH", "hourly")
    
    if btc_hourly is not None and eth_hourly is not None:
        # Create hourly visualizations
        plot_volatility_comparison(btc_hourly, eth_hourly, "hourly")
        create_volatility_pairing_chart(btc_hourly, eth_hourly, "hourly")
        create_volatility_heatmap(btc_hourly, eth_hourly, "hourly")
    else:
        logger.warning("Hourly volatility data not available for both BTC and ETH")
    
    # Print summary
    print_header("VISUALIZATION SUMMARY")
    print(f"Generated visualizations at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # List output files
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    
    if output_files:
        print("\nGenerated visualization files:")
        for file in output_files:
            print(f"  - {file}")
    else:
        print("\nNo visualization files were generated.")
    
    print("\nTo view these files, open them with any image viewer or browser.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 