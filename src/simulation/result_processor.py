"""
Result Processor Module

This module provides functions to analyze, compare, and visualize
the results from multiple backtest runs with different parameters.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

# Set up logger
logger = logging.getLogger(__name__)

class ResultProcessor:
    """
    Process and visualize results from multiple backtests
    to compare performance across different parameter sets.
    """
    
    def __init__(self, results_dir: str = None):
        """
        Initialize the result processor.
        
        Args:
            results_dir (str, optional): Directory to save result visualizations
        """
        self.results_dir = results_dir
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Style configuration for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = sns.color_palette("viridis", 10)
    
    def process_backtest_results(
        self, 
        backtest_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Process multiple backtest results into a DataFrame for analysis.
        
        Args:
            backtest_results (List[Dict[str, Any]]): List of backtest result dictionaries
            
        Returns:
            pd.DataFrame: DataFrame with processed results
        """
        # Extract key metrics from each result
        processed_results = []
        
        for i, result in enumerate(backtest_results):
            # Get parameters
            parameters = result.get('parameters', {})
            metrics = result.get('metrics', {})
            
            # Create a result summary
            result_summary = {
                'result_id': i,
                'width': parameters.get('width', 0),
                'buffer_lower': parameters.get('buffer_lower', 0),
                'buffer_upper': parameters.get('buffer_upper', 0),
                'cutoff_lower': parameters.get('cutoff_lower', 0),
                'cutoff_upper': parameters.get('cutoff_upper', 0),
                'initial_investment': parameters.get('initial_investment', 0),
                'duration_days': parameters.get('duration_days', 0),
                'apr': metrics.get('apr', 0),
                'fees_apr': metrics.get('fees_apr', 0),
                'rebalance_count': metrics.get('rebalance_count', 0),
                'rebalance_frequency': metrics.get('rebalance_frequency', 0),
                'time_in_position_pct': metrics.get('time_in_position_pct', 0),
                'total_fees_earned': metrics.get('total_fees_earned', 0),
                'total_costs': metrics.get('total_costs', 0),
                'net_return': metrics.get('absolute_return', 0),
                'volatility': metrics.get('volatility', 0)
            }
            
            # Calculate risk-adjusted returns
            if metrics.get('volatility', 0) > 0:
                result_summary['sharpe_ratio'] = metrics.get('apr', 0) / metrics.get('volatility', 1)
            else:
                result_summary['sharpe_ratio'] = 0
            
            # Calculate net APR (after costs)
            result_summary['net_apr'] = metrics.get('apr', 0) - \
                ((metrics.get('total_costs', 0) / parameters.get('initial_investment', 1)) * \
                 (365 / parameters.get('duration_days', 1)) * 100)
            
            processed_results.append(result_summary)
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_results)
        
        # Add a combined buffer column
        if 'buffer_lower' in df.columns and 'buffer_upper' in df.columns:
            df['buffer_avg'] = (df['buffer_lower'] + df['buffer_upper']) / 2
        
        # Add a combined cutoff range column
        if 'cutoff_lower' in df.columns and 'cutoff_upper' in df.columns:
            df['cutoff_range'] = df['cutoff_upper'] - df['cutoff_lower']
        
        logger.info(f"Processed {len(df)} backtest results")
        return df
    
    def rank_results(
        self, 
        results_df: pd.DataFrame,
        metrics: List[str] = None,
        weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Rank backtest results based on multiple metrics.
        
        Args:
            results_df (pd.DataFrame): DataFrame with processed results
            metrics (List[str], optional): List of metrics to use for ranking
            weights (Dict[str, float], optional): Weights for each metric
            
        Returns:
            pd.DataFrame: DataFrame with ranked results
        """
        if metrics is None:
            metrics = ['net_apr', 'fees_apr', 'time_in_position_pct', 'sharpe_ratio']
        
        if weights is None:
            weights = {
                'net_apr': 0.4,
                'fees_apr': 0.2,
                'time_in_position_pct': 0.2,
                'sharpe_ratio': 0.2
            }
        
        # Check that all metrics are in the DataFrame
        for metric in metrics:
            if metric not in results_df.columns:
                logger.warning(f"Metric '{metric}' not found in results. Skipping.")
                metrics.remove(metric)
        
        if not metrics:
            logger.error("No valid metrics for ranking")
            return results_df
        
        # Normalize each metric to 0-1 scale
        normalized_df = results_df.copy()
        
        for metric in metrics:
            min_val = results_df[metric].min()
            max_val = results_df[metric].max()
            
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = (results_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric}_norm'] = 0
        
        # Calculate weighted sum
        normalized_df['score'] = 0
        
        for metric in metrics:
            weight = weights.get(metric, 1/len(metrics))
            normalized_df['score'] += normalized_df[f'{metric}_norm'] * weight
        
        # Rank results
        ranked_df = normalized_df.sort_values('score', ascending=False).reset_index(drop=True)
        ranked_df['rank'] = ranked_df.index + 1
        
        logger.info(f"Ranked results using metrics: {metrics}")
        return ranked_df
    
    def plot_parameter_comparison(
        self, 
        results_df: pd.DataFrame,
        parameter: str,
        metric: str,
        output_file: str = None
    ) -> None:
        """
        Create a plot comparing a parameter's effect on a performance metric.
        
        Args:
            results_df (pd.DataFrame): DataFrame with processed results
            parameter (str): Parameter to use as x-axis
            metric (str): Metric to use as y-axis
            output_file (str, optional): Path to save the plot
        """
        if parameter not in results_df.columns:
            logger.error(f"Parameter '{parameter}' not found in results")
            return
        
        if metric not in results_df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return
        
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(
            data=results_df,
            x=parameter,
            y=metric,
            hue='buffer_avg' if 'buffer_avg' in results_df.columns else None,
            palette='viridis',
            s=100,
            alpha=0.7
        )
        
        plt.title(f'Effect of {parameter} on {metric}')
        plt.xlabel(parameter)
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        if output_file and self.results_dir:
            output_path = os.path.join(self.results_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved parameter comparison plot to {output_path}")
        
        plt.close()
    
    def plot_heatmap(
        self,
        results_df: pd.DataFrame,
        x_param: str,
        y_param: str,
        metric: str,
        output_file: str = None
    ) -> None:
        """
        Create a heatmap showing the relationship between two parameters
        and their effect on a performance metric.
        
        Args:
            results_df (pd.DataFrame): DataFrame with processed results
            x_param (str): Parameter for x-axis
            y_param (str): Parameter for y-axis
            metric (str): Metric to use for color intensity
            output_file (str, optional): Path to save the plot
        """
        for param in [x_param, y_param]:
            if param not in results_df.columns:
                logger.error(f"Parameter '{param}' not found in results")
                return
        
        if metric not in results_df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return
        
        # Create pivot table
        try:
            pivot_df = results_df.pivot_table(
                index=y_param, 
                columns=x_param, 
                values=metric,
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                linewidths=0.5,
                cbar_kws={'label': metric}
            )
            
            plt.title(f'Heatmap of {metric} vs {x_param} and {y_param}')
            plt.xlabel(x_param)
            plt.ylabel(y_param)
            
            if output_file and self.results_dir:
                output_path = os.path.join(self.results_dir, output_file)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved heatmap plot to {output_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
    
    def plot_time_series(
        self,
        backtest_results: List[Dict[str, Any]],
        result_ids: List[int],
        metric: str = 'price',
        output_file: str = None
    ) -> None:
        """
        Create a time series plot comparing a specific metric across multiple backtest results.
        
        Args:
            backtest_results (List[Dict[str, Any]]): List of backtest result dictionaries
            result_ids (List[int]): IDs of the results to include in the plot
            metric (str): Metric to plot from price_history
            output_file (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for i, result_id in enumerate(result_ids):
            if result_id >= len(backtest_results):
                logger.warning(f"Result ID {result_id} is out of range")
                continue
                
            result = backtest_results[result_id]
            
            # Extract parameters for the legend
            params = result.get('parameters', {})
            width = params.get('width', 0)
            buffer_lower = params.get('buffer_lower', 0)
            buffer_upper = params.get('buffer_upper', 0)
            
            # Extract time series data
            price_history = result.get('price_history', [])
            if not price_history:
                logger.warning(f"No price history found for result {result_id}")
                continue
                
            # Convert to DataFrame
            price_df = pd.DataFrame(price_history)
            
            if 'timestamp' not in price_df.columns or metric not in price_df.columns:
                logger.warning(f"Required columns not found in price history for result {result_id}")
                continue
                
            # Plot the time series
            plt.plot(
                price_df['timestamp'],
                price_df[metric],
                label=f"Width={width}, Buffer={buffer_lower}/{buffer_upper}",
                color=self.colors[i % len(self.colors)],
                alpha=0.8
            )
                
        plt.title(f'Time Series Comparison of {metric}')
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_file and self.results_dir:
            output_path = os.path.join(self.results_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved time series plot to {output_path}")
        
        plt.close()
    
    def plot_rebalance_events(
        self,
        backtest_results: List[Dict[str, Any]],
        result_id: int,
        output_file: str = None
    ) -> None:
        """
        Create a plot showing when rebalance events occurred along with the price.
        
        Args:
            backtest_results (List[Dict[str, Any]]): List of backtest result dictionaries
            result_id (int): ID of the result to plot
            output_file (str, optional): Path to save the plot
        """
        if result_id >= len(backtest_results):
            logger.error(f"Result ID {result_id} is out of range")
            return
            
        result = backtest_results[result_id]
        
        # Extract price history and rebalance events
        price_history = result.get('price_history', [])
        rebalance_events = result.get('rebalance_events', [])
        
        if not price_history:
            logger.error(f"No price history found for result {result_id}")
            return
            
        # Convert to DataFrames
        price_df = pd.DataFrame(price_history)
        
        if 'timestamp' not in price_df.columns or 'price' not in price_df.columns:
            logger.error(f"Required columns not found in price history for result {result_id}")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Plot price
        plt.plot(price_df['timestamp'], price_df['price'], label='Price', color='blue', alpha=0.7)
        
        # Plot position boundaries if available
        if 'lower_price' in price_df.columns and 'upper_price' in price_df.columns:
            plt.plot(price_df['timestamp'], price_df['lower_price'], 
                    label='Lower Price', color='green', linestyle='--', alpha=0.5)
            plt.plot(price_df['timestamp'], price_df['upper_price'], 
                    label='Upper Price', color='red', linestyle='--', alpha=0.5)
        
        # Plot buffer zones if available
        if 'lower_buffer_price' in price_df.columns and 'upper_buffer_price' in price_df.columns:
            plt.plot(price_df['timestamp'], price_df['lower_buffer_price'], 
                    label='Lower Buffer', color='lightgreen', linestyle=':', alpha=0.5)
            plt.plot(price_df['timestamp'], price_df['upper_buffer_price'], 
                    label='Upper Buffer', color='lightcoral', linestyle=':', alpha=0.5)
        
        # Mark rebalance events
        if rebalance_events:
            rebalance_df = pd.DataFrame(rebalance_events)
            if 'timestamp' in rebalance_df.columns and 'price' in rebalance_df.columns:
                plt.scatter(rebalance_df['timestamp'], rebalance_df['price'], 
                          marker='^', color='purple', s=100, label='Rebalance Event', zorder=5)
        
        # Extract parameters for the title
        params = result.get('parameters', {})
        width = params.get('width', 0)
        buffer_lower = params.get('buffer_lower', 0)
        buffer_upper = params.get('buffer_upper', 0)
        cutoff_lower = params.get('cutoff_lower', 0)
        cutoff_upper = params.get('cutoff_upper', 0)
        
        plt.title(f'Price and Rebalance Events (Width={width}, Buffer={buffer_lower}/{buffer_upper}, Cutoff={cutoff_lower}/{cutoff_upper})')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_file and self.results_dir:
            output_path = os.path.join(self.results_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved rebalance events plot to {output_path}")
        
        plt.close()
    
    def generate_summary_report(
        self,
        results_df: pd.DataFrame,
        top_n: int = 5,
        output_file: str = None
    ) -> str:
        """
        Generate a text summary report of the top performing parameter sets.
        
        Args:
            results_df (pd.DataFrame): DataFrame with processed and ranked results
            top_n (int): Number of top results to include
            output_file (str, optional): Path to save the report
            
        Returns:
            str: Summary report text
        """
        if 'rank' not in results_df.columns:
            logger.warning("Results are not ranked, ranking now...")
            results_df = self.rank_results(results_df)
            
        # Limit to top N
        top_results = results_df.head(top_n)
        
        # Generate report text
        report = []
        report.append("===== USD* Pool Backtest Results Summary =====")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total backtest runs: {len(results_df)}")
        report.append("\n")
        
        report.append("===== Top Parameter Combinations =====")
        for i, row in top_results.iterrows():
            report.append(f"\nRank {int(row.get('rank', i+1))}: Overall Score: {row.get('score', 0):.4f}")
            report.append(f"  Width: {row.get('width', 0):.2f}%")
            report.append(f"  Buffer: {row.get('buffer_lower', 0):.2f}% / {row.get('buffer_upper', 0):.2f}%")
            report.append(f"  Cutoff: {row.get('cutoff_lower', 0):.4f} - {row.get('cutoff_upper', 0):.4f}")
            report.append(f"  Performance Metrics:")
            report.append(f"    - Net APR: {row.get('net_apr', 0):.2f}%")
            report.append(f"    - Fees APR: {row.get('fees_apr', 0):.2f}%")
            report.append(f"    - Rebalance Count: {int(row.get('rebalance_count', 0))}")
            report.append(f"    - Rebalance Frequency: {row.get('rebalance_frequency', 0):.2f} per day")
            report.append(f"    - Time in Position: {row.get('time_in_position_pct', 0):.2f}%")
            report.append(f"    - Sharpe Ratio: {row.get('sharpe_ratio', 0):.2f}")
        
        report.append("\n")
        report.append("===== Statistical Summary =====")
        
        numeric_cols = ['apr', 'fees_apr', 'net_apr', 'rebalance_count', 
                       'time_in_position_pct', 'total_fees_earned', 'sharpe_ratio']
        stats_df = results_df[numeric_cols].describe().transpose()
        
        for metric, row in stats_df.iterrows():
            report.append(f"\n{metric}:")
            report.append(f"  Mean: {row['mean']:.2f}")
            report.append(f"  Min: {row['min']:.2f}")
            report.append(f"  Max: {row['max']:.2f}")
            report.append(f"  Std Dev: {row['std']:.2f}")
        
        # Combine into a single string
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file and self.results_dir:
            output_path = os.path.join(self.results_dir, output_file)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved summary report to {output_path}")
        
        return report_text

if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO)
    
    # Create a results directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Create a sample result processor
    processor = ResultProcessor("test_results")
    
    # Generate sample backtest results
    from datetime import datetime, timedelta
    
    # Create sample backtest results
    backtest_results = []
    
    # Sample parameters to test
    widths = [0.03, 0.05, 0.08]
    buffer_values = [0.05, 0.1, 0.15]
    cutoff_values = [(0.95, 1.05), (0.93, 1.07), (0.9, 1.1)]
    
    for width in widths:
        for buffer in buffer_values:
            for cutoff_lower, cutoff_upper in cutoff_values:
                # Generate a random APR value biased by the parameter values
                # Just for testing - in reality this would come from backtest runs
                base_apr = 15 + np.random.normal(0, 3)
                apr = base_apr * (1 + (width - 0.05) * 10) * (1 + (buffer - 0.1) * 5)
                rebalance_count = int(30 * (1/width) * (1/buffer) * np.random.uniform(0.8, 1.2))
                
                # Create fake result structure similar to what backtest_engine would produce
                result = {
                    'parameters': {
                        'width': width,
                        'buffer_lower': buffer,
                        'buffer_upper': buffer,
                        'cutoff_lower': cutoff_lower,
                        'cutoff_upper': cutoff_upper,
                        'initial_investment': 10000,
                        'duration_days': 30
                    },
                    'metrics': {
                        'apr': apr,
                        'fees_apr': apr * 0.8,
                        'rebalance_count': rebalance_count,
                        'rebalance_frequency': rebalance_count / 30,
                        'time_in_position_pct': 90 - buffer * 200 + np.random.uniform(-5, 5),
                        'total_fees_earned': 10000 * (apr/100) * (30/365) * 0.8,
                        'total_costs': rebalance_count * 5,
                        'absolute_return': 10000 * (apr/100) * (30/365),
                        'volatility': 10 + np.random.uniform(-2, 2)
                    },
                    'price_history': [],
                    'rebalance_events': []
                }
                
                # Generate fake price history and rebalance events
                start_date = datetime.now() - timedelta(days=30)
                price = 1.0
                
                for i in range(24 * 30):  # hourly data for 30 days
                    timestamp = start_date + timedelta(hours=i)
                    
                    # Add some price movement
                    price_change = np.random.normal(0, 0.002)
                    price *= (1 + price_change)
                    
                    # Calculate position boundaries
                    base_price = 1.0
                    lower_price = base_price * (1 - width)
                    upper_price = base_price * (1 + width)
                    lower_buffer_price = lower_price * (1 - buffer)
                    upper_buffer_price = upper_price * (1 + buffer)
                    
                    # Record price point
                    result['price_history'].append({
                        'timestamp': timestamp,
                        'price': price,
                        'lower_price': lower_price,
                        'upper_price': upper_price,
                        'lower_buffer_price': lower_buffer_price,
                        'upper_buffer_price': upper_buffer_price,
                        'in_position': lower_price <= price <= upper_price
                    })
                    
                    # Maybe add a rebalance event
                    if (price < lower_buffer_price or price > upper_buffer_price) and np.random.random() > 0.7:
                        result['rebalance_events'].append({
                            'timestamp': timestamp,
                            'price': price,
                            'gas_cost': 10
                        })
                
                backtest_results.append(result)
    
    # Process results
    results_df = processor.process_backtest_results(backtest_results)
    
    # Rank results
    ranked_df = processor.rank_results(results_df)
    
    # Generate summary report
    report = processor.generate_summary_report(ranked_df, output_file="summary_report.txt")
    print("Generated summary report")
    
    # Create some plots
    processor.plot_parameter_comparison(
        ranked_df, 'width', 'net_apr', output_file="width_vs_apr.png")
    
    processor.plot_heatmap(
        ranked_df, 'width', 'buffer_avg', 'net_apr', output_file="width_buffer_heatmap.png")
    
    processor.plot_rebalance_events(
        backtest_results, 0, output_file="rebalance_events.png")
    
    print("Generated visualizations in test_results directory") 