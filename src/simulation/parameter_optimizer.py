#!/usr/bin/env python3
"""
Parameter Optimizer for vfat Strategy

This script finds optimal parameter combinations for the vfat automated liquidity 
providing strategy. It tests multiple width, buffer, and cutoff values to find the 
best performing combination based on metrics like APR or performance vs HODL.

Usage:
    python parameter_optimizer.py --pair USDC.e/scUSD --amount 1000 --days 30
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from itertools import product
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import time
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simulator directly without relying on simulation package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vfat_simulator import VfatSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"parameter_optimization_{int(time.time())}.log")
    ]
)
logger = logging.getLogger("parameter_optimizer")

class ParameterOptimizer:
    """
    Optimizer for vfat strategy parameters.
    
    Finds optimal parameters for the vfat strategy by running multiple backtests
    with different parameter combinations and comparing the results.
    """
    
    def __init__(
        self,
        pair: str = "USDC.e/scUSD",
        initial_amount: float = 1000.0,
        days: int = 30,
        width_range: Tuple[float, float, float] = (0.5, 5.0, 0.5),  # min, max, step
        buffer_range: Tuple[float, float, float] = (0.1, 0.5, 0.1),  # min, max, step
        cutoff_range: Tuple[float, float, float] = (0.5, 5.0, 0.5),  # min, max, step
        max_dust_range: Tuple[float, float, float] = (0.1, 1.0, 0.2),  # min, max, step
        max_slippage_range: Tuple[float, float, float] = (0.5, 1.0, 0.2),  # min, max, step
        max_price_impact_range: Tuple[float, float, float] = (0.5, 1.0, 0.2),  # min, max, step
        strategy: str = "grid_search",
        combinations: int = 100,
        jobs: int = -1,
        metric: str = "score",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the parameter optimizer.
        
        Args:
            pair: Trading pair
            initial_amount: Initial investment amount
            days: Number of days to backtest
            width_range: Range of width values to test (min, max, step)
            buffer_range: Range of buffer values to test (min, max, step)
            cutoff_range: Range of cutoff values to test (min, max, step)
            max_dust_range: Range of max dust percentage values to test (min, max, step)
            max_slippage_range: Range of max slippage percentage values to test (min, max, step)
            max_price_impact_range: Range of max price impact percentage values to test (min, max, step)
            strategy: Optimization strategy ("grid_search", "random_search", "latin_hypercube")
            combinations: Number of parameter combinations to test (for random and latin hypercube)
            jobs: Number of parallel jobs to run (-1 for all cores)
            metric: Metric to optimize for ("position_apr", "position_roi", "performance_vs_hodl", "score", "final_position_value")
            logger: Optional logger for logging
        """
        self.pair = pair
        self.amount = initial_amount
        self.days = days
        
        self.width_range = width_range
        self.buffer_range = buffer_range
        self.cutoff_range = cutoff_range
        self.max_dust_range = max_dust_range
        self.max_slippage_range = max_slippage_range
        self.max_price_impact_range = max_price_impact_range
        
        self.strategy = strategy
        self.combinations = combinations
        self.jobs = jobs if jobs > 0 else multiprocessing.cpu_count()
        
        # Validate and set the optimization metric
        valid_metrics = [
            "position_apr", "position_roi", "performance_vs_hodl", 
            "score", "final_position_value"
        ]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
        self.metric = metric
        
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger("parameter_optimizer")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger
        
        # Initialize results storage
        self.parameter_combinations = []
        self.results = []
        self.best_params = None
        self.best_result = None
        self.historical_data = None
        self.daily_metrics_by_params = {}  # Store detailed daily metrics for each parameter set
        
        self.logger.info(f"Initialized parameter optimizer for {pair} with ${initial_amount} over {days} days")

    def run_optimization(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run the parameter optimization process.
        
        Returns:
            Tuple containing:
                - Dictionary with best parameters and results
                - List of all parameter combinations and their results
        """
        # Generate parameter combinations
        self._generate_parameter_combinations()
        
        # Fetch historical data (once for all backtests)
        self._fetch_historical_data()
        
        # Run backtests with different parameter combinations
        self._run_backtests()
        
        # Find best parameters based on selected metric
        self._find_best_parameters()
        
        # Print top results for user analysis
        self._print_top_results()
        
        # Save detailed metrics for the best parameter set
        self._save_detailed_metrics()
        
        # Create plots
        self._create_plots()
        
        # Save results to CSV
        self._save_results_to_csv()
        
        return self.best_params, self.results
    
    def _save_detailed_metrics(self) -> None:
        """Save detailed metrics for the best parameter set"""
        if self.best_params is None:
            self.logger.warning("No best parameters found. Cannot save detailed metrics.")
            return
        
        # Check if we have daily metrics for the best parameters
        best_params_key = self._params_to_key(self.best_params)
        if best_params_key not in self.daily_metrics_by_params:
            self.logger.warning(f"No daily metrics available for the best parameters {best_params_key}")
            return
        
        # Save detailed metrics to CSV
        timestamp = int(time.time())
        filename = f"vfat_detailed_metrics_{self.pair.replace('/', '_')}_best_params_{timestamp}.csv"
        self.daily_metrics_by_params[best_params_key].to_csv(filename, index=False)
        self.logger.info(f"Saved detailed metrics for best parameters to {filename}")
    
    def _params_to_key(self, params: Dict[str, Any]) -> str:
        """Convert parameter dictionary to a string key"""
        return f"w{params['width']}_b{params['buffer']}_c{params['cutoff']}_d{params['max_dust_percent']}_s{params['max_slippage']}_p{params['swap_max_price_impact']}"
            
    def _backtest_and_collect_metrics(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """Run a backtest with the given parameters and collect detailed metrics"""
        # Initialize simulator with the parameters
        simulator = VfatSimulator(
            pair=self.pair,
            initial_amount=self.amount,
            backtest_days=self.days,
            width=params['width'],
            buffer=params['buffer'],
            cutoff=params['cutoff'],
            max_dust_percent=params['max_dust_percent'],
            max_slippage=params['max_slippage'],
            swap_max_price_impact=params['swap_max_price_impact']
        )
        
        # Use preloaded historical data
        if self.historical_data is not None:
            simulator.data = self.historical_data.copy()
        
        # Run the simulation
        try:
            simulator.calculate_hodl_value()
            simulator.simulate_vfat_strategy()
            
            # Get the results
            results = simulator.results
            
            # Get detailed daily metrics
            daily_metrics = simulator.get_daily_metrics()
            
            # Calculate a composite score based on multiple metrics
            # We want higher APR, higher ROI, and better performance vs HODL
            position_apr = results.get('position_apr', -100)
            position_roi = results.get('position_roi', -100)
            vs_hodl = ((results.get('final_position_value', 0) / max(results.get('hodl_value', 1), 1)) - 1) * 100
            
            # Add performance_vs_hodl to results explicitly
            results['performance_vs_hodl'] = vs_hodl
            
            # Score is weighted sum: 40% APR, 40% vs HODL, 20% ROI
            score = 0.4 * position_apr + 0.4 * vs_hodl + 0.2 * position_roi
            
            # Add score to results
            results['score'] = score
            
            return results, daily_metrics
            
        except Exception as e:
            self.logger.error(f"Error running simulation with params {params}: {str(e)}")
            return {
                'position_apr': -100,
                'position_roi': -100,
                'performance_vs_hodl': -100,
                'score': -1000,
                'final_position_value': 0,
                'error': str(e)
            }, None
    
    def _run_single_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest with the given parameters"""
        results, daily_metrics = self._backtest_and_collect_metrics(params)
        
        # Store daily metrics for this parameter set
        params_key = self._params_to_key(params)
        if daily_metrics is not None:
            self.daily_metrics_by_params[params_key] = daily_metrics
        
        # Add parameters to results for easy reference
        results.update({
            'width': params['width'],
            'buffer': params['buffer'],
            'cutoff': params['cutoff'],
            'max_dust_percent': params['max_dust_percent'],
            'max_slippage': params['max_slippage'],
            'swap_max_price_impact': params['swap_max_price_impact']
        })
        
        return results

    def _generate_parameter_combinations(self) -> None:
        """Generate parameter combinations based on the selected strategy"""
        if self.strategy == "grid_search":
            self._generate_grid_search_combinations()
        elif self.strategy == "random_search":
            self._generate_random_search_combinations()
        elif self.strategy == "latin_hypercube":
            self._generate_latin_hypercube_combinations()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.logger.info(f"Generated {len(self.parameter_combinations)} parameter combinations using {self.strategy}")
    
    def _generate_grid_search_combinations(self) -> None:
        """Generate parameter combinations using grid search"""
        width_min, width_max, width_step = self.width_range
        buffer_min, buffer_max, buffer_step = self.buffer_range
        cutoff_min, cutoff_max, cutoff_step = self.cutoff_range
        dust_min, dust_max, dust_step = self.max_dust_range
        slippage_min, slippage_max, slippage_step = self.max_slippage_range
        impact_min, impact_max, impact_step = self.max_price_impact_range
        
        # Create parameter grids
        widths = np.arange(width_min, width_max + width_step/2, width_step)
        buffers = np.arange(buffer_min, buffer_max + buffer_step/2, buffer_step)
        cutoffs = np.arange(cutoff_min, cutoff_max + cutoff_step/2, cutoff_step)
        dust_values = np.arange(dust_min, dust_max + dust_step/2, dust_step)
        slippage_values = np.arange(slippage_min, slippage_max + slippage_step/2, slippage_step)
        impact_values = np.arange(impact_min, impact_max + impact_step/2, impact_step)
        
        # Generate all combinations
        parameter_combinations = []
        for width in widths:
            for buffer in buffers:
                # Buffer should be smaller than width
                if buffer >= width:
                    continue
                    
                for cutoff in cutoffs:
                    for dust in dust_values:
                        for slippage in slippage_values:
                            for impact in impact_values:
                                parameter_combinations.append({
                                    "width": round(width, 2),
                                    "buffer": round(buffer, 2),
                                    "cutoff": round(cutoff, 2),
                                    "max_dust_percent": round(dust, 2),
                                    "max_slippage": round(slippage, 2),
                                    "swap_max_price_impact": round(impact, 2)
                                })
        
        self.parameter_combinations = parameter_combinations
    
    def _generate_random_search_combinations(self) -> None:
        """Generate parameter combinations using random search"""
        width_min, width_max, _ = self.width_range
        buffer_min, buffer_max, _ = self.buffer_range
        cutoff_min, cutoff_max, _ = self.cutoff_range
        dust_min, dust_max, _ = self.max_dust_range
        slippage_min, slippage_max, _ = self.max_slippage_range
        impact_min, impact_max, _ = self.max_price_impact_range
        
        np.random.seed(42)  # For reproducibility
        
        parameter_combinations = []
        attempts = 0
        max_attempts = self.combinations * 10  # Safety to avoid infinite loops
        
        while len(parameter_combinations) < self.combinations and attempts < max_attempts:
            attempts += 1
            
            width = np.random.uniform(width_min, width_max)
            buffer = np.random.uniform(buffer_min, buffer_max)
            
            # Ensure buffer < width
            if buffer >= width:
                continue
                
            cutoff = np.random.uniform(cutoff_min, cutoff_max)
            dust = np.random.uniform(dust_min, dust_max)
            slippage = np.random.uniform(slippage_min, slippage_max)
            impact = np.random.uniform(impact_min, impact_max)
            
            parameter_combinations.append({
                "width": round(width, 2),
                "buffer": round(buffer, 2),
                "cutoff": round(cutoff, 2),
                "max_dust_percent": round(dust, 2),
                "max_slippage": round(slippage, 2),
                "swap_max_price_impact": round(impact, 2)
            })
        
        self.parameter_combinations = parameter_combinations
    
    def _generate_latin_hypercube_combinations(self) -> None:
        """Generate parameter combinations using Latin Hypercube Sampling for better coverage"""
        try:
            from scipy.stats import qmc
            
            width_min, width_max, _ = self.width_range
            buffer_min, buffer_max, _ = self.buffer_range
            cutoff_min, cutoff_max, _ = self.cutoff_range
            dust_min, dust_max, _ = self.max_dust_range
            slippage_min, slippage_max, _ = self.max_slippage_range
            impact_min, impact_max, _ = self.max_price_impact_range
            
            # Create Latin Hypercube Sampler
            sampler = qmc.LatinHypercube(d=6, seed=42)  # 6 parameters
            
            # Generate samples
            sample = sampler.random(n=self.combinations * 2)  # Generate extra samples to filter out invalid ones
            
            # Scale the samples to the parameter ranges
            lower_bounds = [width_min, buffer_min, cutoff_min, dust_min, slippage_min, impact_min]
            upper_bounds = [width_max, buffer_max, cutoff_max, dust_max, slippage_max, impact_max]
            scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
            
            parameter_combinations = []
            for params in scaled_sample:
                width = params[0]
                buffer = params[1]
                
                # Ensure buffer < width
                if buffer >= width:
                    continue
                    
                cutoff = params[2]
                dust = params[3]
                slippage = params[4]
                impact = params[5]
                
                parameter_combinations.append({
                    "width": round(width, 2),
                    "buffer": round(buffer, 2),
                    "cutoff": round(cutoff, 2),
                    "max_dust_percent": round(dust, 2),
                    "max_slippage": round(slippage, 2),
                    "swap_max_price_impact": round(impact, 2)
                })
                
                if len(parameter_combinations) >= self.combinations:
                    break
            
            self.parameter_combinations = parameter_combinations
            
        except ImportError:
            self.logger.warning("SciPy not available for Latin Hypercube Sampling, falling back to random search")
            self._generate_random_search_combinations()
    
    def _fetch_historical_data(self) -> None:
        """Fetch historical data once for all backtests"""
        self.logger.info("Fetching historical data once for all backtests")
        
        # Create a temporary simulator to fetch data
        simulator = VfatSimulator(
            pair=self.pair,
            initial_amount=self.amount,
            backtest_days=self.days
        )
        
        # Fetch historical data
        simulator.data = simulator.fetch_pool_historical_data()
        
        # Store the data for reuse in backtests
        if simulator.data is not None and not simulator.data.empty:
            self.historical_data = simulator.data.copy()
            self.logger.info(f"Successfully fetched {len(simulator.data)} days of historical data")
        else:
            self.logger.error("Failed to fetch historical data")
            self.historical_data = pd.DataFrame()
    
    def _run_backtests(self) -> None:
        """Run backtests for all parameter combinations"""
        if not self.parameter_combinations:
            self.logger.warning("No parameter combinations to test")
            return
            
        self.logger.info(f"Running {len(self.parameter_combinations)} backtests using {self.jobs} CPU cores")
        
        # Run backtests
        if self.jobs > 1:
            with multiprocessing.Pool(self.jobs) as pool:
                self.results = list(tqdm(pool.imap(self._run_single_backtest, self.parameter_combinations), 
                                         total=len(self.parameter_combinations)))
        else:
            self.results = [self._run_single_backtest(params) for params in tqdm(self.parameter_combinations)]
    
    def _find_best_parameters(self) -> None:
        """Find the best parameters based on the selected metric"""
        if not self.results:
            self.logger.warning("No results to analyze")
            return
            
        # Filter out results with errors
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            self.logger.warning("No valid results found")
            return
            
        # Find the best result based on the selected metric
        if self.metric in ['position_apr', 'position_roi', 'performance_vs_hodl', 'score', 'final_position_value']:
            self.best_result = max(valid_results, key=lambda x: x.get(self.metric, -float('inf')))
        else:
            self.logger.warning(f"Unknown metric: {self.metric}")
            self.best_result = valid_results[0]
            
        self.best_params = self.best_result
        
        self.logger.info(f"Best parameters found: width={self.best_params.get('width')}, "
                         f"buffer={self.best_params.get('buffer')}, cutoff={self.best_params.get('cutoff')}")
        self.logger.info(f"Best {self.metric}: {self.best_params.get(self.metric)}")
    
    def _print_top_results(self) -> None:
        """Print the top 10 parameter combinations based on the selected metric"""
        if not self.results:
            self.logger.warning("No results to print")
            return
            
        # Filter out results with errors
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            self.logger.warning("No valid results to print")
            return
            
        # Sort by the selected metric
        sorted_results = sorted(valid_results, key=lambda x: x.get(self.metric, -float('inf')), reverse=True)
        
        # Get top 10 or all if less than 10
        top_n = min(10, len(sorted_results))
        top_results = sorted_results[:top_n]
        
        # Print header
        print("\nTOP 10 PARAMETER COMBINATIONS BY", self.metric.upper())
        print("=" * 60)
        
        # Print each result
        for i, result in enumerate(top_results, 1):
            print(f"{i}. Width: {result.get('width')}%, Buffer: {result.get('buffer')}%, Cutoff: {result.get('cutoff')}%")
            
            # Performance metrics
            position_apr = result.get('position_apr', 'N/A')
            position_roi = result.get('position_roi', 'N/A')
            vs_hodl = result.get('performance_vs_hodl', 'N/A')
            
            if isinstance(position_apr, (int, float)):
                print(f"   APR: {position_apr:.2f}%", end=", ")
            else:
                print(f"   APR: {position_apr}", end=", ")
                
            if isinstance(position_roi, (int, float)):
                print(f"ROI: {position_roi:.2f}%", end=", ")
            else:
                print(f"ROI: {position_roi}", end=", ")
                
            if isinstance(vs_hodl, (int, float)):
                print(f"vs HODL: {vs_hodl:.2f}%")
            else:
                print(f"vs HODL: {vs_hodl}")
            
            # Rebalance info
            rebalance_count = result.get('rebalance_count', 'N/A')
            rebalance_costs = result.get('rebalance_costs', 0)
            print(f"   Rebalances: {rebalance_count}", end=", ")
            
            if isinstance(rebalance_count, int) and rebalance_count > 0 and isinstance(self.days, int):
                avg_days_between = self.days / rebalance_count
                print(f"Avg {avg_days_between:.1f} days between", end=", ")
            
            if isinstance(rebalance_costs, (int, float)):
                print(f"Cost: ${rebalance_costs:.2f}")
            else:
                print(f"Cost: {rebalance_costs}")
            
            # Value and Score
            final_value = result.get('final_position_value', 'N/A')
            hodl_value = result.get('hodl_value', 'N/A')
            score = result.get('score', 'N/A')
            
            if isinstance(final_value, (int, float)) and isinstance(hodl_value, (int, float)):
                print(f"   Final Value: ${final_value:.2f}, HODL Value: ${hodl_value:.2f}")
            else:
                print(f"   Final Value: {final_value}, HODL Value: {hodl_value}")
                
            if isinstance(score, (int, float)):
                print(f"   Score: {score:.2f}")
            else:
                print(f"   Score: {score}")
            
            # Fees, value lost, and in-range data
            fees_earned = result.get('fees_earned', 0)
            value_lost = result.get('value_lost_on_rebalances', rebalance_costs)  # Fall back to rebalance_costs if not available
            net_value_gain = fees_earned - value_lost
            in_range_pct = result.get('percent_in_range', 0)
            
            if isinstance(fees_earned, (int, float)):
                print(f"   Fees Earned: ${fees_earned:.2f}", end=", ")
            else:
                print(f"   Fees Earned: {fees_earned}", end=", ")
            
            if isinstance(value_lost, (int, float)):
                print(f"Value Lost: ${value_lost:.2f}", end=", ")
            else:
                print(f"Value Lost: {value_lost}", end=", ")
                
            if isinstance(net_value_gain, (int, float)):
                print(f"Net Value: ${net_value_gain:.2f}")
            else:
                print(f"Net Value: {net_value_gain}")
                
            if isinstance(in_range_pct, (int, float)):
                print(f"   Time In Range: {in_range_pct:.2f}%")
            else:
                print(f"Time In Range: {in_range_pct}")
                
            print("   " + "-" * 50)
    
    def _create_plots(self) -> None:
        """Create plots of the results"""
        if not self.results:
            self.logger.warning("No results to plot")
            return
            
        # Filter out results with errors
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            self.logger.warning("No valid results to plot")
            return
            
        # Create DataFrame for plotting
        results_df = pd.DataFrame(valid_results)
        
        # Sort by the selected metric
        results_df = results_df.sort_values(by=self.metric, ascending=False)
        
        # Plot the top parameters by the selected metric
        plt.figure(figsize=(14, 8))
        
        # Get top 20 results or all if less than 20
        top_n = min(20, len(results_df))
        top_results = results_df.head(top_n)
        
        # Create x-axis labels with parameter values
        param_labels = [f"W:{row['width']}\nB:{row['buffer']}\nC:{row['cutoff']}" 
                        for _, row in top_results.iterrows()]
        
        # Create bar plot
        plt.bar(range(top_n), top_results[self.metric])
        plt.xticks(range(top_n), param_labels, rotation=45)
        plt.title(f"Top {top_n} Parameter Combinations by {self.metric}")
        plt.ylabel(self.metric)
        plt.tight_layout()
        
        # Save plot
        output_file = f"parameter_optimization_{self.pair.replace('/', '_')}_{self.metric}_{int(time.time())}.png"
        plt.savefig(output_file)
        plt.close()
        
        self.logger.info(f"Saved results plot to {output_file}")
    
    def _save_results_to_csv(self) -> None:
        """Save all results to a CSV file"""
        if not self.results:
            self.logger.warning("No results to save")
            return
            
        # Create DataFrame for saving
        results_df = pd.DataFrame(self.results)
        
        # Save to CSV
        output_file = f"parameter_optimization_{self.pair.replace('/', '_')}_{int(time.time())}.csv"
        results_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Saved complete results to {output_file}")

    def _print_optimizer_results(self, results):
        """
        Pretty-print the optimization results
        """
        if not results or len(results) == 0:
            logger.warning("No valid results to display")
            return
        
        # Sort results by the chosen metric
        sorted_results = sorted(results, key=lambda x: -x[self.metric])
        best_result = sorted_results[0]
        
        # Print the top 10 parameter combinations
        print("\nTOP 10 PARAMETER COMBINATIONS BY {}".format(self.metric.upper()))
        print("=" * 60)
        
        for i, result in enumerate(sorted_results[:10]):
            # Ensure all values are displayed even if they're strings
            width = float(result.get('width', 0))
            buffer = float(result.get('buffer', 0))
            cutoff = float(result.get('cutoff', 0))
            
            position_apr = float(result.get('position_apr', 0))
            position_roi = float(result.get('position_roi', 0))
            vs_hodl = float(result.get('performance_vs_hodl', 0))
            final_value = float(result.get('final_position_value', 0))
            hodl_value = float(result.get('hodl_value', 0))
            
            # Additional metrics
            rebalance_count = int(result.get('rebalance_count', 0))
            avg_days_between = float(result.get('days_passed', 30)) / rebalance_count if rebalance_count > 0 else float('inf')
            if avg_days_between == float('inf'):
                avg_days_between_str = "N/A"
            else:
                avg_days_between_str = f"{avg_days_between:.1f}"
            
            rebalance_costs = float(result.get('rebalance_costs', 0))
            fees_earned = float(result.get('fees_earned', 0))
            value_lost = float(result.get('value_lost_on_rebalances', 0))
            net_value = fees_earned - value_lost
            percent_in_range = float(result.get('percent_in_range', 0))
            score = float(result.get('score', 0))
            
            print(f"{i+1}. Width: {width:.1f}%, Buffer: {buffer:.1f}%, Cutoff: {cutoff:.1f}%")
            print(f"   APR: {position_apr:.2f}%, ROI: {position_roi:.2f}%, vs HODL: {vs_hodl:.2f}%")
            print(f"   Rebalances: {rebalance_count}, Avg {avg_days_between_str} days between, Cost: ${rebalance_costs:.2f}")
            print(f"   Final Value: ${final_value:.2f}, HODL Value: ${hodl_value:.2f}")
            print(f"   Score: {score:.2f}")
            print(f"   Fees Earned: ${fees_earned:.2f}, Value Lost: ${value_lost:.2f}, Net Value: ${net_value:.2f}")
            print(f"   Time In Range: {percent_in_range:.2f}%")
            print("   " + "-" * 50)
        
        # Create and print a more detailed summary for the best parameters
        width = float(best_result.get('width', 0))
        buffer = float(best_result.get('buffer', 0))
        cutoff = float(best_result.get('cutoff', 0))
        max_dust = float(best_result.get('max_dust_percent', 0))
        max_slippage = float(best_result.get('max_slippage', 0))
        max_price_impact = float(best_result.get('swap_max_price_impact', 0))
        
        position_apr = float(best_result.get('position_apr', 0))
        position_roi = float(best_result.get('position_roi', 0))
        vs_hodl = float(best_result.get('performance_vs_hodl', 0))
        final_value = float(best_result.get('final_position_value', 0))
        
        # Additional metrics for best result
        rebalance_count = int(best_result.get('rebalance_count', 0))
        days_passed = float(best_result.get('days_passed', 30))
        avg_days_between = days_passed / rebalance_count if rebalance_count > 0 else "N/A"
        if avg_days_between == "N/A":
            avg_days_between_str = "N/A"
        else:
            avg_days_between_str = f"{avg_days_between:.1f}"
        
        rebalance_costs = float(best_result.get('rebalance_costs', 0))
        avg_cost_per_rebalance = rebalance_costs / rebalance_count if rebalance_count > 0 else 0
        
        fees_earned = float(best_result.get('fees_earned', 0))
        fee_apr = float(best_result.get('fee_apr', 0))
        
        value_lost = float(best_result.get('value_lost_on_rebalances', 0))
        net_value = fees_earned - value_lost
        
        # Calculate net value APR (properly annualized)
        net_roi = (net_value / self.amount) * 100
        net_apr = ((1 + net_roi / 100) ** (365 / days_passed) - 1) * 100 if days_passed > 0 else 0
        
        percent_in_range = float(best_result.get('percent_in_range', 0))
        
        print("\nOPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best Parameters Found:")
        print(f"  - Width: {width:.1f}%")
        print(f"  - Buffer: {buffer:.1f}%")
        print(f"  - Cutoff: {cutoff:.1f}%")
        print(f"Max Dust Percentage: {max_dust:.1f}%")
        print(f"Max Slippage Percentage: {max_slippage:.1f}%")
        print(f"Max Price Impact Percentage: {max_price_impact:.1f}%")
        print(f"Performance:")
        print(f"  - APR: {position_apr:.2f}%")
        print(f"  - ROI: {position_roi:.2f}%")
        print(f"  - vs HODL: {vs_hodl:.2f}%")
        print(f"  - Final Value: ${final_value:.2f}")
        print(f"  - Total Fees Earned: ${fees_earned:.2f}")
        print(f"  - Fee APR: {fee_apr:.2f}%") 
        print(f"  - Rebalance Count: {rebalance_count}")
        print(f"  - Avg Days Between Rebalances: {avg_days_between_str}")
        print(f"  - Total Rebalance Costs: ${rebalance_costs:.2f}")
        print(f"  - Avg Cost Per Rebalance: ${avg_cost_per_rebalance:.2f}")
        print(f"  - Value Lost on Rebalances: ${value_lost:.2f}")
        print(f"  - Net Value (Fees - Costs): ${net_value:.2f}")
        print(f"  - Net Value APR: {net_apr:.2f}%")
        print(f"  - Time In Range: {percent_in_range:.2f}%")
        print("=" * 60)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='vfat Strategy Parameter Optimizer')
    
    # Trading pair and investment amount
    parser.add_argument('--pair', type=str, default="USDC.e/scUSD",
                        help='Trading pair to optimize')
    parser.add_argument('--amount', type=float, default=1000.0,
                        help='Initial investment amount in USD')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest')
    
    # Width parameter range
    parser.add_argument('--width-min', type=float, default=0.5,
                        help='Minimum width percentage to test')
    parser.add_argument('--width-max', type=float, default=5.0,
                        help='Maximum width percentage to test')
    parser.add_argument('--width-step', type=float, default=0.5,
                        help='Step size for width percentage range')
    
    # Buffer parameter range
    parser.add_argument('--buffer-min', type=float, default=0.1,
                        help='Minimum buffer percentage to test')
    parser.add_argument('--buffer-max', type=float, default=0.5,
                        help='Maximum buffer percentage to test')
    parser.add_argument('--buffer-step', type=float, default=0.1,
                        help='Step size for buffer percentage range')
    
    # Cutoff parameter range
    parser.add_argument('--cutoff-min', type=float, default=0.5,
                        help='Minimum cutoff percentage to test')
    parser.add_argument('--cutoff-max', type=float, default=5.0,
                        help='Maximum cutoff percentage to test')
    parser.add_argument('--cutoff-step', type=float, default=0.5,
                        help='Step size for cutoff percentage range')
    
    # Max dust parameter range
    parser.add_argument('--max-dust-min', type=float, default=0.1,
                        help='Minimum max dust percentage to test')
    parser.add_argument('--max-dust-max', type=float, default=1.0,
                        help='Maximum max dust percentage to test')
    parser.add_argument('--max-dust-step', type=float, default=0.2,
                        help='Step size for max dust percentage range')
    
    # Max slippage parameter range
    parser.add_argument('--max-slippage-min', type=float, default=0.5,
                        help='Minimum max slippage percentage to test')
    parser.add_argument('--max-slippage-max', type=float, default=1.0,
                        help='Maximum max slippage percentage to test')
    parser.add_argument('--max-slippage-step', type=float, default=0.2,
                        help='Step size for max slippage percentage range')
    
    # Max price impact parameter range
    parser.add_argument('--max-price-impact-min', type=float, default=0.5,
                        help='Minimum max price impact percentage to test')
    parser.add_argument('--max-price-impact-max', type=float, default=1.0,
                        help='Maximum max price impact percentage to test')
    parser.add_argument('--max-price-impact-step', type=float, default=0.2,
                        help='Step size for max price impact percentage range')
    
    # Optimization strategy
    parser.add_argument('--strategy', type=str, choices=['grid_search', 'random_search', 'latin_hypercube'],
                        default='grid_search', help='Optimization strategy')
    parser.add_argument('--combinations', type=int, default=100,
                        help='Number of parameter combinations to test (for random and latin hypercube)')
    
    # Parallelization
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of parallel jobs to run (-1 for all cores)')
    
    # Metric to optimize for
    parser.add_argument('--metric', type=str, 
                        choices=['position_apr', 'position_roi', 'performance_vs_hodl', 'score', 'final_position_value'],
                        default='score', help='Metric to optimize for')
    
    # Additional options
    parser.add_argument('--export-pool-data', action='store_true',
                        help='Export detailed pool data for each parameter set')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('parameter_optimizer')
        
        # Print optimization setup
        print("\nVFAT STRATEGY PARAMETER OPTIMIZER")
        print("=" * 60)
        print(f"Pair: {args.pair}")
        print(f"Initial Investment: ${args.amount:.2f}")
        print(f"Backtest Period: {args.days} days")
        print(f"Parameter Ranges:")
        print(f"  - Width: {args.width_min} to {args.width_max} (step: {args.width_step})")
        print(f"  - Buffer: {args.buffer_min} to {args.buffer_max} (step: {args.buffer_step})")
        print(f"  - Cutoff: {args.cutoff_min} to {args.cutoff_max} (step: {args.cutoff_step})")
        print(f"Max Dust Percentage: {args.max_dust_min} to {args.max_dust_max} (step: {args.max_dust_step})")
        print(f"Max Slippage Percentage: {args.max_slippage_min} to {args.max_slippage_max} (step: {args.max_slippage_step})")
        print(f"Max Price Impact Percentage: {args.max_price_impact_min} to {args.max_price_impact_max} (step: {args.max_price_impact_step})")
        print(f"Strategy: {args.strategy}")
        print(f"Optimizing for: {args.metric}")
        print("=" * 60)
        
        # Confirm if user wants to proceed
        confirm = input("Proceed with optimization? (y/n): ").lower()
        if confirm != 'y':
            print("Optimization aborted")
            return 0
        
        # Create and run optimizer
        optimizer = ParameterOptimizer(
            pair=args.pair,
            initial_amount=args.amount,
            days=args.days,
            width_range=(args.width_min, args.width_max, args.width_step),
            buffer_range=(args.buffer_min, args.buffer_max, args.buffer_step),
            cutoff_range=(args.cutoff_min, args.cutoff_max, args.cutoff_step),
            max_dust_range=(args.max_dust_min, args.max_dust_max, args.max_dust_step),
            max_slippage_range=(args.max_slippage_min, args.max_slippage_max, args.max_slippage_step),
            max_price_impact_range=(args.max_price_impact_min, args.max_price_impact_max, args.max_price_impact_step),
            strategy=args.strategy,
            combinations=args.combinations,
            jobs=args.jobs,
            metric=args.metric,
            logger=logger
        )
        
        best_params, all_results = optimizer.run_optimization()
        
        # If export pool data option is enabled, save detailed metrics for all parameter sets
        if args.export_pool_data:
            timestamp = int(time.time())
            pool_data_dir = f"pool_data_{args.pair.replace('/', '_')}_{timestamp}"
            os.makedirs(pool_data_dir, exist_ok=True)
            
            for params_key, daily_metrics in optimizer.daily_metrics_by_params.items():
                if daily_metrics is not None:
                    filename = os.path.join(pool_data_dir, f"metrics_{params_key}.csv")
                    daily_metrics.to_csv(filename, index=False)
            
            logger.info(f"Exported pool data for all parameter sets to {pool_data_dir}")
        
        # Print optimization summary
        optimizer._print_optimizer_results(all_results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 