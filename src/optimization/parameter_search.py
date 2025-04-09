"""
Parameter Search and Optimization Module

This module handles generating and optimizing parameter combinations
for USD* pool backtesting, focusing on width, buffer, and cutoff values.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from itertools import product

# Set up logger
logger = logging.getLogger(__name__)

# Import config if running directly
try:
    import config as cfg
except ImportError:
    # If called directly, try relative import
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    import config as cfg

class ParameterOptimizer:
    """Class for generating and optimizing parameter combinations."""
    
    def __init__(
        self,
        width_range: Tuple[float, float, float],
        buffer_range: Tuple[float, float, float],
        cutoff_range: Dict[str, float],
        max_combinations: int = 1000
    ):
        """
        Initialize the parameter optimizer.
        
        Args:
            width_range (Tuple[float, float, float]): Range for width parameter (min, max, step)
            buffer_range (Tuple[float, float, float]): Range for buffer parameters (min, max, step)
            cutoff_range (Dict[str, float]): Range for cutoff parameters
            max_combinations (int): Maximum number of parameter combinations to generate
        """
        self.width_range = width_range
        self.buffer_range = buffer_range
        self.cutoff_range = cutoff_range
        self.max_combinations = max_combinations
        
        logger.info(f"Initialized ParameterOptimizer with ranges: "
                    f"width={width_range}, buffer={buffer_range}, cutoff={cutoff_range}")
    
    def generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """
        Generate all parameter combinations within the specified ranges.
        
        Returns:
            List[Dict[str, float]]: List of parameter dictionaries
        """
        # Generate parameter values
        width_min, width_max, width_step = self.width_range
        buffer_min, buffer_max, buffer_step = self.buffer_range
        
        # Create arrays of parameter values
        width_values = np.arange(width_min, width_max + width_step/2, width_step)
        buffer_values = np.arange(buffer_min, buffer_max + buffer_step/2, buffer_step)
        
        # Calculate total combinations
        total_combinations = len(width_values) * len(buffer_values) * len(buffer_values)
        logger.info(f"Potential parameter combinations: {total_combinations}")
        
        # If too many combinations, reduce the number of steps
        if total_combinations > self.max_combinations:
            logger.warning(f"Too many parameter combinations ({total_combinations}). "
                           f"Reducing to approximately {self.max_combinations}.")
            
            # Reduce step size to get fewer combinations
            factor = np.sqrt(total_combinations / self.max_combinations)
            
            width_step_adjusted = width_step * factor
            width_values = np.arange(width_min, width_max + width_step_adjusted/2, width_step_adjusted)
            
            buffer_step_adjusted = buffer_step * factor
            buffer_values = np.arange(buffer_min, buffer_max + buffer_step_adjusted/2, buffer_step_adjusted)
            
            total_combinations = len(width_values) * len(buffer_values) * len(buffer_values)
            logger.info(f"Adjusted parameter combinations: {total_combinations}")
        
        # Generate all combinations
        combinations = []
        
        for width in width_values:
            for buffer_lower in buffer_values:
                for buffer_upper in buffer_values:
                    # For cutoff values, we'll use fixed values based on the buffer
                    cutoff_lower = self.cutoff_range["min_price"]
                    cutoff_upper = self.cutoff_range["max_price"]
                    
                    combinations.append({
                        "width": round(width, 4),
                        "buffer_lower": round(buffer_lower, 4),
                        "buffer_upper": round(buffer_upper, 4),
                        "cutoff_lower": round(cutoff_lower, 4),
                        "cutoff_upper": round(cutoff_upper, 4)
                    })
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def generate_adaptive_cutoffs(
        self,
        price_data: pd.DataFrame,
        percentile_range: Tuple[float, float] = (1.0, 99.0)
    ) -> Dict[str, float]:
        """
        Generate adaptive cutoff values based on historical price data.
        
        Args:
            price_data (pd.DataFrame): Historical price data
            percentile_range (Tuple[float, float]): Percentile range for cutoffs
            
        Returns:
            Dict[str, float]: Dictionary with min_price and max_price cutoffs
        """
        if 'price' not in price_data.columns:
            logger.warning("Price column not found in data, using default cutoffs")
            return self.cutoff_range
        
        min_percentile, max_percentile = percentile_range
        
        # Calculate cutoffs based on historical price percentiles
        min_price = np.percentile(price_data['price'], min_percentile)
        max_price = np.percentile(price_data['price'], max_percentile)
        
        logger.info(f"Generated adaptive cutoffs: min={min_price:.4f}, max={max_price:.4f}")
        
        return {
            "min_price": round(min_price, 4),
            "max_price": round(max_price, 4)
        }
    
    def optimize_parameters(
        self,
        backtest_results: List[Dict[str, Any]],
        objective: str = 'apr',
        weight_factors: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find optimal parameter combinations based on backtest results.
        
        Args:
            backtest_results (List[Dict[str, Any]]): Results from backtesting
            objective (str): Optimization objective ('apr', 'rebalances', 'costs', 'balance')
            weight_factors (Dict[str, float]): Optional weights for balanced optimization
            
        Returns:
            List[Dict[str, Any]]: Sorted list of parameter combinations with results
        """
        if not backtest_results:
            logger.warning("No backtest results provided for optimization")
            return []
        
        # Default weight factors for balanced optimization
        if objective == 'balance' and weight_factors is None:
            weight_factors = {
                'apr': 0.6,
                'rebalance_count': -0.2,
                'total_costs': -0.2
            }
        
        # Sort results based on the objective
        if objective == 'apr':
            sorted_results = sorted(
                backtest_results, 
                key=lambda x: x["metrics"]["apr"], 
                reverse=True
            )
        elif objective == 'rebalances':
            sorted_results = sorted(
                backtest_results, 
                key=lambda x: x["metrics"]["rebalance_count"]
            )
        elif objective == 'costs':
            sorted_results = sorted(
                backtest_results, 
                key=lambda x: x["metrics"]["total_costs"]
            )
        else:  # balance - custom formula weighting multiple factors
            # Calculate a score for each result
            for result in backtest_results:
                score = 0
                for metric, weight in weight_factors.items():
                    if metric in result["metrics"]:
                        score += result["metrics"][metric] * weight
                result["score"] = score
            
            sorted_results = sorted(
                backtest_results, 
                key=lambda x: x.get("score", 0), 
                reverse=True
            )
        
        return sorted_results
    
    def find_parameter_groups(
        self,
        backtest_results: List[Dict[str, Any]],
        top_n: int = 10,
        group_by: str = 'width'
    ) -> Dict[float, List[Dict[str, Any]]]:
        """
        Group top parameter combinations by a specific parameter.
        
        Args:
            backtest_results (List[Dict[str, Any]]): Results from backtesting
            top_n (int): Number of top results to consider
            group_by (str): Parameter to group by ('width', 'buffer_lower', 'buffer_upper')
            
        Returns:
            Dict[float, List[Dict[str, Any]]]: Grouped parameter combinations
        """
        if not backtest_results:
            logger.warning("No backtest results provided for grouping")
            return {}
        
        # Get top N results
        top_results = backtest_results[:top_n]
        
        # Group by parameter
        groups = {}
        for result in top_results:
            param_value = result["parameters"].get(group_by)
            if param_value is not None:
                if param_value not in groups:
                    groups[param_value] = []
                groups[param_value].append(result)
        
        return groups
    
    def analyze_parameter_sensitivity(
        self,
        backtest_results: List[Dict[str, Any]],
        parameter: str = 'width',
        metric: str = 'apr'
    ) -> Dict[str, Any]:
        """
        Analyze how sensitive a performance metric is to changes in a parameter.
        
        Args:
            backtest_results (List[Dict[str, Any]]): Results from backtesting
            parameter (str): Parameter to analyze ('width', 'buffer_lower', 'buffer_upper')
            metric (str): Metric to analyze ('apr', 'rebalance_count', 'total_costs')
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not backtest_results:
            logger.warning("No backtest results provided for sensitivity analysis")
            return {}
        
        # Extract parameter values and corresponding metrics
        param_values = []
        metric_values = []
        
        for result in backtest_results:
            param_value = result["parameters"].get(parameter)
            metric_value = result["metrics"].get(metric)
            
            if param_value is not None and metric_value is not None:
                param_values.append(param_value)
                metric_values.append(metric_value)
        
        if not param_values:
            logger.warning(f"No valid data for sensitivity analysis of {parameter} vs {metric}")
            return {}
        
        # Convert to numpy arrays
        param_array = np.array(param_values)
        metric_array = np.array(metric_values)
        
        # Calculate statistics
        param_unique = np.unique(param_array)
        metric_by_param = {}
        
        for value in param_unique:
            indices = param_array == value
            metric_by_param[float(value)] = metric_array[indices].tolist()
        
        # Calculate correlation
        correlation = np.corrcoef(param_array, metric_array)[0, 1]
        
        # Calculate optimal parameter value
        if metric == 'apr':
            # For APR, we want to maximize
            optimal_indices = np.argmax(metric_array)
        else:
            # For costs and rebalances, we want to minimize
            optimal_indices = np.argmin(metric_array)
        
        optimal_value = param_array[optimal_indices]
        
        return {
            'parameter': parameter,
            'metric': metric,
            'correlation': correlation,
            'optimal_value': float(optimal_value),
            'metric_by_param': metric_by_param,
            'statistics': {
                'mean': float(np.mean(metric_array)),
                'std': float(np.std(metric_array)),
                'min': float(np.min(metric_array)),
                'max': float(np.max(metric_array))
            }
        }

if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    optimizer = ParameterOptimizer(
        width_range=(0.01, 0.5, 0.01),
        buffer_range=(0, 1.0, 0.1),
        cutoff_range={"min_price": 0.95, "max_price": 1.05}
    )
    
    # Generate parameter combinations
    combinations = optimizer.generate_parameter_combinations()
    print(f"Generated {len(combinations)} parameter combinations")
    print("Sample combinations:")
    for combo in combinations[:5]:
        print(combo) 