"""
Simulation Module

This module provides the core backtesting and simulation functionality
for USD* pool performance analysis and optimization.
"""

from .backtest_engine import BacktestEngine
from .result_processor import ResultProcessor
import config as cfg

__all__ = [
    'BacktestEngine',
    'ResultProcessor',
    'run_backtest_suite',
    'get_config'
]


def get_config():
    """
    Return the current configuration.
    
    Returns:
        dict: Configuration dictionary
    """
    return cfg.get_config()


def run_backtest_suite(
    price_data,
    pool_data, 
    parameter_combinations,
    initial_investment=None,
    transaction_costs=None
):
    """
    Run a suite of backtests with different parameter combinations.
    
    Args:
        price_data: DataFrame with price data
        pool_data: Dictionary with pool configuration and performance data
        parameter_combinations: List of parameter dictionaries with width, buffer, and cutoff values
        initial_investment: Initial investment amount (defaults to config value)
        transaction_costs: Whether to include transaction costs (defaults to config value)
        
    Returns:
        tuple: (list of backtest results, processed results DataFrame, ranked results DataFrame)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get default values from config if not specified
    config = get_config()
    if initial_investment is None:
        initial_investment = config["default_backtest_params"]["initial_investment"]
    if transaction_costs is None:
        transaction_costs = config["default_backtest_params"]["transaction_costs"]
        
    gas_settings = config["gas_settings"]
    
    # Initialize the engine
    engine = BacktestEngine(price_data, pool_data, gas_settings)
    
    # Initialize result processor with the results directory from config
    results_dir = config["results_dir"]
    processor = ResultProcessor(results_dir)
    
    # Run backtests with all parameter combinations
    results = []
    logger.info(f"Running {len(parameter_combinations)} backtest scenarios")
    
    for i, params in enumerate(parameter_combinations):
        logger.info(f"Running backtest {i+1}/{len(parameter_combinations)}")
        
        # Extract parameters
        width = params.get("width")
        buffer_lower = params.get("buffer_lower")
        buffer_upper = params.get("buffer_upper", buffer_lower)  # Use lower as default if upper not specified
        cutoff_lower = params.get("cutoff_lower")
        cutoff_upper = params.get("cutoff_upper")
        
        # Run backtest
        result = engine.run_backtest(
            width=width,
            buffer_lower=buffer_lower,
            buffer_upper=buffer_upper,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            initial_investment=initial_investment,
            transaction_costs=transaction_costs
        )
        
        # Add parameter details to result
        result["parameters"].update({
            "buffer_lower": buffer_lower,
            "buffer_upper": buffer_upper,
            "cutoff_lower": cutoff_lower,
            "cutoff_upper": cutoff_upper
        })
        
        results.append(result)
    
    # Process results
    processed_df = processor.process_backtest_results(results)
    
    # Rank results
    ranking_metrics = list(config["ranking_metrics"].keys())
    ranking_weights = config["ranking_metrics"]
    ranked_df = processor.rank_results(processed_df, metrics=ranking_metrics, weights=ranking_weights)
    
    logger.info(f"Completed {len(results)} backtest scenarios")
    
    return results, processed_df, ranked_df 