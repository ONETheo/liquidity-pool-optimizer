"""
Main Module - USD* Rewards Simulator

This module demonstrates the usage of the backtesting engine,
parameter optimizer, and results processor.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import project modules
from src.simulation import BacktestEngine, ResultProcessor, run_backtest_suite
from src.simulation.parameter_optimizer import ParameterOptimizer, find_optimal_parameters
from src.data import load_test_data
import src.simulation.config as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_single_backtest(args):
    """Run a single backtest with specified parameters."""
    logger.info("Running single backtest")
    
    # Load test data
    price_data, pool_data = load_test_data(
        start_date=args.start_date,
        end_date=args.end_date,
        token_pair=args.token_pair
    )
    
    # Get gas settings from config
    config = cfg.get_config()
    gas_settings = config["gas_settings"]
    
    # Create backtest engine
    engine = BacktestEngine(price_data, pool_data, gas_settings)
    
    # Run backtest with specified parameters
    result = engine.run_backtest(
        width=args.width,
        buffer_lower=args.buffer_lower,
        buffer_upper=args.buffer_upper,
        cutoff_lower=args.cutoff_lower,
        cutoff_upper=args.cutoff_upper,
        initial_investment=args.investment,
        transaction_costs=not args.no_costs
    )
    
    # Initialize result processor
    processor = ResultProcessor(config["results_dir"])
    
    # Create visualizations
    processor.plot_rebalance_events(
        [result], 0, output_file=f"backtest_rebalance_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    # Print results
    metrics = result["metrics"]
    print("\nBacktest Results:")
    print(f"APR: {metrics['apr']:.2f}%")
    print(f"Net APR (after costs): {metrics['apr'] - (metrics['total_costs']/args.investment)*(365/result['parameters']['duration_days'])*100:.2f}%")
    print(f"Fees APR: {metrics['fees_apr']:.2f}%")
    print(f"Rebalance Count: {metrics['rebalance_count']}")
    print(f"Rebalance Frequency: {metrics['rebalance_frequency']:.2f} per day")
    print(f"Time in Position: {metrics['time_in_position_pct']:.2f}%")
    print(f"Total Fees Earned: ${metrics['total_fees_earned']:.2f}")
    print(f"Total Gas Costs: ${metrics['total_costs']:.2f}")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    
    return result


def run_parameter_comparison(args):
    """Run backtests with multiple parameter combinations."""
    logger.info("Running parameter comparison")
    
    # Load test data
    price_data, pool_data = load_test_data(
        start_date=args.start_date,
        end_date=args.end_date,
        token_pair=args.token_pair
    )
    
    # Define parameter combinations to test
    param_combinations = [
        {"width": 0.5, "buffer_lower": 0.1, "buffer_upper": 0.1, "cutoff_lower": 0.95, "cutoff_upper": 1.05},
        {"width": 0.3, "buffer_lower": 0.05, "buffer_upper": 0.05, "cutoff_lower": 0.97, "cutoff_upper": 1.03},
        {"width": 0.8, "buffer_lower": 0.2, "buffer_upper": 0.2, "cutoff_lower": 0.9, "cutoff_upper": 1.1},
        {"width": 0.2, "buffer_lower": 0.02, "buffer_upper": 0.02, "cutoff_lower": 0.98, "cutoff_upper": 1.02},
        {"width": 1.0, "buffer_lower": 0.15, "buffer_upper": 0.15, "cutoff_lower": 0.92, "cutoff_upper": 1.08}
    ]
    
    # Run backtests
    results, processed_df, ranked_df = run_backtest_suite(
        price_data, 
        pool_data, 
        param_combinations,
        initial_investment=args.investment,
        transaction_costs=not args.no_costs
    )
    
    # Get results directory from config
    config = cfg.get_config()
    results_dir = config["results_dir"]
    
    # Initialize result processor
    processor = ResultProcessor(results_dir)
    
    # Generate visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Parameter comparison plots
    processor.plot_parameter_comparison(
        ranked_df, 'width', 'net_apr', 
        output_file=f"width_vs_apr_{timestamp}.png"
    )
    
    processor.plot_parameter_comparison(
        ranked_df, 'buffer_avg', 'time_in_position_pct', 
        output_file=f"buffer_vs_time_in_position_{timestamp}.png"
    )
    
    # Plot top result
    if results:
        processor.plot_rebalance_events(
            results, 0, 
            output_file=f"top_result_rebalance_events_{timestamp}.png"
        )
    
    # Generate summary report
    report = processor.generate_summary_report(
        ranked_df, top_n=len(param_combinations), 
        output_file=f"parameter_comparison_report_{timestamp}.txt"
    )
    
    # Print summary
    print("\nParameter Comparison Results:")
    print(f"Generated visualizations in: {results_dir}")
    print(f"Summary report saved to: {os.path.join(results_dir, f'parameter_comparison_report_{timestamp}.txt')}")
    
    return results, ranked_df


def run_parameter_optimization(args):
    """Run parameter optimization to find the best combination."""
    logger.info("Running parameter optimization")
    
    # Load test data
    price_data, pool_data = load_test_data(
        start_date=args.start_date,
        end_date=args.end_date,
        token_pair=args.token_pair
    )
    
    # Get gas settings from config
    config = cfg.get_config()
    gas_settings = config["gas_settings"]
    
    # Create backtest engine
    engine = BacktestEngine(price_data, pool_data, gas_settings)
    
    # Customize parameter ranges if needed
    param_ranges = {
        "width": {
            "min": 0.1,
            "max": 1.0,
            "step": 0.1
        },
        "buffer_lower": {
            "min": 0.05,
            "max": 0.2,
            "step": 0.05
        },
        "buffer_upper": {
            "min": 0.05,
            "max": 0.2,
            "step": 0.05
        },
        "cutoff_lower": {
            "min": 0.9,
            "max": 0.99,
            "step": 0.01
        },
        "cutoff_upper": {
            "min": 1.01,
            "max": 1.1,
            "step": 0.01
        }
    }
    
    # Create parameter optimizer
    optimizer = ParameterOptimizer(param_ranges, max_combinations=args.max_samples)
    
    # Run optimization
    results, ranked_df, best_params = optimizer.optimize(
        engine, 
        price_data, 
        pool_data, 
        strategy=args.strategy,
        sample_size=args.samples,
        n_jobs=args.jobs
    )
    
    # Get results directory from config
    results_dir = config["results_dir"]
    
    # Initialize result processor
    processor = ResultProcessor(results_dir)
    
    # Generate visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Parameter heatmaps
    processor.plot_heatmap(
        ranked_df, 'width', 'buffer_avg', 'net_apr', 
        output_file=f"width_buffer_heatmap_{timestamp}.png"
    )
    
    processor.plot_heatmap(
        ranked_df, 'width', 'cutoff_range', 'net_apr', 
        output_file=f"width_cutoff_heatmap_{timestamp}.png"
    )
    
    # Plot top result
    if results:
        processor.plot_rebalance_events(
            results, 0, 
            output_file=f"optimal_rebalance_events_{timestamp}.png"
        )
    
    # Generate summary report
    report = processor.generate_summary_report(
        ranked_df, top_n=10, 
        output_file=f"optimization_report_{timestamp}.txt"
    )
    
    # Print best parameters
    print("\nParameter Optimization Results:")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    if not ranked_df.empty:
        best_result = ranked_df.iloc[0]
        print("\nPerformance with best parameters:")
        print(f"  Net APR: {best_result.get('net_apr', 0):.2f}%")
        print(f"  Fees APR: {best_result.get('fees_apr', 0):.2f}%")
        print(f"  Rebalance Count: {int(best_result.get('rebalance_count', 0))}")
        print(f"  Time in Position: {best_result.get('time_in_position_pct', 0):.2f}%")
        print(f"  Score: {best_result.get('score', 0):.4f}")
    
    print(f"\nGenerated visualizations in: {results_dir}")
    print(f"Full report saved to: {os.path.join(results_dir, f'optimization_report_{timestamp}.txt')}")
    
    return results, ranked_df, best_params


def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description="USD* Rewards Simulator")
    
    # Common parameters
    parser.add_argument("--start-date", type=str, 
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, 
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--token-pair", type=str, default="USD*/USDC",
                        help="Token pair (default: USD*/USDC)")
    parser.add_argument("--investment", type=float, default=10000.0,
                        help="Initial investment amount (default: 10000.0)")
    parser.add_argument("--no-costs", action="store_true",
                        help="Disable transaction costs")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a single backtest")
    backtest_parser.add_argument("--width", type=float, default=0.5,
                                help="Position width in percentage (default: 0.5)")
    backtest_parser.add_argument("--buffer-lower", type=float, default=0.1,
                                help="Lower buffer in percentage (default: 0.1)")
    backtest_parser.add_argument("--buffer-upper", type=float, default=0.1,
                                help="Upper buffer in percentage (default: 0.1)")
    backtest_parser.add_argument("--cutoff-lower", type=float, default=0.95,
                                help="Lower cutoff value (default: 0.95)")
    backtest_parser.add_argument("--cutoff-upper", type=float, default=1.05,
                                help="Upper cutoff value (default: 1.05)")
    
    # Parameter comparison command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple parameter sets")
    
    # Parameter optimization command
    optimize_parser = subparsers.add_parser("optimize", help="Find optimal parameters")
    optimize_parser.add_argument("--strategy", type=str, default="latin",
                                choices=["grid", "random", "latin"],
                                help="Optimization strategy (default: latin)")
    optimize_parser.add_argument("--samples", type=int, default=100,
                                help="Number of parameter combinations to test (default: 100)")
    optimize_parser.add_argument("--max-samples", type=int, default=1000,
                                help="Maximum number of parameter combinations (default: 1000)")
    optimize_parser.add_argument("--jobs", type=int, default=1,
                                help="Number of parallel jobs (default: 1)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default dates if not provided
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Run the appropriate command
    if args.command == "backtest":
        run_single_backtest(args)
    elif args.command == "compare":
        run_parameter_comparison(args)
    elif args.command == "optimize":
        run_parameter_optimization(args)
    else:
        # Default to parameter comparison if no command specified
        run_parameter_comparison(args)


if __name__ == "__main__":
    main() 