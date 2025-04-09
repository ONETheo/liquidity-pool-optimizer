"""
Backtesting Engine Module

This module implements the core backtesting engine for simulating
USD* pool performance with different parameter combinations.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

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

class BacktestEngine:
    """
    Core backtesting engine for simulating USD* pool performance
    with different width, buffer, and cutoff parameter combinations.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        pool_data: Dict[str, Any],
        gas_settings: Dict[str, float]
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            price_data (pd.DataFrame): Historical price data with timestamp and price columns
            pool_data (Dict[str, Any]): Dictionary with pool configuration and performance data
            gas_settings (Dict[str, float]): Gas price and cost settings
        """
        self.price_data = price_data
        self.pool_data = pool_data
        self.gas_settings = gas_settings
        
        # Initial validation
        if 'timestamp' not in price_data.columns:
            raise ValueError("Price data must contain a 'timestamp' column")
        
        if 'price' not in price_data.columns:
            # Try to derive price if possible
            if all(col in price_data.columns for col in ['token1_price', 'token2_price']):
                price_data['price'] = price_data['token1_price'] / price_data['token2_price']
            else:
                raise ValueError("Price data must contain a 'price' column or token price columns")
        
        # Ensure data is sorted by timestamp
        self.price_data = self.price_data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Initialized BacktestEngine with {len(price_data)} price data points")
    
    def run_backtest(
        self,
        width: float,
        buffer_lower: float,
        buffer_upper: float,
        cutoff_lower: float,
        cutoff_upper: float,
        initial_investment: float = 10000.0,
        transaction_costs: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest simulation with the specified parameters.
        
        Args:
            width (float): Position width parameter (percentage)
            buffer_lower (float): Lower buffer margin (percentage)
            buffer_upper (float): Upper buffer margin (percentage)
            cutoff_lower (float): Lower cutoff value (price ratio)
            cutoff_upper (float): Upper cutoff value (price ratio)
            initial_investment (float): Initial investment amount in USD
            transaction_costs (bool): Whether to include transaction costs
            
        Returns:
            Dict[str, Any]: Dictionary with backtest results
        """
        logger.info(f"Running backtest with width={width}, buffer_lower={buffer_lower}, "
                   f"buffer_upper={buffer_upper}, cutoff_lower={cutoff_lower}, "
                   f"cutoff_upper={cutoff_upper}")
        
        # Initialize simulation state
        results = self._initialize_simulation(width, initial_investment)
        
        # Run the simulation through each price point
        for i, row in self.price_data.iterrows():
            timestamp = row['timestamp']
            price = row['price']
            
            # Current price boundaries based on width
            results['current_position']['lower_price'] = results['current_position']['base_price'] * (1 - width/100)
            results['current_position']['upper_price'] = results['current_position']['base_price'] * (1 + width/100)
            
            # Buffer zone boundaries
            lower_buffer_price = results['current_position']['lower_price'] * (1 - buffer_lower/100)
            upper_buffer_price = results['current_position']['upper_price'] * (1 + buffer_upper/100)
            
            # Check if price is outside buffer zone
            price_below_buffer = price < lower_buffer_price
            price_above_buffer = price > upper_buffer_price
            
            # Check if price is outside cutoff values
            price_below_cutoff = price < cutoff_lower
            price_above_cutoff = price > cutoff_upper
            
            # Track position state
            results['price_history'].append({
                'timestamp': timestamp,
                'price': price,
                'lower_price': results['current_position']['lower_price'],
                'upper_price': results['current_position']['upper_price'],
                'lower_buffer_price': lower_buffer_price,
                'upper_buffer_price': upper_buffer_price,
                'in_position': lower_buffer_price <= price <= upper_buffer_price,
                'outside_buffer': price_below_buffer or price_above_buffer,
                'outside_cutoff': price_below_cutoff or price_above_cutoff
            })
            
            # Check if rebalancing is needed
            rebalance_needed = (price_below_buffer or price_above_buffer) and not (price_below_cutoff or price_above_cutoff)
            
            if rebalance_needed:
                # Perform rebalancing
                rebalance_result = self._perform_rebalance(
                    timestamp=timestamp,
                    current_price=price,
                    current_position=results['current_position'],
                    width=width,
                    transaction_costs=transaction_costs
                )
                
                results['rebalance_events'].append(rebalance_result)
                results['current_position'] = rebalance_result['new_position']
            
            # Calculate fees earned for this period
            period_fees = self._calculate_period_fees(
                timestamp=timestamp,
                current_price=price,
                current_position=results['current_position'],
                in_position=lower_buffer_price <= price <= upper_buffer_price
            )
            
            results['fee_events'].append(period_fees)
            results['total_fees_earned'] += period_fees['fees_earned']
        
        # Calculate final metrics
        self._calculate_final_metrics(results, transaction_costs)
        
        return results
    
    def _initialize_simulation(
        self,
        width: float,
        initial_investment: float
    ) -> Dict[str, Any]:
        """Initialize the simulation state."""
        # Get initial price
        initial_price = self.price_data.iloc[0]['price']
        
        # Calculate position boundaries
        lower_price = initial_price * (1 - width/100)
        upper_price = initial_price * (1 + width/100)
        
        # Initialize result structure
        results = {
            'parameters': {
                'width': width,
                'initial_investment': initial_investment,
                'start_time': self.price_data.iloc[0]['timestamp'],
                'end_time': self.price_data.iloc[-1]['timestamp'],
                'duration_days': (self.price_data.iloc[-1]['timestamp'] - 
                                 self.price_data.iloc[0]['timestamp']).total_seconds() / 86400
            },
            'current_position': {
                'base_price': initial_price,
                'lower_price': lower_price,
                'upper_price': upper_price,
                'investment_value': initial_investment,
                'token1_amount': initial_investment / 2,
                'token2_amount': (initial_investment / 2) / initial_price
            },
            'price_history': [],
            'rebalance_events': [],
            'fee_events': [],
            'total_fees_earned': 0,
            'metrics': {}
        }
        
        return results
    
    def _perform_rebalance(
        self,
        timestamp: datetime,
        current_price: float,
        current_position: Dict[str, float],
        width: float,
        transaction_costs: bool
    ) -> Dict[str, Any]:
        """
        Perform a rebalancing operation.
        
        Args:
            timestamp (datetime): Current timestamp
            current_price (float): Current price
            current_position (Dict[str, float]): Current position state
            width (float): Position width parameter
            transaction_costs (bool): Whether to include transaction costs
            
        Returns:
            Dict[str, Any]: Rebalance event details
        """
        # Calculate new position centered around current price
        lower_price = current_price * (1 - width/100)
        upper_price = current_price * (1 + width/100)
        
        # Calculate position value before rebalance
        position_value = (
            current_position['token1_amount'] + 
            current_position['token2_amount'] * current_price
        )
        
        # Calculate gas costs
        gas_cost = 0
        if transaction_costs:
            gas_price_gwei = self.gas_settings.get('avg_gas_price', 20)
            gas_used = self.gas_settings.get('rebalance_gas_cost', 150000)
            gas_cost_eth = (gas_price_gwei * 1e-9) * gas_used
            
            # Convert ETH cost to USD (using a placeholder price of $3000)
            eth_price_usd = 3000
            gas_cost = gas_cost_eth * eth_price_usd
        
        # Calculate new token amounts after rebalance
        new_token1_amount = (position_value - gas_cost) / 2
        new_token2_amount = (position_value - gas_cost - new_token1_amount) / current_price
        
        # New position after rebalance
        new_position = {
            'base_price': current_price,
            'lower_price': lower_price,
            'upper_price': upper_price,
            'investment_value': position_value - gas_cost,
            'token1_amount': new_token1_amount,
            'token2_amount': new_token2_amount
        }
        
        # Calculate impermanent loss from this rebalance
        # This is a simplified calculation for demonstration
        price_ratio = current_price / current_position['base_price']
        k = np.sqrt(price_ratio)
        hodl_value = (
            current_position['token1_amount'] + 
            current_position['token2_amount'] * current_price
        )
        no_il_value = current_position['investment_value'] * (1 + price_ratio) / 2
        impermanent_loss = (hodl_value - no_il_value) / no_il_value
        
        return {
            'timestamp': timestamp,
            'old_position': current_position.copy(),
            'new_position': new_position,
            'price': current_price,
            'gas_cost': gas_cost,
            'value_before': position_value,
            'value_after': position_value - gas_cost,
            'impermanent_loss': impermanent_loss
        }
    
    def _calculate_period_fees(
        self,
        timestamp: datetime,
        current_price: float,
        current_position: Dict[str, float],
        in_position: bool
    ) -> Dict[str, Any]:
        """
        Calculate fees earned for a single period.
        
        Args:
            timestamp (datetime): Current timestamp
            current_price (float): Current price
            current_position (Dict[str, float]): Current position state
            in_position (bool): Whether the current price is within position boundaries
            
        Returns:
            Dict[str, Any]: Fee event details
        """
        # Get volume and fee rate data for this timestamp
        # In a real implementation, we'd interpolate from actual historical data
        
        # For volume, check if we have historical data
        volume = 0
        if 'volume_history' in self.pool_data:
            volume_df = self.pool_data['volume_history']
            # Find closest volume data point
            if not volume_df.empty:
                closest_idx = (volume_df['timestamp'] - timestamp).abs().idxmin()
                volume = volume_df.iloc[closest_idx]['volume']
        
        # For fee rate, check if we have historical data
        fee_rate = 0.003  # Default fee rate (0.3%)
        if 'fee_history' in self.pool_data:
            fee_df = self.pool_data['fee_history']
            # Find closest fee data point
            if not fee_df.empty:
                closest_idx = (fee_df['timestamp'] - timestamp).abs().idxmin()
                fee_rate = fee_df.iloc[closest_idx]['fee_rate']
        
        # Calculate fees based on liquidity share and whether in position
        position_value = (
            current_position['token1_amount'] + 
            current_position['token2_amount'] * current_price
        )
        
        # Get total TVL data
        tvl = 5000000  # Default TVL
        if 'tvl_history' in self.pool_data:
            tvl_df = self.pool_data['tvl_history']
            # Find closest TVL data point
            if not tvl_df.empty:
                closest_idx = (tvl_df['timestamp'] - timestamp).abs().idxmin()
                tvl = tvl_df.iloc[closest_idx]['tvl']
        
        # Calculate liquidity share
        liquidity_share = position_value / tvl if tvl > 0 else 0
        
        # Calculate fees earned (only if price is within position)
        fees_earned = 0
        if in_position:
            fees_earned = volume * fee_rate * liquidity_share
        
        return {
            'timestamp': timestamp,
            'price': current_price,
            'in_position': in_position,
            'volume': volume,
            'fee_rate': fee_rate,
            'liquidity_share': liquidity_share,
            'fees_earned': fees_earned
        }
    
    def _calculate_final_metrics(
        self,
        results: Dict[str, Any],
        transaction_costs: bool
    ) -> None:
        """Calculate final performance metrics for the backtest."""
        # Calculate total rebalance count
        rebalance_count = len(results['rebalance_events'])
        
        # Calculate total gas costs
        total_gas_costs = sum(event['gas_cost'] for event in results['rebalance_events'])
        
        # Calculate time in position percentage
        time_in_position = sum(1 for event in results['price_history'] if event['in_position'])
        time_in_position_pct = time_in_position / len(results['price_history']) * 100
        
        # Calculate final position value
        final_price = results['price_history'][-1]['price'] if results['price_history'] else 0
        final_position_value = (
            results['current_position']['token1_amount'] + 
            results['current_position']['token2_amount'] * final_price
        )
        
        # Calculate returns
        initial_investment = results['parameters']['initial_investment']
        absolute_return = final_position_value + results['total_fees_earned'] - initial_investment
        percentage_return = absolute_return / initial_investment * 100
        
        # Calculate APR
        duration_days = results['parameters']['duration_days']
        apr = percentage_return * (365 / duration_days) if duration_days > 0 else 0
        
        # Calculate rebalance frequency
        if duration_days > 0 and rebalance_count > 0:
            rebalance_frequency = rebalance_count / duration_days
        else:
            rebalance_frequency = 0
        
        # Calculate fees APR
        fees_apr = (results['total_fees_earned'] / initial_investment) * (365 / duration_days) * 100 if duration_days > 0 else 0
        
        # Calculate price volatility
        if len(results['price_history']) > 1:
            prices = [event['price'] for event in results['price_history']]
            price_changes = np.diff(prices) / prices[:-1]
            volatility = np.std(price_changes) * np.sqrt(len(price_changes) / duration_days * 365) * 100
        else:
            volatility = 0
        
        # Store the metrics
        results['metrics'] = {
            'rebalance_count': rebalance_count,
            'rebalance_frequency': rebalance_frequency,
            'total_costs': total_gas_costs if transaction_costs else 0,
            'time_in_position_pct': time_in_position_pct,
            'final_position_value': final_position_value,
            'total_fees_earned': results['total_fees_earned'],
            'absolute_return': absolute_return,
            'percentage_return': percentage_return,
            'apr': apr,
            'fees_apr': fees_apr,
            'volatility': volatility
        }
        
        logger.info(f"Backtest completed: APR={apr:.2f}%, Rebalances={rebalance_count}, "
                   f"Time in position={time_in_position_pct:.2f}%")

if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample price data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample price data with some randomness and a trend
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(hours=i) for i in range(24*30)]
    
    # Generate price series with randomness and a slight trend
    initial_price = 1.0
    price_changes = np.random.normal(0, 0.002, len(dates))
    price_changes[0] = 0
    price_series = initial_price * (1 + np.cumsum(price_changes))
    
    # Create DataFrame
    price_data = pd.DataFrame({
        'timestamp': dates,
        'price': price_series
    })
    
    # Sample pool data
    pool_data = {
        'volume_history': pd.DataFrame({
            'timestamp': dates,
            'volume': np.random.uniform(50000, 150000, len(dates))
        }),
        'fee_history': pd.DataFrame({
            'timestamp': dates,
            'fee_rate': [0.003] * len(dates)
        }),
        'tvl_history': pd.DataFrame({
            'timestamp': dates,
            'tvl': np.random.uniform(4500000, 5500000, len(dates))
        })
    }
    
    # Sample gas settings
    gas_settings = {
        'avg_gas_price': 20,
        'rebalance_gas_cost': 150000
    }
    
    # Create backtest engine
    engine = BacktestEngine(price_data, pool_data, gas_settings)
    
    # Run a backtest
    results = engine.run_backtest(
        width=0.05,
        buffer_lower=0.1,
        buffer_upper=0.1,
        cutoff_lower=0.95,
        cutoff_upper=1.05,
        initial_investment=10000.0
    )
    
    # Print results
    print("\nBacktest Results:")
    print(f"APR: {results['metrics']['apr']:.2f}%")
    print(f"Rebalance Count: {results['metrics']['rebalance_count']}")
    print(f"Rebalance Frequency: {results['metrics']['rebalance_frequency']:.2f} per day")
    print(f"Time in Position: {results['metrics']['time_in_position_pct']:.2f}%")
    print(f"Total Fees Earned: ${results['metrics']['total_fees_earned']:.2f}")
    print(f"Final Position Value: ${results['metrics']['final_position_value']:.2f}")
    print(f"Total Gas Costs: ${results['metrics']['total_costs']:.2f}") 