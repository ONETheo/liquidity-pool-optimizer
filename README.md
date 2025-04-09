# Liquidity Pool Optimizer

A comprehensive backtesting and parameter optimization tool for USD* liquidity pools.

## Overview

This tool systematically tests and optimizes liquidity provision strategies for USD* pools, with a focus on:

- Determining optimal position width, buffer, and cutoff parameters
- Calculating expected APR and ROI based on historical data
- Simulating rebalancing costs and fee earnings
- Comparing performance against HODL strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/liquidity-pool-optimizer.git
cd liquidity-pool-optimizer

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── lp-optimizer/          # Core optimization framework
│   ├── src/
│   │   ├── simulation/    # Core simulation engine
│   │   ├── utils/         # Utility functions
│   │   └── data/          # Data handling
│   ├── data/              # Historical and test data
│   ├── cache/             # Data cache directory
│   └── output/            # Output files and results
```

## Quick Start

### Run a Basic Simulation

```bash
python lp-optimizer/src/simulation/vfat_simulator.py \
    --pair "USDC.e/scUSD" \
    --amount 1000 \
    --days 30 \
    --width 1.0 \
    --buffer 0.2 \
    --cutoff 2.0 \
    --plot
```

### Optimize Parameters

```bash
python lp-optimizer/src/simulation/parameter_optimizer.py \
    --pair "USDC.e/scUSD" \
    --amount 1000 \
    --days 30 \
    --width-min 0.5 \
    --width-max 2.0 \
    --width-step 0.5 \
    --buffer-min 0.1 \
    --buffer-max 0.5 \
    --buffer-step 0.1 \
    --cutoff-min 1.0 \
    --cutoff-max 3.0 \
    --cutoff-step 1.0 \
    --strategy grid_search \
    --metric final_position_value \
    --jobs 11
```

## Key Parameters Explained

### Simulation Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `width` | Position width in percent (±width/2 from current price) | 0.5-5.0% |
| `buffer` | Buffer zone before rebalancing (hysteresis) | 0.1-0.5% |
| `cutoff` | Maximum price deviation before forced rebalancing | 1.0-5.0% |
| `max_dust_percent` | Maximum percentage of dust allowed during rebalancing | 0.5-1.5% |

### Optimization Parameters

| Parameter | Description |
|-----------|-------------|
| `strategy` | Optimization strategy: `grid_search`, `random_search`, or `latin_hypercube` |
| `metric` | Optimization target: `position_apr`, `position_roi`, `performance_vs_hodl`, `score`, `final_position_value` |
| `jobs` | Number of parallel processes for optimization |

## Interpreting Results

### Simulation Output

The simulator outputs key performance metrics:

- **Position APR (annualized)**: Annualized return rate
- **ROI**: Return on investment over the simulation period
- **Fee APR**: Annual percentage rate from earned fees
- **vs HODL**: Performance comparison against buy-and-hold strategy
- **Time In Range**: Percentage of time the position stays within range
- **Rebalance Count**: Number of rebalancing events
- **Rebalance Costs**: Total costs associated with rebalancing

### Parameter Optimization

Optimization results are presented in ranked order with:

1. Parameter combinations (width, buffer, cutoff)
2. Performance metrics (APR, ROI, vs HODL)
3. Final position value
4. Visualization of parameter impact on performance

## Advanced Usage

### Custom Fee Tiers

To simulate with different fee tiers, modify the `FEE_TIERS` dictionary in `vfat_simulator.py`:

```python
FEE_TIERS = {
    "USDC.e/scUSD": 0.0001,  # 0.01%
    "YOUR_PAIR": 0.0025,     # 0.25%
}
```

### External Data Sources

To use external data sources, implement a custom data loader in the `utils` directory and modify the `fetch_pool_historical_data` method in `vfat_simulator.py`.

## Outputs and Visualization

The tool generates:

1. Detailed CSV files with simulation results
2. Performance plots showing:
   - Price ratio over time
   - Position value vs HODL value
   - Daily and cumulative fees
   - Rebalancing events
3. Parameter optimization heatmaps

## Best Practices

- **Backtesting Period**: Use at least 30 days of data for meaningful results
- **Parameter Ranges**: Start with wide ranges and narrow down in subsequent optimizations
- **Validation**: Compare results against actual pool performance when possible
- **Market Conditions**: Be aware that optimal parameters may vary with market volatility

## License

This project is licensed under the MIT License - see the LICENSE file for details. 