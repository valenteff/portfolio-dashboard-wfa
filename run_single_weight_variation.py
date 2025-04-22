#!/usr/bin/env python3
"""
Single Weight Variation Runner for Portfolio Backtest

This script runs the portfolio_backtest_weight.py script with a single specified
weight combination for profit, win rate, and recovery factor.

Usage:
    python run_single_weight_variation.py <profit_weight> <win_rate_weight> <recovery_weight> [name]

Example:
    python run_single_weight_variation.py 0.5 0.25 0.25 balancedProfit
"""

import os
import sys
import datetime
import subprocess
import time

# Define the input directory containing the WFA result files
INPUT_DIRECTORY = "/Users/fabio/Backtest_Portfolio_WFA_Reformulado/output_20250315_101842"

def create_modified_script(weights):
    """
    Create a temporary modified version of portfolio_backtest_weight.py with the specified weights.

    Args:
        weights: Dictionary containing the weights for profit, win_rate, and recovery

    Returns:
        Path to the temporary script file
    """
    # Read the original script
    with open("portfolio_backtest_weight.py", "r") as f:
        content = f.read()

    # Replace the weights in the calculateFinalRank method
    old_weights_code = """
            coin['final_rank'] = (
                0.4 * profit_ranks[symbol] +
                0.3 * wins_ranks[symbol] +
                0.3 * recovery_ranks[symbol]
            )"""

    new_weights_code = f"""
            coin['final_rank'] = (
                {weights['profit']} * profit_ranks[symbol] +
                {weights['win_rate']} * wins_ranks[symbol] +
                {weights['recovery']} * recovery_ranks[symbol]
            )"""

    modified_content = content.replace(old_weights_code, new_weights_code)

    # Create a temporary file with the modified content
    temp_script_path = f"temp_portfolio_backtest_weight_{weights['name']}.py"
    with open(temp_script_path, "w") as f:
        f.write(modified_content)

    return temp_script_path

def run_backtest(weights):
    """
    Run the portfolio backtest with the specified weights.

    Args:
        weights: Dictionary containing the weights for profit, win_rate, and recovery
    """
    # Create a modified script with the specified weights
    temp_script_path = create_modified_script(weights)

    # Create a timestamped directory name with weight information
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_str = f"p{int(weights['profit']*100)}_w{int(weights['win_rate']*100)}_r{int(weights['recovery']*100)}"
    results_dir = f"results_weight_{weights['name']}_{weights_str}_{timestamp}"

    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)

    # Run the modified script
    print(f"\n{'='*80}")
    print(f"Running backtest with weights: Profit={weights['profit']}, Win Rate={weights['win_rate']}, Recovery={weights['recovery']}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*80}\n")

    # Prepare the Python code to execute
    python_code = f"""
import os
import datetime
import sys

# Import the modified backtest class
from {os.path.splitext(os.path.basename(temp_script_path))[0]} import PortfolioBacktestWeight

# Initialize with correct parameters
backtest = PortfolioBacktestWeight(
    input_dir="{INPUT_DIRECTORY}",
    normalized_dir="{results_dir}"
)

# Run the backtest
backtest.run_backtest(num_top_portfolios=10)
"""

    # Execute the Python code
    process = subprocess.Popen([sys.executable, "-c", python_code],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)

    # Print output in real-time
    for line in process.stdout:
        print(line, end='')

    # Wait for the process to complete
    process.wait()

    # Check for errors
    if process.returncode != 0:
        print(f"Error running backtest with weights: {weights}")
        for line in process.stderr:
            print(line, end='')

    # Clean up the temporary script
    os.remove(temp_script_path)

    return results_dir

def main():
    """Run a single weight variation."""
    if len(sys.argv) < 4:
        print("Usage: python run_single_weight_variation.py <profit_weight> <win_rate_weight> <recovery_weight> [name]")
        print("Example: python run_single_weight_variation.py 0.5 0.25 0.25 balancedProfit")
        sys.exit(1)

    # Parse command line arguments
    profit_weight = float(sys.argv[1])
    win_rate_weight = float(sys.argv[2])
    recovery_weight = float(sys.argv[3])

    # Check if weights sum to 1.0
    total_weight = profit_weight + win_rate_weight + recovery_weight
    if abs(total_weight - 1.0) > 0.001:
        print(f"Warning: Weights do not sum to 1.0 (sum = {total_weight})")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Get optional name parameter
    name = sys.argv[4] if len(sys.argv) > 4 else f"custom"

    weights = {
        "profit": profit_weight,
        "win_rate": win_rate_weight,
        "recovery": recovery_weight,
        "name": name
    }

    start_time = time.time()
    results_dir = run_backtest(weights)
    end_time = time.time()

    # Print summary
    print("\n\n" + "="*100)
    print("WEIGHT VARIATION TEST SUMMARY")
    print("="*100)
    print(f"Weight Combination: Profit={profit_weight:.2f}, Win Rate={win_rate_weight:.2f}, Recovery={recovery_weight:.2f}")
    print(f"Results Directory: {results_dir}")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    print("="*100)
    print("Weight variation test completed successfully!")

if __name__ == "__main__":
    main()
