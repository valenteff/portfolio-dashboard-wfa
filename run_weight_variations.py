#!/usr/bin/env python3
"""
Weight Variation Runner for Portfolio Backtest

This script runs the portfolio_backtest_weight.py script with different weight combinations
for profit, win rate, and recovery factor.

Each run creates a separate results directory with a name that clearly indicates the weights used.
"""

import os
import sys
import datetime
import subprocess
import time

# Define the input directory containing the WFA result files
INPUT_DIRECTORY = "/Users/fabio/Backtest_Portfolio_WFA_Reformulado/output_20250315_101842"

# Define the weight variations to test
WEIGHT_VARIATIONS = [
    # Current default weights
    {"profit": 0.4, "win_rate": 0.3, "recovery": 0.3, "name": "default"},

    # Profit-Focused Variations
    {"profit": 0.6, "win_rate": 0.2, "recovery": 0.2, "name": "highProfit"},
    {"profit": 0.5, "win_rate": 0.25, "recovery": 0.25, "name": "balancedProfit"},

    # Risk-Focused Variations
    {"profit": 0.3, "win_rate": 0.2, "recovery": 0.5, "name": "highRecovery"},
    {"profit": 0.3, "win_rate": 0.5, "recovery": 0.2, "name": "highWinRate"},

    # Balanced Risk-Return Variations
    {"profit": 0.33, "win_rate": 0.33, "recovery": 0.34, "name": "equal"},
    {"profit": 0.45, "win_rate": 0.1, "recovery": 0.45, "name": "profitRecovery"},

    # Market Condition-Specific Weights
    {"profit": 0.7, "win_rate": 0.15, "recovery": 0.15, "name": "bullMarket"},
    {"profit": 0.2, "win_rate": 0.3, "recovery": 0.5, "name": "bearMarket"},
    {"profit": 0.2, "win_rate": 0.6, "recovery": 0.2, "name": "sidewaysMarket"}
]

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
    """Run all weight variations."""
    print(f"Starting weight variation tests with {len(WEIGHT_VARIATIONS)} combinations")

    results = []
    for weights in WEIGHT_VARIATIONS:
        start_time = time.time()
        results_dir = run_backtest(weights)
        end_time = time.time()

        results.append({
            "weights": weights,
            "results_dir": results_dir,
            "runtime": end_time - start_time
        })

    # Print summary
    print("\n\n" + "="*100)
    print("WEIGHT VARIATION TESTS SUMMARY")
    print("="*100)
    print(f"{'Weight Combination':<30} {'Results Directory':<50} {'Runtime (s)':<15}")
    print("-"*100)

    for result in results:
        weights = result["weights"]
        weight_str = f"P:{weights['profit']:.2f}, W:{weights['win_rate']:.2f}, R:{weights['recovery']:.2f}"
        print(f"{weight_str:<30} {result['results_dir']:<50} {result['runtime']:.2f}")

    print("="*100)
    print("All weight variation tests completed successfully!")

if __name__ == "__main__":
    main()
