"""
Update Sample Data for Deployed Dashboard

This script sets up sample data for the deployed dashboard by ensuring 
there's at least one result directory with meaningful data.

For cloud deployments, this script can be run during the build process.
"""

import os
import shutil
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_data():
    """Create sample data for the dashboard if no results directories exist."""
    # Check if there are already result directories
    result_dirs = [d for d in os.listdir() if d.startswith("results_")]
    
    if result_dirs:
        print("Found existing result directories, no need to create sample data.")
        return
    
    # Create a sample result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = f"results_{timestamp}"
    
    # Create directory structure
    dirs = [
        sample_dir,
        f"{sample_dir}/normalized_data",
        f"{sample_dir}/portfolios",
        f"{sample_dir}/equity_curves",
        f"{sample_dir}/plots",
        f"{sample_dir}/statistics",
        f"{sample_dir}/monthly_analysis",
        f"{sample_dir}/adjusted_monthly_analysis"
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Create sample portfolio summary
    portfolios = ["Top1", "Top2", "Top3", "Top5", "Top10"]
    summary_data = []
    
    for p in portfolios:
        summary_data.append({
            "Portfolio": p,
            "Total Return (%)": random.uniform(50, 150),
            "CAGR (%)": random.uniform(15, 40),
            "Max Drawdown (%)": random.uniform(-25, -5),
            "Sharpe Ratio": random.uniform(1.0, 3.0),
            "Sortino Ratio": random.uniform(1.5, 4.0),
            "Calmar Ratio": random.uniform(0.8, 2.5),
            "Win Rate (%)": random.uniform(40, 70),
            "Profit Factor": random.uniform(1.2, 2.5),
            "Recovery Factor": random.uniform(3, 8),
            "Volatility (%)": random.uniform(10, 25),
            "Number of Trades": random.randint(50, 200)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{sample_dir}/portfolios_comparative_summary.csv", index=False)
    
    # Create sample equity curve for each portfolio
    for portfolio in portfolios:
        # Create equity curve data
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Generate dates (every 3 days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='3D')
        
        # Generate equity curve with randomness and trend
        starting_equity = 10000
        trend = random.uniform(0.0001, 0.0005)  # Daily upward trend
        equity_values = [starting_equity]
        
        for i in range(1, len(date_range)):
            # Random daily change with upward bias
            daily_change = np.random.normal(trend, 0.01)
            new_equity = equity_values[-1] * (1 + daily_change)
            equity_values.append(new_equity)
        
        # Create DataFrame
        equity_df = pd.DataFrame({
            'datetime': date_range,
            'equity': equity_values,
            'volume': np.random.randint(1, 5, size=len(date_range))
        })
        
        # Save equity curve
        equity_df.to_csv(f"{sample_dir}/equity_curves/{portfolio}_equity_curve.csv", index=False)
        
        # Create monthly statistics
        months = pd.date_range(start=start_date, end=end_date, freq='MS')
        monthly_data = []
        
        for i, month_start in enumerate(months):
            year = month_start.year
            month = month_start.month
            month_name = month_start.strftime('%B')
            month_end = (month_start + pd.offsets.MonthEnd(0)).date()
            
            # Monthly return with some randomness
            monthly_return = np.random.normal(2.0, 4.0)  # Mean 2%, std 4%
            
            # Add outlier for testing
            if i == len(months) // 2:  # Middle month
                monthly_return = 15.0  # Outlier month
            
            monthly_data.append({
                'Portfolio': portfolio,
                'Year': year,
                'Month': month,
                'Month Name': month_name,
                'Year-Month': f"{year}-{month:02d}",
                'Start Date': month_start.date(),
                'End Date': month_end,
                'Start Equity': 10000 * (1 + 0.02) ** i,
                'End Equity': 10000 * (1 + 0.02) ** (i + 1),
                'Monthly Return (%)': monthly_return,
                'Max Drawdown (%)': random.uniform(-8, -1),
                'Volatility (%)': random.uniform(5, 15),
                'Number of Trades': random.randint(5, 20),
                'Win Rate (%)': random.uniform(40, 70)
            })
            
        monthly_df = pd.DataFrame(monthly_data)
        monthly_df.to_csv(f"{sample_dir}/monthly_analysis/{portfolio}_monthly_statistics.csv", index=False)
        
        # Create adjusted monthly statistics (replace outlier with trimmed mean)
        adjusted_df = monthly_df.copy()
        # Find the outlier month
        outlier_idx = adjusted_df['Monthly Return (%)'].idxmax()
        outlier_month = adjusted_df.loc[outlier_idx, 'Year-Month']
        
        # Adjust the outlier
        adjusted_df.loc[outlier_idx, 'Original Return (%)'] = adjusted_df.loc[outlier_idx, 'Monthly Return (%)']
        adjusted_df.loc[outlier_idx, 'Monthly Return (%)'] = 2.5  # Replaced with trimmed mean
        adjusted_df.loc[outlier_idx, 'Is Outlier'] = True
        adjusted_df.loc[outlier_idx, 'Z-Score'] = 3.2  # Mock z-score
        
        # Save adjusted monthly stats
        adjusted_df.to_csv(f"{sample_dir}/adjusted_monthly_analysis/{portfolio}_adjusted_monthly_statistics.csv", index=False)
        
        # Create outlier details
        outlier_details = [{
            'Year-Month': outlier_month,
            'Original Return (%)': monthly_df.loc[outlier_idx, 'Monthly Return (%)'],
            'Z-Score': 3.2,
            'Adjusted Return (%)': 2.5,
            'Adjustment': monthly_df.loc[outlier_idx, 'Monthly Return (%)'] - 2.5
        }]
        
        outlier_df = pd.DataFrame(outlier_details)
        outlier_df.to_csv(f"{sample_dir}/adjusted_monthly_analysis/{portfolio}_outlier_details.csv", index=False)
        
        # Create outlier summary
        summary_data = {
            'Portfolio': [portfolio],
            'Number of Months': [len(monthly_df)],
            'Number of Outliers': [1],
            'Outlier Percentage': [1 / len(monthly_df) * 100],
            'Original CAGR (%)': [28.5],
            'Adjusted CAGR (%)': [24.2],
            'CAGR Difference (%)': [4.3]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{sample_dir}/adjusted_monthly_analysis/{portfolio}_outlier_adjustment_summary.csv", index=False)
    
    print(f"Created sample data in directory: {sample_dir}")

if __name__ == '__main__':
    create_sample_data() 