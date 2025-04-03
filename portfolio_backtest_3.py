import os
import pandas as pd
import glob
import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import skew, kurtosis, zscore
import seaborn as sns
import calendar

class PortfolioBacktest:
    def __init__(self, 
                 input_dir: str, 
                 performance_metric: str = 'Out-of-sample net profit (%)',
                 output_dir: str = None):
        """
        Initialize the portfolio backtest system.
        
        Args:
            input_dir: Directory containing the WFA result files
            performance_metric: Metric used for ranking symbols (default: 'Out-of-sample net profit (%)')
            output_dir: Directory for output files (default: results_YYYYMMDD_HHMMSS in current directory)
        """
        self.input_dir = input_dir
        self.performance_metric = performance_metric
        
        # Create a timestamped output directory if not specified
        if output_dir is None:
            # Get current timestamp for directory name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create output directory within the project directory
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{timestamp}")
        else:
            self.output_dir = output_dir
            
        self.wfa_files = {}
        self.equity_files = {}
        self.normalized_data = {}
        self.top_portfolios = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            
        # Create subdirectories for different types of outputs
        self.normalized_dir = os.path.join(self.output_dir, "normalized_data")
        self.portfolios_dir = os.path.join(self.output_dir, "portfolios")
        self.equity_curves_dir = os.path.join(self.output_dir, "equity_curves")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.stats_dir = os.path.join(self.output_dir, "statistics")
        self.monthly_dir = os.path.join(self.output_dir, "monthly_analysis")
        self.adjusted_monthly_dir = os.path.join(self.output_dir, "adjusted_monthly_analysis")
        
        for directory in [self.normalized_dir, self.portfolios_dir, self.equity_curves_dir, 
                         self.plots_dir, self.stats_dir, self.monthly_dir, self.adjusted_monthly_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created subdirectory: {directory}")
            
    def load_data(self):
        """Load all WFA and equity curve files from the input directory"""
        # Find all WFA files
        wfa_pattern = os.path.join(self.input_dir, "*parallel_walk_forward_analysis_*.csv")
        for file_path in glob.glob(wfa_pattern):
            symbol = os.path.basename(file_path).split('_')[0]
            self.wfa_files[symbol] = file_path
            
        # Find all equity curve files
        equity_pattern = os.path.join(self.input_dir, "*closed_trade_equity_curves_parallel_*.csv")
        for file_path in glob.glob(equity_pattern):
            symbol = os.path.basename(file_path).split('_')[0]
            self.equity_files[symbol] = file_path
            
        print(f"Loaded data for {len(self.wfa_files)} symbols")
        return len(self.wfa_files) > 0
    
    def normalize_periods(self):
        """Normalize periods across all symbols"""
        earliest_date = None
        symbol_dataframes = {}
        
        # Load all WFA files and find earliest out-of-sample start date
        for symbol, file_path in self.wfa_files.items():
            df = pd.read_csv(file_path)
            symbol_dataframes[symbol] = df
            
            # Extract start dates from the "Out-of-sample date range" column
            for _, row in df.iterrows():
                date_range = row['Out-of-sample date range']
                start_date_str = date_range.split(' to ')[0]
                start_date = pd.to_datetime(start_date_str)
                
                if earliest_date is None or start_date < earliest_date:
                    earliest_date = start_date
        
        print(f"Earliest out-of-sample start date: {earliest_date}")
        
        # Assign normalized periods to each symbol's periods
        for symbol, df in symbol_dataframes.items():
            df['normalized_period'] = 0  # Initialize
            
            for idx, row in df.iterrows():
                date_range = row['Out-of-sample date range']
                start_date_str = date_range.split(' to ')[0]
                start_date = pd.to_datetime(start_date_str)
                
                # Calculate time difference in days and divide by approximate period length
                # Assuming periods are roughly 2-3 days based on the example data
                days_diff = (start_date - earliest_date).days
                normalized_period = round(days_diff / 2) + 1  # +1 so periods start at 1
                
                df.at[idx, 'normalized_period'] = normalized_period
            
            # Save the normalized dataframe to the normalized_data subdirectory
            output_path = os.path.join(self.normalized_dir, f"{symbol}_normalized_wfa.csv")
            df.to_csv(output_path, index=False)
            self.normalized_data[symbol] = df
            
        return self.normalized_data
    
    def calculate_trade_metrics(self, equity_df):
        """
        Calculate trade metrics from equity curve data.
        
        Args:
            equity_df: DataFrame containing the equity curve data
            
        Returns:
            Dictionary containing trade metrics
        """
        if equity_df.empty:
            return {
                'winning_trades': 0,
                'total_trades': 0,
                'win_rate': 0,
                'net_profit': 0,
                'median_return': 0,
                'trade_returns': []
            }
        
        # Calculate trade-by-trade changes
        equity_df['equity_diff'] = equity_df['equity'].diff()
        equity_df['trade_direction'] = np.sign(equity_df['equity_diff'])
        equity_df['trade_changed'] = equity_df['trade_direction'].diff().fillna(0) != 0
        equity_df['trade_id'] = equity_df['trade_changed'].cumsum()
        
        # Calculate individual trade statistics
        trades = []
        trade_returns = []
        for trade_id, group in equity_df.groupby('trade_id'):
            if len(group) >= 2:  # Ensure we have at least start and end points
                trade_start = group['equity'].iloc[0]
                trade_end = group['equity'].iloc[-1]
                trade_return = (trade_end / trade_start - 1) * 100
                
                trades.append({
                    'id': trade_id,
                    'return': trade_return,
                    'profitable': trade_return > 0
                })
                trade_returns.append(trade_return)
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profitable']])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        net_profit = sum(t['return'] for t in trades)
        median_return = np.median(trade_returns) if trade_returns else 0
        
        return {
            'winning_trades': winning_trades,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'net_profit': net_profit,
            'median_return': median_return,
            'trade_returns': trade_returns
        }
    
    def create_top_portfolios(self, num_top_portfolios: int = 10):
        """
        Create top N portfolios based on median trade returns from the PREVIOUS period,
        requiring a minimum of 3 trades per period.
        """
        # Get all unique normalized periods
        all_periods = set()
        for df in self.normalized_data.values():
            all_periods.update(df['normalized_period'].unique())
        
        all_periods = sorted(all_periods)
        print(f"Total unique normalized periods: {len(all_periods)}")
        
        # For each period, store available symbols and their performance metrics
        period_symbol_data = {}
        for period in all_periods:
            period_symbol_data[period] = []
            
            for symbol, df in self.normalized_data.items():
                period_rows = df[df['normalized_period'] == period]
                
                if not period_rows.empty:
                    for _, row in period_rows.iterrows():
                        # Get the date range for this period
                        date_range = row['Out-of-sample date range']
                        start_date_str, end_date_str = date_range.split(' to ')
                        start_date = pd.to_datetime(start_date_str)
                        end_date = pd.to_datetime(end_date_str)
                        
                        # Load and filter equity curve data for this period
                        if symbol in self.equity_files:
                            equity_df = pd.read_csv(self.equity_files[symbol])
                            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
                            
                            # Filter equity data for this period
                            period_equity = equity_df[
                                (equity_df['datetime'] >= start_date) & 
                                (equity_df['datetime'] <= end_date)
                            ].copy()
                            
                            # Calculate trade metrics for this period
                            trade_metrics = self.calculate_trade_metrics(period_equity)
                            
                            # Only include periods with at least 3 trades
                            if trade_metrics['total_trades'] >= 3:
                                period_symbol_data[period].append({
                                    'symbol': symbol,
                                    'period_number': row['Period number'],
                                    'total_trades': trade_metrics['total_trades'],
                                    'median_return': trade_metrics['median_return'],
                                    'win_rate': trade_metrics['win_rate'],
                                    'net_profit': trade_metrics['net_profit'],
                                    'out_sample_range': date_range
                                })
        
        # Create top N portfolios
        for n in range(1, num_top_portfolios + 1):
            portfolio_data = []
            
            # Skip the first period as there's no previous period to base selection on
            for period in all_periods:
                if period <= 1:
                    continue
                
                # Get rankings from the PREVIOUS period
                prev_period = period - 1
                prev_period_data = period_symbol_data.get(prev_period, [])
                
                if not prev_period_data:
                    continue
                
                # Sort by median return (primary), total trades (secondary), and win rate (tertiary)
                prev_period_data.sort(key=lambda x: (
                    x['median_return'],
                    x['total_trades'],
                    x['win_rate']
                ), reverse=True)
                
                # Check if we have enough symbols for this top-N selection
                if len(prev_period_data) >= n:
                    # Select the nth best symbol from the previous period
                    selected_symbol = prev_period_data[n-1]['symbol']
                    
                    # Get this symbol's data for the CURRENT period
                    current_period_data = [
                        item for item in period_symbol_data.get(period, [])
                        if item['symbol'] == selected_symbol
                    ]
                    
                    # If the selected symbol has data for the current period, add it to the portfolio
                    if current_period_data:
                        # Choose the entry with the highest median return if multiple exist
                        current_period_data.sort(key=lambda x: (
                            x['median_return'],
                            x['total_trades'],
                            x['win_rate']
                        ), reverse=True)
                        selected_data = current_period_data[0]
                        
                        portfolio_data.append({
                            'normalized_period': period,
                            'symbol': selected_symbol,
                            'original_period': selected_data['period_number'],
                            'out_sample_range': selected_data['out_sample_range'],
                            'total_trades': selected_data['total_trades'],
                            'median_return': selected_data['median_return'],
                            'win_rate': selected_data['win_rate'],
                            'net_profit': selected_data['net_profit'],
                            'selected_based_on_period': prev_period
                        })
            
            # Create DataFrame for this portfolio and save to portfolios subdirectory
            if portfolio_data:
                portfolio_df = pd.DataFrame(portfolio_data)
                output_path = os.path.join(self.portfolios_dir, f"top{n}_parallel_walk_forward_analysis.csv")
                portfolio_df.to_csv(output_path, index=False)
                self.top_portfolios[f"top{n}"] = portfolio_df
                print(f"Created Top {n} portfolio with {len(portfolio_df)} periods")
                
                # Print some statistics about the selection
                avg_trades = portfolio_df['total_trades'].mean()
                avg_median_return = portfolio_df['median_return'].mean()
                avg_win_rate = portfolio_df['win_rate'].mean()
                print(f"Average metrics for Top {n} portfolio:")
                print(f"  - Trades per period: {avg_trades:.2f}")
                print(f"  - Median return per period: {avg_median_return:.2f}%")
                print(f"  - Win rate: {avg_win_rate:.2f}%")
        
        return self.top_portfolios
    
    def identify_positive_outliers(self, monthly_stats_df, z_threshold=3.0):
        """
        Identify positive outlier months using z-score.
        
        Args:
            monthly_stats_df: DataFrame with monthly statistics
            z_threshold: Z-score threshold for identifying outliers (default: 3.0)
            
        Returns:
            Tuple of (adjusted_df, outlier_info_df)
        """
        if monthly_stats_df.empty or len(monthly_stats_df) < 3:  # Need at least 3 data points for meaningful z-scores
            return monthly_stats_df.copy(), pd.DataFrame()
        
        # Create a copy of the dataframe to work with
        adjusted_df = monthly_stats_df.copy()
        
        # Calculate z-scores for monthly returns
        returns = monthly_stats_df['Monthly Return (%)'].values
        z_scores = zscore(returns)
        
        # Identify positive outliers (z-score > threshold)
        outlier_mask = (z_scores > z_threshold)
        
        # Only consider positive returns as outliers
        positive_outlier_mask = outlier_mask & (returns > 0)
        
        # Create outlier information dataframe
        outlier_info = []
        for i, is_outlier in enumerate(positive_outlier_mask):
            if is_outlier:
                outlier_info.append({
                    'Year-Month': monthly_stats_df['Year-Month'].iloc[i],
                    'Original Return (%)': monthly_stats_df['Monthly Return (%)'].iloc[i],
                    'Z-Score': z_scores[i]
                })
        
        outlier_info_df = pd.DataFrame(outlier_info)
        
        if not outlier_info_df.empty:
            # Calculate trimmed mean (excluding outliers)
            non_outlier_returns = returns[~positive_outlier_mask]
            if len(non_outlier_returns) > 0:
                trimmed_mean = np.mean(non_outlier_returns)
                
                # Replace outliers with trimmed mean
                for i, is_outlier in enumerate(positive_outlier_mask):
                    if is_outlier:
                        # Store the original return in a new column before replacing
                        adjusted_df.at[monthly_stats_df.index[i], 'Original Return (%)'] = adjusted_df.at[monthly_stats_df.index[i], 'Monthly Return (%)']
                        adjusted_df.at[monthly_stats_df.index[i], 'Monthly Return (%)'] = trimmed_mean
                        adjusted_df.at[monthly_stats_df.index[i], 'Is Outlier'] = True
                        adjusted_df.at[monthly_stats_df.index[i], 'Z-Score'] = z_scores[i]
            
            # Recalculate end equity values based on adjusted returns
            start_equity = adjusted_df['Start Equity'].iloc[0]
            
            # Reset the equity values based on the adjusted returns
            for i in range(len(adjusted_df)):
                if i == 0:
                    previous_end = start_equity
                else:
                    previous_end = adjusted_df['End Equity'].iloc[i-1]
                    
                monthly_return = adjusted_df['Monthly Return (%)'].iloc[i]
                adjusted_df.at[adjusted_df.index[i], 'Start Equity'] = previous_end
                adjusted_df.at[adjusted_df.index[i], 'End Equity'] = previous_end * (1 + monthly_return / 100)
        
        return adjusted_df, outlier_info_df
    
    def generate_adjusted_monthly_plots(self, monthly_stats_df, adjusted_df, outlier_info_df, portfolio_name):
        """
        Generate visualizations comparing original and adjusted monthly statistics.
        
        Args:
            monthly_stats_df: Original monthly statistics dataframe
            adjusted_df: Adjusted monthly statistics dataframe
            outlier_info_df: Information about identified outliers
            portfolio_name: Name of the portfolio
        """
        if monthly_stats_df.empty or adjusted_df.empty:
            return
            
        # 1. Comparative Monthly Returns Bar Chart
        plt.figure(figsize=(16, 8))
        
        # Plot both original and adjusted returns
        bar_width = 0.35
        index = np.arange(len(monthly_stats_df))
        
        # Original returns
        original_bars = plt.bar(index, monthly_stats_df['Monthly Return (%)'], 
                               bar_width, alpha=0.7, label='Original')
        
        # Adjusted returns
        adjusted_bars = plt.bar(index + bar_width, adjusted_df['Monthly Return (%)'], 
                               bar_width, alpha=0.7, label='Adjusted')
        
        # Color the bars
        for i, bar in enumerate(original_bars):
            if monthly_stats_df['Monthly Return (%)'].iloc[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        for i, bar in enumerate(adjusted_bars):
            if adjusted_df['Monthly Return (%)'].iloc[i] >= 0:
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightcoral')
        
        # Highlight outliers
        if not outlier_info_df.empty:
            for _, row in outlier_info_df.iterrows():
                month = row['Year-Month']
                idx = monthly_stats_df[monthly_stats_df['Year-Month'] == month].index[0]
                i = monthly_stats_df.index.get_loc(idx)
                
                plt.axvline(x=i + bar_width/2, color='blue', linestyle='--', alpha=0.5)
                plt.text(i + bar_width/2, monthly_stats_df['Monthly Return (%)'].max(),
                        f"Z={row['Z-Score']:.1f}", ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.title(f"{portfolio_name} - Original vs. Adjusted Monthly Returns", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(index + bar_width/2, monthly_stats_df['Year-Month'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the comparative monthly returns chart
        comp_returns_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_comparative_monthly_returns.png")
        plt.savefig(comp_returns_path)
        plt.close()
        
        # 2. Comparative Cumulative Returns
        # Calculate cumulative returns for both original and adjusted
        monthly_stats_df['Original Cumulative Return'] = (1 + monthly_stats_df['Monthly Return (%)'] / 100).cumprod() * 10000
        adjusted_df['Adjusted Cumulative Return'] = (1 + adjusted_df['Monthly Return (%)'] / 100).cumprod() * 10000
        
        plt.figure(figsize=(14, 7))
        plt.plot(monthly_stats_df['Year-Month'], monthly_stats_df['Original Cumulative Return'], 
                marker='o', linestyle='-', label='Original')
        plt.plot(adjusted_df['Year-Month'], adjusted_df['Adjusted Cumulative Return'], 
                marker='s', linestyle='--', label='Adjusted (Outliers Removed)')
        
        # Highlight where outliers were adjusted
        if not outlier_info_df.empty:
            for _, row in outlier_info_df.iterrows():
                month = row['Year-Month']
                idx = monthly_stats_df[monthly_stats_df['Year-Month'] == month].index[0]
                original_value = monthly_stats_df.loc[idx, 'Original Cumulative Return']
                adjusted_value = adjusted_df.loc[idx, 'Adjusted Cumulative Return']
                
                plt.annotate('', xy=(month, adjusted_value), xytext=(month, original_value),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))
        
        plt.title(f"{portfolio_name} - Original vs. Adjusted Cumulative Returns", fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Cumulative Value (Starting: 10,000)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the comparative cumulative returns chart
        comp_cumulative_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_comparative_cumulative_returns.png")
        plt.savefig(comp_cumulative_path)
        plt.close()
        
        # 3. Monthly Heatmap for Adjusted Returns by Year/Month (if we have data across multiple years)
        if len(adjusted_df['Year'].unique()) > 1:
            # Pivot the data to create a year x month grid
            try:
                heatmap_data = adjusted_df.pivot(index='Year', columns='Month', values='Monthly Return (%)')
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
                plt.title(f"{portfolio_name} Adjusted Monthly Returns by Year (%)", fontsize=16)
                
                # Use month names for x-axis labels
                month_labels = [calendar.month_abbr[i] for i in range(1, 13)]
                plt.xticks(np.arange(12) + 0.5, month_labels)
                
                plt.tight_layout()
                
                # Save the adjusted monthly heatmap
                monthly_heatmap_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_adjusted_monthly_heatmap.png")
                plt.savefig(monthly_heatmap_path)
                plt.close()
            except:
                # Skip heatmap if there's an issue with pivoting (e.g., duplicate entries)
                pass
            
            # Create comparative heatmap showing the difference between original and adjusted
            try:
                # Create original heatmap data
                original_heatmap_data = monthly_stats_df.pivot(index='Year', columns='Month', values='Monthly Return (%)')
                
                # Create adjusted heatmap data
                adjusted_heatmap_data = adjusted_df.pivot(index='Year', columns='Month', values='Monthly Return (%)')
                
                # Calculate difference (original - adjusted)
                diff_heatmap_data = original_heatmap_data - adjusted_heatmap_data
                
                plt.figure(figsize=(12, 8))
                # Use a diverging colormap with white at center
                sns.heatmap(diff_heatmap_data, annot=True, cmap='PiYG', center=0, fmt='.1f')
                plt.title(f"{portfolio_name} Difference (Original - Adjusted) Monthly Returns by Year (%)", fontsize=16)
                
                # Use month names for x-axis labels
                month_labels = [calendar.month_abbr[i] for i in range(1, 13)]
                plt.xticks(np.arange(12) + 0.5, month_labels)
                
                plt.tight_layout()
                
                # Save the difference heatmap
                diff_heatmap_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_monthly_returns_difference_heatmap.png")
                plt.savefig(diff_heatmap_path)
                plt.close()
            except:
                # Skip difference heatmap if there's an issue
                pass
        
        # 3. Save outlier information table
        if not outlier_info_df.empty:
            outlier_info_df['Adjusted Return (%)'] = [adjusted_df.loc[monthly_stats_df[monthly_stats_df['Year-Month'] == month].index[0], 'Monthly Return (%)'] 
                                                    for month in outlier_info_df['Year-Month']]
            outlier_info_df['Adjustment'] = outlier_info_df['Original Return (%)'] - outlier_info_df['Adjusted Return (%)']
            
            # Add summary statistics
            summary_data = {
                'Portfolio': [portfolio_name],
                'Number of Months': [len(monthly_stats_df)],
                'Number of Outliers': [len(outlier_info_df)],
                'Outlier Percentage': [len(outlier_info_df) / len(monthly_stats_df) * 100],
                'Original CAGR (%)': [self.calculate_cagr(monthly_stats_df['Original Cumulative Return'].iloc[0], 
                                                    monthly_stats_df['Original Cumulative Return'].iloc[-1], 
                                                    len(monthly_stats_df) / 12)],
                'Adjusted CAGR (%)': [self.calculate_cagr(adjusted_df['Adjusted Cumulative Return'].iloc[0], 
                                                    adjusted_df['Adjusted Cumulative Return'].iloc[-1], 
                                                    len(adjusted_df) / 12)],
                'CAGR Difference (%)': [self.calculate_cagr(monthly_stats_df['Original Cumulative Return'].iloc[0], 
                                                    monthly_stats_df['Original Cumulative Return'].iloc[-1], 
                                                    len(monthly_stats_df) / 12) - 
                                      self.calculate_cagr(adjusted_df['Adjusted Cumulative Return'].iloc[0], 
                                                    adjusted_df['Adjusted Cumulative Return'].iloc[-1], 
                                                    len(adjusted_df) / 12)]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_outlier_adjustment_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            # Save detailed outlier info
            outlier_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_outlier_details.csv")
            outlier_info_df.to_csv(outlier_path, index=False)
    
    def calculate_cagr(self, start_value, end_value, years):
        """Calculate Compound Annual Growth Rate"""
        if years <= 0 or start_value <= 0:
            return 0
        return ((end_value / start_value) ** (1 / years) - 1) * 100
    
    def calculate_monthly_statistics(self, equity_df, portfolio_name):
        """
        Calculate monthly performance statistics from equity curve data.
        
        Args:
            equity_df: DataFrame containing the equity curve data
            portfolio_name: Name of the portfolio
            
        Returns:
            DataFrame with monthly statistics
        """
        if equity_df.empty:
            return pd.DataFrame()
        
        # Make sure datetime is sorted
        equity_df = equity_df.sort_values('datetime')
        
        # Extract year and month for grouping
        equity_df['year'] = equity_df['datetime'].dt.year
        equity_df['month'] = equity_df['datetime'].dt.month
        equity_df['year_month'] = equity_df['datetime'].dt.strftime('%Y-%m')
        
        monthly_stats = []
        
        # Group by year and month
        for (year, month), group in equity_df.groupby(['year', 'month']):
            if len(group) < 2:
                continue  # Skip months with insufficient data
                
            # Basic metrics
            start_date = group['datetime'].min()
            end_date = group['datetime'].max()
            start_equity = group['equity'].iloc[0]
            end_equity = group['equity'].iloc[-1]
            max_equity = group['equity'].max()
            min_equity = group['equity'].min()
            
            # Return calculations
            monthly_return_pct = ((end_equity / start_equity) - 1) * 100
            
            # Drawdown calculations
            group['previous_peak'] = group['equity'].cummax()
            group['drawdown'] = (group['equity'] / group['previous_peak'] - 1) * 100
            max_drawdown_pct = group['drawdown'].min()
            
            # Daily returns for volatility
            group['daily_return'] = group['equity'].pct_change()
            daily_returns = group['daily_return'].dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
            
            # Trade identification (similar to the overall statistics method)
            group['equity_diff'] = group['equity'].diff()
            group['trade_direction'] = np.sign(group['equity_diff'])
            group['trade_changed'] = group['trade_direction'].diff().fillna(0) != 0
            group['trade_id'] = group['trade_changed'].cumsum()
            
            trades = []
            for trade_id, trade_group in group.groupby('trade_id'):
                if len(trade_group) >= 2:
                    trade_start = trade_group['equity'].iloc[0]
                    trade_end = trade_group['equity'].iloc[-1]
                    trade_return = (trade_end / trade_start - 1) * 100
                    
                    trades.append({
                        'id': trade_id,
                        'return': trade_return,
                        'profitable': trade_return > 0
                    })
            
            # Calculate win rate and profit metrics
            if trades:
                profitable_trades = [t for t in trades if t['profitable']]
                num_trades = len(trades)
                win_rate = len(profitable_trades) / num_trades * 100 if num_trades > 0 else 0
            else:
                num_trades = 0
                win_rate = 0
            
            # Month name for readability
            month_name = calendar.month_name[month]
            
            # Compile statistics for this month
            monthly_stats.append({
                'Portfolio': portfolio_name,
                'Year': year,
                'Month': month,
                'Month Name': month_name,
                'Year-Month': f"{year}-{month:02d}",
                'Start Date': start_date,
                'End Date': end_date,
                'Start Equity': start_equity,
                'End Equity': end_equity,
                'Monthly Return (%)': monthly_return_pct,
                'Max Drawdown (%)': max_drawdown_pct,
                'Volatility (%)': volatility,
                'Number of Trades': num_trades,
                'Win Rate (%)': win_rate
            })
        
        # Convert to DataFrame
        monthly_df = pd.DataFrame(monthly_stats)
        
        # Sort by date
        if not monthly_df.empty:
            monthly_df = monthly_df.sort_values(['Year', 'Month'])
        
        return monthly_df
    
    def generate_equity_curves(self):
        """Generate equity curves for each top portfolio"""
        all_portfolio_stats = []
        
        for portfolio_name, portfolio_df in self.top_portfolios.items():
            equity_curve_data = []
            last_equity = 10000.0  # Starting equity
            
            for _, row in portfolio_df.iterrows():
                symbol = row['symbol']
                period = row['original_period']
                out_sample_range = row['out_sample_range']
                
                # Extract start and end dates
                date_parts = out_sample_range.split(' to ')
                start_date = pd.to_datetime(date_parts[0])
                end_date = pd.to_datetime(date_parts[1])
                
                # Load equity curve data for this symbol
                if symbol in self.equity_files:
                    equity_df = pd.read_csv(self.equity_files[symbol])
                    equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
                    
                    # Find starting point (equity = 10000) closest to our period start date
                    equity_reset_points = equity_df[equity_df['equity'] == 10000.0].copy()
                    
                    if not equity_reset_points.empty:
                        # Find the closest reset point to our period start date
                        equity_reset_points['date_diff'] = abs(equity_reset_points['datetime'] - start_date)
                        closest_reset = equity_reset_points.loc[equity_reset_points['date_diff'].idxmin()]
                        
                        # Get the index of this reset point
                        reset_idx = equity_df[equity_df['datetime'] == closest_reset['datetime']].index[0]
                        
                        # Find the next reset point or end of data
                        next_reset_points = equity_df[(equity_df['equity'] == 10000.0) & 
                                                     (equity_df.index > reset_idx)]
                        
                        if not next_reset_points.empty:
                            next_reset_idx = next_reset_points.index[0]
                            period_equity = equity_df.iloc[reset_idx:next_reset_idx].copy()
                        else:
                            # If no next reset point, use until end of data
                            period_equity = equity_df.iloc[reset_idx:].copy()
                        
                        # Scale equity based on previous end value
                        if equity_curve_data:
                            # Scale factor is last_equity / 10000 (the starting equity in each period)
                            scale_factor = last_equity / 10000.0
                            period_equity['equity'] = period_equity['equity'] * scale_factor
                        
                        # Add to the combined equity curve
                        for _, eq_row in period_equity.iterrows():
                            equity_curve_data.append({
                                'datetime': eq_row['datetime'],
                                'equity': eq_row['equity'],
                                'volume': eq_row.get('volume', np.nan),
                                'symbol': symbol,
                                'normalized_period': row['normalized_period']
                            })
                        
                        # Update last equity for next period
                        if not period_equity.empty:
                            last_equity = period_equity['equity'].iloc[-1]
            
            # Create DataFrame for this portfolio's equity curve and save to equity_curves subdirectory
            if equity_curve_data:
                equity_df = pd.DataFrame(equity_curve_data)
                equity_df.sort_values('datetime', inplace=True)
                
                # Save the equity curve
                output_path = os.path.join(self.equity_curves_dir, f"{portfolio_name}_equity_curve.csv")
                equity_df.to_csv(output_path, index=False)
                
                # Calculate detailed trading statistics from the equity curve
                stats = self.calculate_trading_statistics(equity_df, portfolio_name, portfolio_df)
                all_portfolio_stats.append(stats)
                
                # Calculate monthly statistics
                monthly_stats_df = self.calculate_monthly_statistics(equity_df, portfolio_name)
                if not monthly_stats_df.empty:
                    # Save monthly statistics
                    monthly_stats_path = os.path.join(self.monthly_dir, f"{portfolio_name}_monthly_statistics.csv")
                    monthly_stats_df.to_csv(monthly_stats_path, index=False)
                    
                    # Create monthly return bar chart
                    self.create_monthly_return_plots(monthly_stats_df, portfolio_name)
                    
                    # Identify positive outliers and generate adjusted statistics
                    adjusted_monthly_df, outlier_info = self.identify_positive_outliers(monthly_stats_df, z_threshold=3.0)
                    
                    # Save adjusted monthly statistics
                    if not adjusted_monthly_df.empty:
                        adjusted_path = os.path.join(self.adjusted_monthly_dir, f"{portfolio_name}_adjusted_monthly_statistics.csv")
                        adjusted_monthly_df.to_csv(adjusted_path, index=False)
                        
                        # Generate comparative visualizations
                        self.generate_adjusted_monthly_plots(monthly_stats_df, adjusted_monthly_df, outlier_info, portfolio_name)
                
                # Plot 1: Standard equity curve plot
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df['datetime'], equity_df['equity'])
                plt.title(f"{portfolio_name} Equity Curve")
                plt.xlabel('Date')
                plt.ylabel('Equity')
                plt.grid(True)
                plot_path = os.path.join(self.plots_dir, f"{portfolio_name}_equity_curve_plot.png")
                plt.savefig(plot_path)
                plt.close()
                
                # Plot 2: Enhanced datetime x-axis plot with proper formatting
                fig, ax = plt.subplots(figsize=(15, 7))
                ax.plot(equity_df['datetime'], equity_df['equity'], linewidth=2)
                
                # Format the x-axis to show dates properly
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                
                # Rotate date labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Add grid, title and labels
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_title(f"{portfolio_name} Equity Curve (Time-Based)", fontsize=16)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Equity', fontsize=12)
                
                # Add annotations for key points
                if len(equity_df) > 0:
                    # Mark start point
                    start_point = equity_df.iloc[0]
                    ax.scatter(start_point['datetime'], start_point['equity'], color='green', s=100, zorder=5)
                    ax.annotate('Start', (start_point['datetime'], start_point['equity']), 
                                xytext=(10, 10), textcoords='offset points', fontsize=10)
                    
                    # Mark end point
                    end_point = equity_df.iloc[-1]
                    ax.scatter(end_point['datetime'], end_point['equity'], color='red', s=100, zorder=5)
                    ax.annotate('End', (end_point['datetime'], end_point['equity']), 
                                xytext=(10, -15), textcoords='offset points', fontsize=10)
                    
                    # Mark highest equity point
                    max_equity_idx = equity_df['equity'].idxmax()
                    max_point = equity_df.loc[max_equity_idx]
                    ax.scatter(max_point['datetime'], max_point['equity'], color='blue', s=100, zorder=5)
                    ax.annotate('Max', (max_point['datetime'], max_point['equity']), 
                                xytext=(10, 10), textcoords='offset points', fontsize=10)
                
                # Adjust layout to make room for datetime labels
                plt.tight_layout()
                
                # Save the time-based plot
                time_plot_path = os.path.join(self.plots_dir, f"{portfolio_name}_equity_curve_datetime_plot.png")
                plt.savefig(time_plot_path)
                plt.close()
                
                # Plot 3: Trade-based plot (sequential trade numbers on x-axis)
                equity_df_reset = equity_df.reset_index(drop=True)
                equity_df_reset['trade_number'] = equity_df_reset.index + 1
                
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df_reset['trade_number'], equity_df_reset['equity'])
                plt.title(f"{portfolio_name} Equity Curve (By Trade Sequence)")
                plt.xlabel('Trade Sequence Number')
                plt.ylabel('Equity')
                plt.grid(True)
                
                # Add annotations for start, end, and max points
                if len(equity_df_reset) > 0:
                    start_point = equity_df_reset.iloc[0]
                    plt.scatter(start_point['trade_number'], start_point['equity'], color='green', s=100, zorder=5)
                    plt.annotate('Start', (start_point['trade_number'], start_point['equity']), 
                                xytext=(10, 10), textcoords='offset points')
                    
                    end_point = equity_df_reset.iloc[-1]
                    plt.scatter(end_point['trade_number'], end_point['equity'], color='red', s=100, zorder=5)
                    plt.annotate('End', (end_point['trade_number'], end_point['equity']), 
                                xytext=(10, -15), textcoords='offset points')
                    
                    max_equity_idx = equity_df_reset['equity'].idxmax()
                    max_point = equity_df_reset.loc[max_equity_idx]
                    plt.scatter(max_point['trade_number'], max_point['equity'], color='blue', s=100, zorder=5)
                    plt.annotate('Max', (max_point['trade_number'], max_point['equity']), 
                                xytext=(10, 10), textcoords='offset points')
                
                # Save the trade sequence plot
                trade_plot_path = os.path.join(self.plots_dir, f"{portfolio_name}_equity_curve_trade_plot.png")
                plt.savefig(trade_plot_path)
                plt.close()
                
                print(f"Generated equity curve, statistics, and plots for {portfolio_name}")
        
        # Create a comparative summary of all portfolios
        if all_portfolio_stats:
            combined_stats = pd.DataFrame(all_portfolio_stats)
            combined_stats.sort_values('Total Return (%)', ascending=False, inplace=True)
            combined_stats_path = os.path.join(self.output_dir, "portfolios_comparative_summary.csv")
            combined_stats.to_csv(combined_stats_path, index=False)
            print(f"Created comparative summary of all portfolios")
    
    def create_monthly_return_plots(self, monthly_stats_df, portfolio_name):
        """
        Create monthly return plots for a portfolio.
        
        Args:
            monthly_stats_df: DataFrame with monthly statistics
            portfolio_name: Name of the portfolio
        """
        if monthly_stats_df.empty:
            return
        
        # 1. Monthly Returns Bar Chart
        plt.figure(figsize=(14, 7))
        bars = plt.bar(monthly_stats_df['Year-Month'], monthly_stats_df['Monthly Return (%)'])
        
        # Color the bars based on positive/negative returns
        for i, bar in enumerate(bars):
            if monthly_stats_df['Monthly Return (%)'].iloc[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.title(f"{portfolio_name} Monthly Returns", fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the monthly returns bar chart
        monthly_returns_path = os.path.join(self.monthly_dir, f"{portfolio_name}_monthly_returns.png")
        plt.savefig(monthly_returns_path)
        plt.close()
        
        # 2. Monthly Equity Curve (End of Month Values)
        plt.figure(figsize=(14, 7))
        plt.plot(monthly_stats_df['Year-Month'], monthly_stats_df['End Equity'], marker='o', linestyle='-')
        plt.title(f"{portfolio_name} Monthly Equity Curve", fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Equity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the monthly equity curve
        monthly_equity_path = os.path.join(self.monthly_dir, f"{portfolio_name}_monthly_equity.png")
        plt.savefig(monthly_equity_path)
        plt.close()
        
        # 3. Monthly Heatmap by Year/Month (if we have data across multiple years)
        if len(monthly_stats_df['Year'].unique()) > 1:
            # Pivot the data to create a year x month grid
            try:
                heatmap_data = monthly_stats_df.pivot(index='Year', columns='Month', values='Monthly Return (%)')
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
                plt.title(f"{portfolio_name} Monthly Returns by Year (%)", fontsize=16)
                
                # Use month names for x-axis labels
                month_labels = [calendar.month_abbr[i] for i in range(1, 13)]
                plt.xticks(np.arange(12) + 0.5, month_labels)
                
                plt.tight_layout()
                
                # Save the monthly heatmap
                monthly_heatmap_path = os.path.join(self.monthly_dir, f"{portfolio_name}_monthly_heatmap.png")
                plt.savefig(monthly_heatmap_path)
                plt.close()
            except:
                # Skip heatmap if there's an issue with pivoting (e.g., duplicate entries)
                pass
        
        # 4. Cumulative Monthly Returns
        monthly_stats_df['Cumulative Return'] = (1 + monthly_stats_df['Monthly Return (%)'] / 100).cumprod() * 10000
        
        plt.figure(figsize=(14, 7))
        plt.plot(monthly_stats_df['Year-Month'], monthly_stats_df['Cumulative Return'], marker='o', linestyle='-')
        plt.title(f"{portfolio_name} Cumulative Monthly Returns", fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Cumulative Value (Starting: 10,000)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the cumulative monthly returns chart
        cumulative_path = os.path.join(self.monthly_dir, f"{portfolio_name}_cumulative_monthly_returns.png")
        plt.savefig(cumulative_path)
        plt.close()
    
    def calculate_trading_statistics(self, equity_df, portfolio_name, portfolio_df):
        """
        Calculate comprehensive trading statistics from equity curve data.
        """
        # Ensure we have data
        if equity_df.empty:
            return {
                'Portfolio': portfolio_name,
                'Start Date': None,
                'End Date': None,
                'Total Return (%)': 0,
                'CAGR (%)': 0,
                'Max Drawdown (%)': 0,
                'Recovery Factor': 0,
                'Sharpe Ratio': 0,
                'Sortino Ratio': 0,
                'Calmar Ratio': 0,
                'Volatility (%)': 0,
                'Win Rate (%)': 0,
                'Profit Factor': 0,
                'Average Profit (%)': 0,
                'Average Loss (%)': 0,
                'Profit/Loss Ratio': 0,
                'Number of Trades': 0,
                'Number of Periods': len(portfolio_df)
            }
        
        # Basic information
        start_date = equity_df['datetime'].min()
        end_date = equity_df['datetime'].max()
        start_equity = equity_df['equity'].iloc[0]  # First equity value
        end_equity = equity_df['equity'].iloc[-1]  # Last equity value
        
        # Calculate returns
        total_return_pct = ((end_equity / start_equity) - 1) * 100
        
        # Calculate CAGR (Compound Annual Growth Rate)
        days = (end_date - start_date).days
        years = days / 365.25
        cagr = ((end_equity / start_equity) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
        
        # Calculate drawdowns
        equity_df['previous_peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['previous_peak'] - 1) * 100
        max_drawdown_pct = equity_df['drawdown'].min()
        
        # Recovery Factor
        recovery_factor = abs(total_return_pct / min(max_drawdown_pct, -0.01)) if max_drawdown_pct < 0 else 0
        
        # Calculate daily returns
        equity_df = equity_df.sort_values('datetime')
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        
        # Calculate risk metrics - drop NA values to avoid issues with first row
        daily_returns = equity_df['daily_return'].dropna().values
        
        if len(daily_returns) > 1:
            # Volatility (annualized standard deviation of returns)
            volatility = np.std(daily_returns) * np.sqrt(252) * 100
            
            # Sharpe Ratio (assuming risk-free rate of 0% for simplicity)
            avg_daily_return = np.mean(daily_returns)
            sharpe_ratio = (avg_daily_return * 252) / (np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
            
            # Sortino Ratio (using only negative returns for risk)
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0.0001
            sortino_ratio = (avg_daily_return * 252) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar Ratio (annualized return / maximum drawdown)
            calmar_ratio = cagr / abs(min(max_drawdown_pct, -0.01)) if max_drawdown_pct < 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        # Identify individual trades
        # A trade is defined as a continuous movement in equity before a reversal
        equity_df['equity_diff'] = equity_df['equity'].diff()
        equity_df['trade_direction'] = np.sign(equity_df['equity_diff'])
        equity_df['trade_changed'] = equity_df['trade_direction'].diff().fillna(0) != 0
        equity_df['trade_id'] = equity_df['trade_changed'].cumsum()
        
        # Calculate individual trade statistics
        trades = []
        for trade_id, group in equity_df.groupby('trade_id'):
            if len(group) >= 2:  # Ensure we have at least start and end points
                trade_start = group['equity'].iloc[0]
                trade_end = group['equity'].iloc[-1]
                trade_return = (trade_end / trade_start - 1) * 100
                
                trades.append({
                    'id': trade_id,
                    'return': trade_return,
                    'profitable': trade_return > 0
                })
        
        # Win rate and profit metrics
        if trades:
            profitable_trades = [t for t in trades if t['profitable']]
            losing_trades = [t for t in trades if not t['profitable']]
            
            win_rate = len(profitable_trades) / len(trades) * 100 if trades else 0
            
            # Calculate profit factor (sum of gains / sum of losses)
            total_gains = sum(t['return'] for t in profitable_trades) if profitable_trades else 0
            total_losses = abs(sum(t['return'] for t in losing_trades)) if losing_trades else 0.0001
            profit_factor = total_gains / total_losses if total_losses > 0 else 0
            
            # Average profit/loss
            avg_profit = np.mean([t['return'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_profit = 0
            avg_loss = 0
            profit_loss_ratio = 0
        
        # Create detailed statistics dictionary
        stats = {
            'Portfolio': portfolio_name,
            'Start Date': start_date,
            'End Date': end_date,
            'Total Return (%)': total_return_pct,
            'CAGR (%)': cagr,
            'Max Drawdown (%)': max_drawdown_pct,
            'Recovery Factor': recovery_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Volatility (%)': volatility,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Average Profit (%)': avg_profit,
            'Average Loss (%)': avg_loss,
            'Profit/Loss Ratio': profit_loss_ratio,
            'Number of Trades': len(trades),
            'Number of Periods': len(portfolio_df)
        }
        
        # Save detailed statistics to CSV
        stats_df = pd.DataFrame([stats])
        stats_path = os.path.join(self.stats_dir, f"{portfolio_name}_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        
        return stats
    
    def run_backtest(self, num_top_portfolios: int = 10):
        """Run the complete backtest process"""
        print("Starting portfolio backtest...")
        
        # Load data
        if not self.load_data():
            print("No data found. Please check the input directory.")
            return False
        
        # Normalize periods
        self.normalize_periods()
        
        # Create top portfolios
        self.create_top_portfolios(num_top_portfolios)
        
        # Generate equity curves
        self.generate_equity_curves()
        
        print("Portfolio backtest completed successfully!")
        return True


if __name__ == "__main__":
    # Example usage
    input_directory = "/Users/fabio/Downloads/output_20250315_101842_2"  # Change this to your input directory
    performance_metric = "Out-of-sample net profit (%)"  # Change this to your preferred metric
    
    # No output_dir specified, will create timestamped directory automatically
    backtest = PortfolioBacktest(
        input_dir=input_directory,
        performance_metric=performance_metric
    )
    
    backtest.run_backtest(num_top_portfolios=10) 