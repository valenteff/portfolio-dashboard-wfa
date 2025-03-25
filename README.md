# Portfolio Backtest System

A Python tool for creating and testing portfolios based on walk-forward analysis results from multiple trading symbols.

## Overview

This system takes the results of walk-forward analyses performed on multiple symbols and creates optimal portfolios by ranking symbols according to performance metrics. It then generates consolidated equity curves for these portfolios.

## Features

- Normalizes trading periods across different symbols
- Creates ranked portfolios (top 1 through top N) based on configurable performance metrics
- Prevents look-ahead bias by selecting symbols based on their previous period performance
- Generates combined equity curves for each portfolio strategy
- Visualizes performance with multiple equity curve plot styles (date-based, trade-based)
- Provides detailed monthly performance analysis and visualizations
- Identifies and adjusts positive outlier months using z-score based approach
- Automatically creates timestamped output directories with organized subdirectories
- Calculates comprehensive trading statistics including Sharpe ratio, recovery factor, and win rate
- Generates performance summaries and comparative analysis across portfolios
- Interactive Dash dashboard for exploring and sharing results

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- scipy
- seaborn
- glob
- dash (for interactive dashboard)
- plotly (for interactive dashboard)

## Usage

1. Organize your walk-forward analysis results in a directory with the following files for each symbol:
   - `SYMBOL_parallel_walk_forward_analysis_*.csv` - Contains period performance data
   - `SYMBOL_closed_trade_equity_curves_parallel_*.csv` - Contains trade-by-trade equity data

2. Run the portfolio backtest:

```python
from portfolio_backtest import PortfolioBacktest

# Initialize the backtest with your input directory and preferred performance metric
backtest = PortfolioBacktest(
    input_dir="path/to/results",
    performance_metric="Out-of-sample net profit (%)"
    # output_dir parameter is optional - if not specified, a timestamped directory will be created
)

# Run the complete backtest process
backtest.run_backtest(num_top_portfolios=10)
```

3. Examine the results in the organized output directory:
   - By default, results are saved to a directory named `results_YYYYMMDD_HHMMSS` in the project directory
   - Subdirectories are created for different types of outputs:
     - `normalized_data/`: Original WFA data with normalized periods
     - `portfolios/`: Symbol selections for each top-N portfolio
     - `equity_curves/`: Combined equity curves data for each portfolio
     - `plots/`: Visual plots of the equity curves (three versions per portfolio)
     - `statistics/`: Detailed trading statistics for each portfolio
     - `monthly_analysis/`: Monthly statistics and visualizations
     - `adjusted_monthly_analysis/`: Monthly analysis with positive outliers adjusted
   - Summary files in the main output directory:
     - `portfolios_comparative_summary.csv`: Comparison of all portfolios sorted by performance

4. Launch the interactive dashboard to explore results:

```bash
python run_dashboard.py
```

Then open your browser and navigate to http://127.0.0.1:8050/ to access the dashboard.

## Interactive Dashboard

The system includes an interactive Dash dashboard that allows for dynamic exploration of backtest results:

### Dashboard Features

- **Portfolio Selection**: Easily compare different portfolio strategies
- **Performance Metrics**: Interactive charts showing key performance indicators
- **Equity Curve Analysis**: Zoomable equity curves with drawdown highlighting
- **Monthly Returns Analysis**: Multiple visualizations of monthly performance
- **Outlier Detection**: Visual indication of outlier months with adjustment effects
- **Comparative Analysis**: Side-by-side comparison of original and adjusted performance
- **Responsive Design**: Works on desktop and tablet devices

### Dashboard Installation

The dashboard requires additional packages:

```bash
pip install dash plotly
```

### Sharing Results

The dashboard can be:

1. **Accessed Locally**: Run on your machine for personal analysis
2. **Deployed Internally**: Host on your company's intranet for team access
3. **Deployed to Cloud**: Host on services like Heroku, AWS, or Render for wider access

## Deployment

The dashboard can be deployed to the internet for wider access. The repository includes configuration files for multiple deployment options.

### Option 1: Deploy to Render (Recommended)

1. Create a [Render](https://render.com/) account
2. Connect your GitHub repository
3. Create a new Web Service, select your repository
4. Render will automatically detect the configuration in `render.yaml`
5. Click "Create Web Service"

Your dashboard will be deployed and accessible at the URL provided by Render.

### Option 2: Deploy to Heroku

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Log in to Heroku:
   ```bash
   heroku login
   ```
3. Create a new Heroku app:
   ```bash
   heroku create portfolio-dashboard
   ```
4. Deploy to Heroku:
   ```bash
   git push heroku main
   ```

### Important Notes for Deployment

- The dashboard requires result directories to function. When deploying to a cloud service, you'll need to:
  1. Include sample result directories in your repository, or
  2. Set up a way to upload result files to the deployed environment
  
- For security, you may want to add authentication before deploying sensitive financial data:
  ```python
  # Add to dashboard.py
  import dash_auth
  
  VALID_USERNAME_PASSWORD_PAIRS = {
      'username': 'password'
  }
  
  auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
  ```

## How It Works

### Avoiding Look-Ahead Bias

The system prevents look-ahead bias by selecting symbols based on their performance in the previous period:

1. For each normalized period N (starting from period 2):
   - The system examines all symbols' performance in period N-1
   - Ranks symbols based on their performance metric in period N-1
   - Selects the appropriate symbol (top1 = best performer, top2 = second best, etc.)
   - Uses that selected symbol's performance data for period N

2. Period 1 has no data in the final portfolios, since there's no prior period to base selection on

This approach simulates how a real portfolio would be managed: analyzing recent performance to make allocation decisions for the next period.

### Equity Curve Visualization

The system generates three different visualizations for each portfolio's equity curve:

1. **Standard Plot**: Basic equity curve plot with datetime on the x-axis
2. **Enhanced Datetime Plot**: Formatted datetime plot with annotated key points (start, end, max equity)
3. **Trade Sequence Plot**: Equity plotted against sequential trade numbers rather than dates

These different visualization styles provide multiple perspectives on the portfolio's performance over time.

### Monthly Performance Analysis

The system provides detailed month-by-month analysis of each portfolio's performance:

1. **Monthly Statistics**: CSV file with metrics for each month including:
   - Monthly returns
   - Monthly drawdowns
   - Monthly win rates
   - Number of trades per month

2. **Monthly Visualizations**:
   - **Monthly Returns Bar Chart**: Shows performance of each month with color-coded bars
   - **Monthly Equity Curve**: End-of-month equity values plotted over time
   - **Monthly Returns Heatmap**: Visual representation of returns by year/month (if multi-year data)
   - **Cumulative Monthly Returns**: Growth of investment based on monthly performances

This monthly breakdown helps identify seasonal patterns and consistency in the strategy's performance.

### Outlier Handling

The system identifies and adjusts positive outlier months to provide a more conservative view of performance:

1. **Identification Method**:
   - Uses z-score with a threshold of 3.0 standard deviations to identify positive outlier months
   - Only positive returns (gains) are considered as potential outliers
   - Preserves negative months (losses) in their original form

2. **Adjustment Method**:
   - Replaces identified outliers with a trimmed mean of non-outlier months
   - Recalculates the equity curve based on adjusted monthly returns
   - Maintains both original and adjusted statistics for comparison

3. **Comparative Analysis**:
   - **Side-by-side bar charts**: Shows original vs. adjusted monthly returns
   - **Comparative cumulative returns**: Illustrates the impact of outlier adjustment
   - **Monthly returns heatmaps**: Visual representation of adjusted returns by year/month
   - **Difference heatmap**: Highlights the magnitude of adjustments across year/month
   - **Outlier summary**: Details which months were adjusted and by how much
   - **Performance metrics**: Compares original and adjusted CAGR and other statistics

This feature helps assess how dependent the strategy's performance is on a few exceptional months, providing a more realistic expectation of consistent performance.

### Trading Statistics

For each portfolio, the system calculates a comprehensive set of trading statistics directly from the equity curve:

#### Performance Metrics
- **Total Return (%)**: Overall percentage return of the portfolio
- **CAGR (%)**: Compound Annual Growth Rate
- **Max Drawdown (%)**: Largest percentage drop from peak to trough

#### Risk-Adjusted Metrics
- **Recovery Factor**: Total return divided by maximum drawdown
- **Sharpe Ratio**: Risk-adjusted return (using standard deviation of returns)
- **Sortino Ratio**: Risk-adjusted return (using only negative returns for risk calculation)
- **Calmar Ratio**: CAGR divided by maximum drawdown
- **Volatility (%)**: Annualized standard deviation of returns

#### Trade Analysis
- **Win Rate (%)**: Percentage of profitable trades
- **Profit Factor**: Sum of gains divided by sum of losses
- **Average Profit (%)**: Average percentage gain on winning trades
- **Average Loss (%)**: Average percentage loss on losing trades
- **Profit/Loss Ratio**: Ratio of average profit to average loss
- **Number of Trades**: Total number of trades identified in the equity curve

All statistics are calculated directly from the portfolio equity curve data, providing an accurate assessment of each portfolio's performance.

## Customization

You can modify the ranking metric by changing the `performance_metric` parameter. Available metrics include:
- "Out-of-sample net profit (%)"
- "Out-of-sample max drawdown (%)"
- "Out-of-sample win rate (%)"

## File Format Requirements

### Walk-Forward Analysis Files

Required columns:
- "Period number"
- "Out-of-sample date range" (format: "YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS")
- Performance metrics (e.g., "Out-of-sample net profit (%)")

### Equity Curve Files

Required columns:
- "datetime" (format: "YYYY-MM-DD HH:MM:SS")
- "equity" (numeric values, with 10000 marking period starts)
- "volume" (optional)

## Running the Dashboard

To use the interactive dashboard:

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the dashboard:

```bash
python run_dashboard.py
```

3. Open your web browser and navigate to http://127.0.0.1:8050/

The dashboard will automatically detect all results directories (folders starting with `results_`) and allow you to interactively explore the data. You can:

- Select different results directories
- Compare portfolios side by side
- Analyze equity curves with zooming capability
- View outlier detection results and impact
- Explore monthly performance patterns through multiple visualizations
- Examine detailed statistics for each portfolio

For company presentations, you can share the dashboard by:
- Running it locally during presentations
- Deploying it to your company intranet
- Hosting it on a cloud service for wider access

### Dashboard Screenshots

![Portfolio Dashboard](assets/dashboard_screenshot.png) 