"""
Portfolio Backtest Dashboard Runner

This script runs the interactive dashboard for the portfolio backtest system.
"""

from dashboard import app

if __name__ == '__main__':
    print("Starting Portfolio Backtest Dashboard...")
    print("Access the dashboard at http://127.0.0.1:8050/")
    app.run(debug=True, port=8050) 