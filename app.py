"""
Portfolio Backtest Dashboard - Deployment Entry Point

This is the main deployment entry point for the dashboard.
"""

from dashboard import server

# This is for Render and other deployment platforms
if __name__ == '__main__':
    # This won't be run on Render, which looks for the server variable
    from dashboard import app
    app.run(debug=False, host='0.0.0.0', port=8080) 