# Portfolio Backtest Dashboard

This branch contains the deployment resources for the Portfolio Backtest Dashboard.

## Deployment Options

### Option 1: Deploy to Render (Recommended)

The simplest way to deploy this dashboard is using [Render](https://render.com):

1. Click the button below:

   [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/valenteff/portfolio-dashboard-wfa)

2. Follow the prompts to deploy your application.

### Option 2: Deploy to Heroku

You can also deploy to Heroku:

```bash
heroku create portfolio-dashboard
git push heroku main
```

## Local Development

To run the dashboard locally:

```bash
git clone https://github.com/valenteff/portfolio-dashboard-wfa.git
cd portfolio-dashboard-wfa
pip install -r requirements.txt
python run_dashboard.py
```

Then access the dashboard at http://127.0.0.1:8050/

## Documentation

For complete documentation, please see the [main branch README](https://github.com/valenteff/portfolio-dashboard-wfa/blob/main/README.md).