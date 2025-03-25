import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Initialize the Dash app with a professional theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server  # For deployment
app.title = "Portfolio Backtest Dashboard"

# Find result directories
def get_results_dirs():
    """Get all results directories sorted by creation date (newest first)"""
    dirs = [d for d in os.listdir() if d.startswith("results_")]
    # Sort by creation date (newest first)
    dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return dirs

# Load portfolio summary
def load_summary(results_dir):
    """Load portfolio summary data"""
    summary_path = os.path.join(results_dir, "portfolios_comparative_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        return df
    return pd.DataFrame()

# Load equity curve
def load_equity_curve(results_dir, portfolio):
    """Load equity curve data for a specific portfolio"""
    equity_path = os.path.join(results_dir, "equity_curves", f"{portfolio}_equity_curve.csv")
    if os.path.exists(equity_path):
        df = pd.read_csv(equity_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    return pd.DataFrame()

# Load monthly statistics
def load_monthly_stats(results_dir, portfolio, adjusted=False):
    """Load monthly statistics for a specific portfolio"""
    if adjusted:
        monthly_path = os.path.join(results_dir, "adjusted_monthly_analysis", f"{portfolio}_adjusted_monthly_statistics.csv")
    else:
        monthly_path = os.path.join(results_dir, "monthly_analysis", f"{portfolio}_monthly_statistics.csv")
    
    if os.path.exists(monthly_path):
        df = pd.read_csv(monthly_path)
        return df
    return pd.DataFrame()

# Load outlier details
def load_outlier_details(results_dir, portfolio):
    """Load outlier details for a specific portfolio"""
    outlier_path = os.path.join(results_dir, "adjusted_monthly_analysis", f"{portfolio}_outlier_details.csv")
    if os.path.exists(outlier_path):
        df = pd.read_csv(outlier_path)
        return df
    return pd.DataFrame()

# Load outlier adjustment summary
def load_outlier_summary(results_dir, portfolio):
    """Load outlier adjustment summary for a specific portfolio"""
    summary_path = os.path.join(results_dir, "adjusted_monthly_analysis", f"{portfolio}_outlier_adjustment_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        return df
    return pd.DataFrame()

# Create color scale for performance metrics
def get_color_scale(metric_name):
    """Get appropriate color scale based on metric name"""
    if "Drawdown" in metric_name:
        return "RdYlGn_r"  # Reversed color scale (red is good for small drawdowns)
    return "RdYlGn"  # Default scale (green is good)

# Header section with title and logo
header = html.Div([
    html.Div([
        html.H1("Portfolio Backtest Dashboard", className="display-4"),
        html.P("Interactive analysis of trading strategy performance with outlier detection", className="lead")
    ], className="col-md-10"),
    html.Div([
        html.Img(src="assets/chart-logo.png", height="70px", className="float-right")
    ], className="col-md-2"),
], className="row mb-4 mt-3")

# Controls section
controls = html.Div([
    html.Div([
        html.Label("Select Results Directory"),
        dcc.Dropdown(
            id='results-dir-dropdown',
            options=[{'label': d, 'value': d} for d in get_results_dirs()],
            value=get_results_dirs()[0] if get_results_dirs() else None,
            clearable=False
        ),
    ], className="col-md-4"),
    
    html.Div([
        html.Label("Select Portfolio"),
        dcc.Dropdown(id='portfolio-dropdown', clearable=False)
    ], className="col-md-4"),
    
    html.Div([
        html.Label("Performance Metric"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Total Return (%)', 'value': 'Total Return (%)'},
                {'label': 'CAGR (%)', 'value': 'CAGR (%)'},
                {'label': 'Max Drawdown (%)', 'value': 'Max Drawdown (%)'},
                {'label': 'Sharpe Ratio', 'value': 'Sharpe Ratio'},
                {'label': 'Sortino Ratio', 'value': 'Sortino Ratio'},
                {'label': 'Calmar Ratio', 'value': 'Calmar Ratio'},
                {'label': 'Win Rate (%)', 'value': 'Win Rate (%)'},
                {'label': 'Profit Factor', 'value': 'Profit Factor'}
            ],
            value='CAGR (%)',
            clearable=False
        ),
    ], className="col-md-4"),
], className="row mb-4")

# Summary cards section
summary_cards = html.Div([
    html.Div([
        html.Div([
            html.H5("Portfolio Summary", className="card-header"),
            html.Div([
                dash_table.DataTable(
                    id='portfolio-summary-table',
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        }
                    ]
                )
            ], className="card-body")
        ], className="card h-100")
    ], className="col-md-6"),
    
    html.Div([
        html.Div([
            html.H5("Outlier Adjustment Summary", className="card-header"),
            html.Div([
                dash_table.DataTable(
                    id='outlier-summary-table',
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px'
                    }
                )
            ], className="card-body")
        ], className="card h-100")
    ], className="col-md-6"),
], className="row mb-4")

# Performance comparison chart
performance_section = html.Div([
    html.Div([
        html.Div([
            html.H5("Portfolio Performance Comparison", className="card-header"),
            html.Div([
                dcc.Graph(id='performance-chart')
            ], className="card-body")
        ], className="card")
    ], className="col-md-12"),
], className="row mb-4")

# Equity curve section
equity_section = html.Div([
    html.Div([
        html.Div([
            html.H5("Equity Curve", className="card-header"),
            html.Div([
                dcc.Graph(id='equity-curve'),
                dcc.RangeSlider(
                    id='date-range-slider',
                    min=0,
                    max=1,
                    step=None,
                    marks={},
                    value=[0, 1]
                )
            ], className="card-body")
        ], className="card")
    ], className="col-md-12"),
], className="row mb-4")

# Monthly returns analysis section
monthly_section = html.Div([
    html.Div([
        html.Div([
            html.H5("Monthly Returns Analysis", className="card-header d-flex justify-content-between align-items-center"),
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label="Bar Chart", children=[
                        dcc.Graph(id='monthly-returns-bar')
                    ]),
                    dcc.Tab(label="Heatmap", children=[
                        dcc.Graph(id='monthly-returns-heatmap')
                    ]),
                ])
            ], className="card-body")
        ], className="card")
    ], className="col-md-12"),
], className="row mb-4")

# Original vs Adjusted section
comparison_section = html.Div([
    html.Div([
        html.Div([
            html.H5("Original vs Adjusted Cumulative Returns", className="card-header"),
            html.Div([
                dcc.Graph(id='returns-comparison')
            ], className="card-body")
        ], className="card")
    ], className="col-md-12"),
], className="row mb-4")

# Outlier details section
outlier_section = html.Div([
    html.Div([
        html.Div([
            html.H5("Outlier Details", className="card-header"),
            html.Div([
                dash_table.DataTable(
                    id='outlier-details-table',
                    sort_action='native',
                    filter_action='native',
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f8f9fa'
                        },
                        {
                            'if': {'column_id': 'Z-Score', 'filter_query': '{Z-Score} > 3.5'},
                            'backgroundColor': '#ffcccc'
                        }
                    ]
                )
            ], className="card-body")
        ], className="card")
    ], className="col-md-12"),
], className="row mb-4")

# Footer section
footer = html.Div([
    html.Div([
        html.P(f"© {datetime.now().year} Trading Analytics Dashboard. Generated with Dash.", className="text-muted")
    ], className="col-md-12 text-center mt-3")
], className="row mt-5")

# Construct the layout
app.layout = html.Div([
    html.Div([
        header,
        controls,
        html.Hr(),
        summary_cards,
        performance_section,
        equity_section,
        monthly_section,
        comparison_section,
        outlier_section,
        footer
    ], className="container-fluid py-3 px-4")
])

# Callbacks
@callback(
    Output('portfolio-dropdown', 'options'),
    Output('portfolio-dropdown', 'value'),
    Input('results-dir-dropdown', 'value')
)
def update_portfolio_options(results_dir):
    if not results_dir:
        return [], None
    
    summary_df = load_summary(results_dir)
    if not summary_df.empty:
        portfolios = summary_df["Portfolio"].unique()
        options = [{'label': p, 'value': p} for p in portfolios]
        # Default to top portfolio (typically the best performer)
        default_value = portfolios[0] if len(portfolios) > 0 else None
        return options, default_value
    return [], None

@callback(
    Output('performance-chart', 'figure'),
    Input('results-dir-dropdown', 'value'),
    Input('metric-dropdown', 'value')
)
def update_performance_chart(results_dir, metric):
    if not results_dir or not metric:
        return go.Figure()
    
    summary_df = load_summary(results_dir)
    if summary_df.empty or metric not in summary_df.columns:
        return go.Figure()
    
    # Sort by the selected metric (descending for most metrics, but ascending for drawdown)
    ascending = "drawdown" in metric.lower()
    sorted_df = summary_df.sort_values(metric, ascending=ascending)
    
    # Create bar chart
    fig = px.bar(
        sorted_df, 
        x="Portfolio", 
        y=metric,
        color=metric,
        color_continuous_scale=get_color_scale(metric),
        title=f"Portfolio Comparison by {metric}"
    )
    
    fig.update_layout(
        xaxis_title="Portfolio",
        yaxis_title=metric,
        coloraxis_showscale=False,
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

@callback(
    Output('portfolio-summary-table', 'data'),
    Output('portfolio-summary-table', 'columns'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_portfolio_summary(results_dir, portfolio):
    if not results_dir or not portfolio:
        return [], []
    
    summary_df = load_summary(results_dir)
    if summary_df.empty:
        return [], []
    
    # Filter for selected portfolio
    portfolio_data = summary_df[summary_df["Portfolio"] == portfolio]
    if portfolio_data.empty:
        return [], []
    
    # Format data for display
    display_data = pd.DataFrame({
        'Metric': summary_df.columns[1:],  # Skip the Portfolio column
        'Value': portfolio_data.iloc[0, 1:].values
    })
    
    # Format numeric values to 2 decimal places
    display_data['Value'] = display_data['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    columns = [
        {'name': 'Metric', 'id': 'Metric'},
        {'name': 'Value', 'id': 'Value'}
    ]
    
    return display_data.to_dict('records'), columns

@callback(
    Output('outlier-summary-table', 'data'),
    Output('outlier-summary-table', 'columns'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_outlier_summary(results_dir, portfolio):
    if not results_dir or not portfolio:
        return [], []
    
    summary_df = load_outlier_summary(results_dir, portfolio)
    if summary_df.empty:
        return [], []
    
    # Format for display
    display_data = pd.DataFrame({
        'Metric': summary_df.columns,
        'Value': summary_df.iloc[0].values
    })
    
    # Format numeric values
    display_data['Value'] = display_data['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    columns = [
        {'name': 'Metric', 'id': 'Metric'},
        {'name': 'Value', 'id': 'Value'}
    ]
    
    return display_data.to_dict('records'), columns

@callback(
    Output('date-range-slider', 'min'),
    Output('date-range-slider', 'max'),
    Output('date-range-slider', 'marks'),
    Output('date-range-slider', 'value'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_date_slider(results_dir, portfolio):
    if not results_dir or not portfolio:
        return 0, 1, {}, [0, 1]
    
    equity_df = load_equity_curve(results_dir, portfolio)
    if equity_df.empty or 'datetime' not in equity_df.columns:
        return 0, 1, {}, [0, 1]
    
    # Get min and max dates as timestamps
    min_date = equity_df['datetime'].min().timestamp()
    max_date = equity_df['datetime'].max().timestamp()
    
    # Create slider marks (show only a few dates for clarity)
    date_range = pd.date_range(start=equity_df['datetime'].min(), end=equity_df['datetime'].max(), periods=5)
    marks = {ts.timestamp(): ts.strftime('%Y-%m') for ts in date_range}
    
    return min_date, max_date, marks, [min_date, max_date]

@callback(
    Output('equity-curve', 'figure'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value'),
    Input('date-range-slider', 'value')
)
def update_equity_curve(results_dir, portfolio, date_range):
    if not results_dir or not portfolio:
        return go.Figure()
    
    equity_df = load_equity_curve(results_dir, portfolio)
    if equity_df.empty:
        return go.Figure()
    
    # Filter by date range
    if date_range and len(date_range) == 2:
        min_date = datetime.fromtimestamp(date_range[0])
        max_date = datetime.fromtimestamp(date_range[1])
        equity_df = equity_df[(equity_df['datetime'] >= min_date) & (equity_df['datetime'] <= max_date)]
    
    # Create equity curve figure
    fig = go.Figure()
    
    # Add equity curve line
    fig.add_trace(go.Scatter(
        x=equity_df['datetime'],
        y=equity_df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add drawdown shading
    if 'equity' in equity_df.columns:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        
        # Find drawdown periods
        equity_df['is_drawdown'] = equity_df['drawdown'] < -5  # Only show significant drawdowns
        
        # Create drawdown regions
        drawdown_starts = equity_df.index[equity_df['is_drawdown'] & ~equity_df['is_drawdown'].shift(1, fill_value=False)]
        drawdown_ends = equity_df.index[equity_df['is_drawdown'] & ~equity_df['is_drawdown'].shift(-1, fill_value=False)]
        
        for start, end in zip(drawdown_starts, drawdown_ends):
            if start < end:  # Ensure valid range
                fig.add_vrect(
                    x0=equity_df.loc[start, 'datetime'],
                    x1=equity_df.loc[end, 'datetime'],
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below",
                    line_width=0,
                )
    
    # Update layout
    fig.update_layout(
        title=f"{portfolio} - Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

@callback(
    Output('monthly-returns-bar', 'figure'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_monthly_returns_bar(results_dir, portfolio):
    if not results_dir or not portfolio:
        return go.Figure()
    
    monthly_df = load_monthly_stats(results_dir, portfolio)
    adjusted_df = load_monthly_stats(results_dir, portfolio, adjusted=True)
    
    if monthly_df.empty:
        return go.Figure()
    
    # Create bar chart
    fig = go.Figure()
    
    # Add original returns
    fig.add_trace(go.Bar(
        x=monthly_df['Year-Month'],
        y=monthly_df['Monthly Return (%)'],
        name='Original',
        marker_color=monthly_df['Monthly Return (%)'].apply(lambda x: 'green' if x >= 0 else 'red'),
        opacity=0.7
    ))
    
    # Add adjusted returns if available
    if not adjusted_df.empty and 'Monthly Return (%)' in adjusted_df.columns:
        fig.add_trace(go.Bar(
            x=adjusted_df['Year-Month'],
            y=adjusted_df['Monthly Return (%)'],
            name='Adjusted',
            marker_color=adjusted_df['Monthly Return (%)'].apply(lambda x: 'lightgreen' if x >= 0 else 'lightcoral'),
            opacity=0.7
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{portfolio} - Monthly Returns",
        xaxis_title="Month",
        yaxis_title="Return (%)",
        template="plotly_white",
        barmode='group',
        xaxis=dict(
            tickangle=45,
            type='category'
        ),
        margin=dict(l=50, r=20, t=50, b=70)
    )
    
    return fig

@callback(
    Output('monthly-returns-heatmap', 'figure'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_monthly_returns_heatmap(results_dir, portfolio):
    if not results_dir or not portfolio:
        return go.Figure()
    
    # Load monthly stats
    monthly_df = load_monthly_stats(results_dir, portfolio, adjusted=True)
    if monthly_df.empty or len(monthly_df['Year'].unique()) <= 1:
        # Fallback to regular monthly stats if adjusted not available for heatmap
        monthly_df = load_monthly_stats(results_dir, portfolio)
        if monthly_df.empty or len(monthly_df['Year'].unique()) <= 1:
            # Create empty figure with message if no multi-year data
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough multi-year data for heatmap visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(template="plotly_white")
            return fig
    
    # Create heatmap data
    try:
        # Create pivot table
        heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Monthly Return (%)')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[calendar.month_abbr[m] for m in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 1),
            texttemplate="%{text}%",
            colorbar=dict(title='Return (%)'),
            zmid=0
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{portfolio} - Monthly Returns by Year (%)",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        return fig
    except:
        # Create empty figure if pivoting fails
        fig = go.Figure()
        fig.add_annotation(
            text="Unable to create heatmap from current data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(template="plotly_white")
        return fig

@callback(
    Output('returns-comparison', 'figure'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_returns_comparison(results_dir, portfolio):
    if not results_dir or not portfolio:
        return go.Figure()
    
    # Load monthly stats
    monthly_df = load_monthly_stats(results_dir, portfolio)
    adjusted_df = load_monthly_stats(results_dir, portfolio, adjusted=True)
    
    if monthly_df.empty or adjusted_df.empty:
        return go.Figure()
    
    # Calculate cumulative returns
    try:
        # Original cumulative returns
        if 'Original Cumulative Return' not in monthly_df.columns:
            monthly_df['Original Cumulative Return'] = (1 + monthly_df['Monthly Return (%)'] / 100).cumprod() * 10000
        
        # Adjusted cumulative returns
        if 'Adjusted Cumulative Return' not in adjusted_df.columns:
            adjusted_df['Adjusted Cumulative Return'] = (1 + adjusted_df['Monthly Return (%)'] / 100).cumprod() * 10000
        
        # Create figure
        fig = go.Figure()
        
        # Add original line
        fig.add_trace(go.Scatter(
            x=monthly_df['Year-Month'],
            y=monthly_df['Original Cumulative Return'],
            mode='lines+markers',
            name='Original',
            line=dict(color='royalblue', width=2)
        ))
        
        # Add adjusted line
        fig.add_trace(go.Scatter(
            x=adjusted_df['Year-Month'],
            y=adjusted_df['Adjusted Cumulative Return'],
            mode='lines+markers',
            name='Adjusted (Outliers Removed)',
            line=dict(color='firebrick', width=2, dash='dash')
        ))
        
        # Add indicators for outlier months
        outlier_info = load_outlier_details(results_dir, portfolio)
        if not outlier_info.empty:
            for _, row in outlier_info.iterrows():
                month = row['Year-Month']
                
                # Get values for original and adjusted
                idx_orig = monthly_df[monthly_df['Year-Month'] == month].index
                idx_adj = adjusted_df[adjusted_df['Year-Month'] == month].index
                
                if len(idx_orig) > 0 and len(idx_adj) > 0:
                    orig_val = monthly_df.loc[idx_orig[0], 'Original Cumulative Return']
                    adj_val = adjusted_df.loc[idx_adj[0], 'Adjusted Cumulative Return']
                    
                    # Add annotation arrow
                    fig.add_annotation(
                        x=month,
                        y=orig_val,
                        xref="x",
                        yref="y",
                        text=f"Z={row['Z-Score']:.1f}",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1.5,
                        ax=0,
                        ay=30
                    )
        
        # Update layout
        fig.update_layout(
            title=f"{portfolio} - Original vs. Adjusted Cumulative Returns",
            xaxis_title="Month",
            yaxis_title="Cumulative Value (Starting: 10,000)",
            template="plotly_white",
            xaxis=dict(
                tickangle=45,
                type='category'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=20, t=50, b=70)
        )
        
        return fig
    except Exception as e:
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating comparison chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(template="plotly_white")
        return fig

@callback(
    Output('outlier-details-table', 'data'),
    Output('outlier-details-table', 'columns'),
    Input('results-dir-dropdown', 'value'),
    Input('portfolio-dropdown', 'value')
)
def update_outlier_details(results_dir, portfolio):
    if not results_dir or not portfolio:
        return [], []
    
    outlier_df = load_outlier_details(results_dir, portfolio)
    if outlier_df.empty:
        return [], []
    
    # Format numbers for display
    display_df = outlier_df.copy()
    for col in ['Original Return (%)', 'Z-Score', 'Adjusted Return (%)', 'Adjustment']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    columns = [{"name": col, "id": col} for col in display_df.columns]
    
    return display_df.to_dict('records'), columns

# Run the app
if __name__ == '__main__':
    # Create assets folder if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    app.run(debug=True, port=8050) 