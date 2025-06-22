# This file will contain methods for modular Plotly charts for Greeks/surfaces/overlays - namely, 3D surfaces and scatter plots,
# heatmaps, and time series data.

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_greeks_surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    x_label: str, y_label: str, z_label: str
) -> go.Figure:
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        title=f'{z_label} Surface',
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        autosize=True
    )
    return fig

def plot_strategy_payoff(payoff_df: pd.DataFrame):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=payoff_df["Spot"],
        y=payoff_df["Net P&L"],
        mode="lines",
        fill="tozeroy",
        name="Payoff"
    ))
    fig.update_layout(
        title="Payoff at Expiration",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Net Profit / Loss",
        template="plotly_dark",
        height=500
    )
    return fig

