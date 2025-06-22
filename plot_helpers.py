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

def _contract_code(row) -> str:
    return f"{row['dte']}d {row['type'][0].upper()} {row['strike']}"


def plot_iv_skew(df: pd.DataFrame) -> go.Figure:
    """
    IV vs strike scatter, colour by |skew_z|.
    """
    fig = go.Figure()
    for opt_type, colr in [("call", "royalblue"), ("put", "firebrick")]:
        sub = df[df["type"] == opt_type]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["strike"],
                y=sub["iv"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=sub["skew_z"],
                    colorscale="RdBu",
                    showscale=True,
                    colorbar=dict(title="Skew z"),
                ),
                name=opt_type.capitalize(),
                text=sub.apply(_contract_code, axis=1),
                hovertemplate=(
                    "Strike: %{x}<br>IV: %{y:.2%}<br>Skew z: %{marker.color:.2f}<br>"
                    "%{text}"
                ),
            )
        )
    fig.update_layout(title="IV skew by strike", xaxis_title="Strike", yaxis_title="IV")
    return fig


def plot_iv_hv_scatter(df: pd.DataFrame) -> go.Figure:
    """
    IV vs HV scatter; all points share same HV so draw a diagonal reference line.
    """
    if df.empty:
        return go.Figure()
    hv_val = float(df["iv"].iloc[0] / df["iv_hv_ratio"].iloc[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["iv_hv_ratio"] * hv_val,
            y=df["iv"],
            mode="markers",
            marker=dict(size=7, color="darkorange"),
            text=df.apply(_contract_code, axis=1),
            hovertemplate="IV/HV: %{x:.2f}<br>IV: %{y:.2%}<br>%{text}",
        )
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=hv_val * 3, y1=hv_val * 3,
                  line=dict(dash="dot", width=1), name="1:1")

    fig.update_layout(
        title="IV vs HV (contracts)",
        xaxis_title="HV × (IV/HV)",
        yaxis_title="IV",
        showlegend=False,
    )
    return fig


def plot_volume_spikes(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of volume multiples for flagged contracts.
    """
    fig = go.Figure(
        data=go.Bar(
            x=df.apply(_contract_code, axis=1),
            y=df["vol_mult"],
            marker_color="seagreen",
            hovertemplate="Vol mult: %{y:.1f}<br>%{x}",
        )
    )
    fig.update_layout(
        title="Volume spike multiples",
        xaxis_title="Contract",
        yaxis_title="Volume / baseline",
    )
    return fig