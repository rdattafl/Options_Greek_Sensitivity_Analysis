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
    IV vs strike scatter, colour = skew-z.  Legend (Call / Put) is moved to
    the top-left and the colour-bar is nudged right to avoid overlap.
    """
    fig = go.Figure()
    for opt_type, colr in [("call", "royalblue"), ("put", "firebrick")]:
        sub = df[df["type"] == opt_type]
        if sub.empty:
            continue
        # only *one* trace gets the colour-bar so we don't have two bars
        showscale = opt_type == "call"
        fig.add_trace(
            go.Scatter(
                x=sub["strike"],
                y=sub["iv"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=sub["skew_z"],
                    colorscale="RdBu",
                    showscale=showscale,
                    colorbar=dict(
                        title="Skew z",
                        x=1.05,           # push bar to the right
                        len=0.85,
                    ),
                ),
                name=opt_type.capitalize(),
                text=sub.apply(
                    lambda r: f"{r['dte']}d {r['type'][0].upper()} {r['strike']}", axis=1
                ),
                hovertemplate=(
                    "Strike: %{x}<br>IV: %{y:.2%}<br>Skew z: %{marker.color:.2f}<br>%{text}"
                ),
                showlegend=True,
            )
        )

    fig.update_layout(
        title="IV skew by strike",
        xaxis_title="Strike",
        yaxis_title="IV",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
        ),
        margin=dict(r=90),  # leave room for colour-bar
    )
    return fig


def plot_iv_hv_scatter(df: pd.DataFrame, ratio_thresh: float = 1.2) -> go.Figure:
    """
    Scatter of IV/HV ratio (x) vs IV (y).
    Points to the right of the red line are overpriced ≥ ratio_thresh.
    """
    import numpy as np
    import plotly.graph_objects as go

    if df.empty:
        return go.Figure()

    # Use a colour map: grey if cheap/fair, orange if rich
    colours = np.where(df["iv_hv_ratio"] >= ratio_thresh, "orange", "lightgrey")

    fig = go.Figure(
        data=go.Scatter(
            x=df["iv_hv_ratio"],
            y=df["iv"],
            mode="markers",
            marker=dict(size=8, color=colours),
            text=df.apply(
                lambda r: f"{r['dte']}d {r['type'][0].upper()} {r['strike']}", axis=1
            ),
            hovertemplate="IV/HV: %{x:.2f}<br>IV: %{y:.2%}<br>%{text}",
        )
    )

    # vertical reference = threshold
    fig.add_shape(
        type="line",
        x0=ratio_thresh,
        x1=ratio_thresh,
        y0=0,
        y1=max(df["iv"]) * 1.05,
        line=dict(color="red", dash="dot"),
    )

    fig.update_layout(
        title="IV / HV mispricing",
        xaxis_title="IV ÷ HV  (ratio)",
        yaxis_title="IV",
        showlegend=False,
    )
    return fig