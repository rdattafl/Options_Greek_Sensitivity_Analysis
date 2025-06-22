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

def plot_greeks_2d(
    mesh_x: np.ndarray, mesh_y: np.ndarray, z: np.ndarray,
    x_label: str, y_label: str
) -> go.Figure:
    fig = go.Figure(data=go.Contour(z=z, x=mesh_x, y=mesh_y, colorscale='Viridis'))
    fig.update_layout(
        title='Greek Contour Plot',
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    return fig

def plot_option_chain(df: pd.DataFrame, greek: str) -> go.Figure:
    fig = px.scatter(
        df,
        x='strike',
        y=greek,
        color='option_type',
        symbol='option_type',
        title=f'{greek.capitalize()} vs Strike'
    )
    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title=f'{greek.capitalize()}',
        legend_title='Option Type'
    )
    return fig

def plot_chain_overlay(
    strikes: np.ndarray, model_vals: np.ndarray,
    real_strikes: np.ndarray, real_vals: np.ndarray,
    greek: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=model_vals, mode='lines', name='Model'))
    fig.add_trace(go.Scatter(x=real_strikes, y=real_vals, mode='markers', name='Market'))
    fig.update_layout(
        title=f'{greek.capitalize()} - Model vs Market',
        xaxis_title='Strike Price',
        yaxis_title=f'{greek.capitalize()}'
    )
    return fig

def plot_strategy_exposure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        if greek in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[greek], mode='lines', name=greek.capitalize()))
    fig.update_layout(
        title='Greek Exposure Over Time',
        xaxis_title='Time',
        yaxis_title='Exposure Value',
        legend_title='Greek'
    )
    return fig

def plot_pnl_surface(
    pnl_matrix: np.ndarray, s_grid: np.ndarray, vol_grid: np.ndarray
) -> go.Figure:
    fig = go.Figure(data=[
        go.Surface(z=pnl_matrix, x=s_grid, y=vol_grid, colorscale='Viridis')
    ])
    fig.update_layout(
        title='PnL Surface Over Spot and Volatility',
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='PnL'
        )
    )
    return fig

def plot_trade_scanner_table(df: pd.DataFrame) -> None:
    import streamlit as st
    styled = df.style.background_gradient(
        subset=['vega', 'theta', 'gamma'], cmap='Blues'
    ).format(precision=3)
    st.dataframe(styled)
