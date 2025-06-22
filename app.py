# This web app is designed to allow users to explore how the Greeks (i.e., Delta, Gamma, Theta, Vega, and Rho) affect option pricing under the 
# Black-Scholes-Merton (BSM) model by adjusting input parameters (ex: strike price, time to maturity, volatility, interest rate, etc.) using sliders.
# The web app was designed using Streamlit and plotly.

import streamlit as st
import numpy as np
from greeks_calculator import price, delta, gamma, vega, theta, rho
from utils import fetch_chain, clean_chain, add_greeks_to_chain, simulate_strategy
from plot_helpers import (
    plot_greeks_surface, plot_greeks_2d,
    plot_option_chain, plot_chain_overlay,
    plot_strategy_exposure, plot_pnl_surface,
    plot_trade_scanner_table
)
import yfinance as yf
import json

st.set_page_config(page_title="Options Risk Diagnostics", layout="wide")

# --- Sidebar: Global Parameters ---
st.sidebar.header("Market & Option Parameters")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
r = st.sidebar.number_input("Risk-free rate (r)", value=0.03)
q = st.sidebar.number_input("Dividend yield (q)", value=0.0)

# --- Tabs ---
tabs = st.tabs([
    "Tab 1 - Greeks Explorer",
    "Tab 2 - Chain Analyzer",
    "Tab 3 - Strategy Simulator",
    "Tab 4 - Trade Scanner"
])

# =======================
# Tab 1: Greeks Explorer
# =======================
with tabs[0]:
    st.header("Greeks Sensitivity Explorer")
    st.subheader("Developed by Riju Datta")
    st.markdown("Explore how option Greeks behave across different market conditions.")

    col1, col2 = st.columns(2)
    with col1:
        S = st.slider("Spot Price (S)", 50, 300, 150)
        K = st.slider("Strike Price (K)", 50, 300, 150)
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.3)
    with col2:
        T = st.slider("Time to Maturity (T, in years)", 0.01, 2.0, 0.5)
        option_type = st.selectbox("Option Type", ["call", "put"])
        mesh_choice = st.selectbox("Meshgrid", ["S vs σ", "S vs T", "σ vs T"])

    # Meshgrid based on user choice
    s_vals = np.linspace(50, 300, 40)
    sigma_vals = np.linspace(0.05, 1.0, 40)
    T_vals = np.linspace(0.01, 2.0, 40)

    # Greek selection
    selected_greek = st.radio(
        "Select Greek to View:",
        ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        horizontal=True
    )

    # Inline interpretation
    greek_descriptions = {
        "Delta": "📉 **Delta** measures how much the option price changes with a $1 change in the underlying asset's price.",
        "Gamma": "🔁 **Gamma** measures how much the Delta changes with a $1 change in the underlying asset's price (i.e., convexity of Delta).",
        "Vega": "🌪️ **Vega** measures sensitivity of the option price to changes in implied volatility.",
        "Theta": "⏳ **Theta** measures how much the option price decays per day, holding everything else constant.",
        "Rho": "💰 **Rho** measures sensitivity of the option price to changes in interest rates."
    }

    st.markdown(greek_descriptions[selected_greek])

    if mesh_choice == "S vs σ":
        X, Y = np.meshgrid(s_vals, sigma_vals)

        if selected_greek == "Delta":
            Z = delta(X, K, T, Y, r, q, option_type)
        elif selected_greek == "Gamma":
            Z = gamma(X, K, T, Y, r, q)
        elif selected_greek == "Vega":
            Z = vega(X, K, T, Y, r, q)
        elif selected_greek == "Theta":
            Z = theta(X, K, T, Y, r, q, option_type)
        elif selected_greek == "Rho":
            Z = rho(X, K, T, Y, r, q, option_type)

        st.plotly_chart(plot_greeks_surface(X, Y, Z, "S", "σ", selected_greek), use_container_width=True)

    elif mesh_choice == "S vs T":
        X, Y = np.meshgrid(s_vals, T_vals)

        if selected_greek == "Delta":
            Z = delta(X, K, T, Y, r, q, option_type)
        elif selected_greek == "Gamma":
            Z = gamma(X, K, T, Y, r, q)
        elif selected_greek == "Vega":
            Z = vega(X, K, T, Y, r, q)
        elif selected_greek == "Theta":
            Z = theta(X, K, T, Y, r, q, option_type)
        elif selected_greek == "Rho":
            Z = rho(X, K, T, Y, r, q, option_type)

        st.plotly_chart(plot_greeks_surface(X, Y, Z, "S", "T", selected_greek), use_container_width=True)

    elif mesh_choice == "σ vs T":
        X, Y = np.meshgrid(sigma_vals, T_vals)

        if selected_greek == "Delta":
            Z = delta(X, K, T, Y, r, q, option_type)
        elif selected_greek == "Gamma":
            Z = gamma(X, K, T, Y, r, q)
        elif selected_greek == "Vega":
            Z = vega(X, K, T, Y, r, q)
        elif selected_greek == "Theta":
            Z = theta(X, K, T, Y, r, q, option_type)
        elif selected_greek == "Rho":
            Z = rho(X, K, T, Y, r, q, option_type)

        st.plotly_chart(plot_greeks_surface(X, Y, Z, "σ", "T", selected_greek), use_container_width=True)


    st.subheader("Current Greeks at Selected Point")
    st.write({
        "Price": round(price(S, K, T, sigma, r, q, option_type), 4),
        "Delta": round(delta(S, K, T, sigma, r, q, option_type), 4),
        "Gamma": round(gamma(S, K, T, sigma, r, q), 4),
        "Vega": round(vega(S, K, T, sigma, r, q), 4),
        "Theta": round(theta(S, K, T, sigma, r, q, option_type), 4),
        "Rho": round(rho(S, K, T, sigma, r, q, option_type), 4),
    })

# =======================
# Tab 2: Chain Analyzer
# =======================
with tabs[1]:
    st.header("Real Options Chain Analyzer")

    # Fetch valid expiries for this ticker
    ticker_obj = yf.Ticker(ticker)
    try:
        valid_expiries = ticker_obj.options
    except Exception:
        st.error("Could not fetch options data for this ticker.")
        st.stop()

    if not valid_expiries:
        st.error("No available option expirations for this ticker.")
        st.stop()

    expiry = st.selectbox("Select Expiry Date", valid_expiries, index=0)
    show_chain = st.button("Fetch & Analyze Chain")

    if show_chain:
        raw_df = fetch_chain(ticker, expiry)
        clean_df = clean_chain(raw_df)
        chain_with_greeks = add_greeks_to_chain(clean_df, r, q)

        greek = st.selectbox("Plot Greek vs Strike", ["delta", "gamma", "theta", "vega", "rho"])
        fig_chain = plot_option_chain(chain_with_greeks, greek)
        st.plotly_chart(fig_chain, use_container_width=True)

        if st.checkbox("Overlay Theoretical Model"):
            K_vals = chain_with_greeks["strike"].values
            T_vals = chain_with_greeks["T"].values
            σ_vals = chain_with_greeks["impliedVolatility"].values
            model_vals = delta(S, K_vals, T_vals, σ_vals, r, q)
            overlay_fig = plot_chain_overlay(
                K_vals, model_vals,
                K_vals, chain_with_greeks[greek.lower()].values,
                greek
            )
            st.plotly_chart(overlay_fig, use_container_width=True)

# =======================
# Tab 3: Strategy Simulator
# =======================
with tabs[2]:
    st.header("Options Strategy Risk Simulator")
    st.markdown("Build a strategy and simulate its Greek exposures and PnL under changing market scenarios.")

    strategy_input = st.text_area("Enter strategy (JSON list)", 
        '[{"type": "call", "strike": 150, "position": 1, "expiry": 0.5}]')

    try:
        legs = json.loads(strategy_input)
        s_grid = np.linspace(50, 300, 30)
        vol_grid = np.linspace(0.05, 1.0, 30)

        # Simulate PnL for each (S, vol) pair
        pnl_matrix = np.zeros((len(vol_grid), len(s_grid)))

        for i, sigma_val in enumerate(vol_grid):
            for j, spot_val in enumerate(s_grid):
                spot_paths = np.array([[spot_val]])  # shape (1, 1)
                vol_paths = np.array([[sigma_val]])  # shape (1, 1)
                sim_df = simulate_strategy(legs, spot_paths, vol_paths, r, q)
                pnl_matrix[i, j] = sim_df["pnl"].values[0]

        # Plot
        st.plotly_chart(plot_pnl_surface(pnl_matrix, s_grid, vol_grid), use_container_width=True)
    except Exception as e:
        st.error(f"Error parsing strategy input: {e}")

# =======================
# Tab 4: Trade Scanner
# =======================
with tabs[3]:
    st.header("Real-Time Trade Scanner")
    st.markdown("Sort by Vega/Gamma/Theta to identify top-risk contracts.")

    if "chain_with_greeks" in locals():
        metric = st.selectbox("Sort By", ["vega", "gamma", "theta"])
        min_iv = st.slider("Min IV", 0.0, 1.0, 0.1)
        min_vol = st.slider("Min Volume", 0, 5000, 100)

        filtered = chain_with_greeks[
            (chain_with_greeks["impliedVolatility"] >= min_iv) &
            (chain_with_greeks["volume"] >= min_vol)
        ].sort_values(by=metric, ascending=False).head(15)

        plot_trade_scanner_table(filtered)
    else:
        st.warning("Load option chain in Tab 2 before scanning.")

# =======================
# Tab 5: Glossary
# =======================
with tabs[4]:
    st.header("Glossary & Interpretation")
    st.markdown("""
    **Delta**: Sensitivity of option price to underlying price movement.  
    **Gamma**: Sensitivity of Delta to price movement.  
    **Vega**: Sensitivity to volatility changes.  
    **Theta**: Time decay — how much the option loses daily.  
    **Rho**: Sensitivity to interest rate changes.  

    All Greeks are dynamic — they evolve with price, volatility, and time. Use the Explorer tab to build intuition.  
    """)

