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

st.set_page_config(page_title="Options Risk Diagnostics", layout="wide")

# --- Sidebar: Global Parameters ---
st.sidebar.header("Market & Option Parameters")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
r = st.sidebar.number_input("Risk-free rate (r)", value=0.03)
q = st.sidebar.number_input("Dividend yield (q)", value=0.0)

# --- Tabs ---
tabs = st.tabs([
    "1️⃣ Greeks Explorer",
    "2️⃣ Chain Analyzer",
    "3️⃣ Strategy Simulator",
    "4️⃣ Trade Scanner",
    "5️⃣ Glossary & Interpretation"
])

# =======================
# Tab 1: Greeks Explorer
# =======================
with tabs[0]:
    st.header("Greeks Sensitivity Explorer")
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

if mesh_choice == "S vs σ":
    X, Y = np.meshgrid(s_vals, sigma_vals)
    st.subheader("Meshgrid: Spot Price (S) vs Volatility (σ)")
    
    st.plotly_chart(plot_greeks_surface(X, Y, delta(X, K, T, Y, r, q, option_type), "S", "σ", "Delta"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, gamma(X, K, T, Y, r, q), "S", "σ", "Gamma"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, vega(X, K, T, Y, r, q), "S", "σ", "Vega"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, theta(X, K, T, Y, r, q, option_type), "S", "σ", "Theta"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, rho(X, K, T, Y, r, q, option_type), "S", "σ", "Rho"), use_container_width=True)

elif mesh_choice == "S vs T":
    X, Y = np.meshgrid(s_vals, T_vals)
    st.subheader("Meshgrid: Spot Price (S) vs Time to Maturity (T)")

    st.plotly_chart(plot_greeks_surface(X, Y, delta(X, K, Y, sigma, r, q, option_type), "S", "T", "Delta"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, gamma(X, K, Y, sigma, r, q), "S", "T", "Gamma"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, vega(X, K, Y, sigma, r, q), "S", "T", "Vega"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, theta(X, K, Y, sigma, r, q, option_type), "S", "T", "Theta"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, rho(X, K, Y, sigma, r, q, option_type), "S", "T", "Rho"), use_container_width=True)

elif mesh_choice == "σ vs T":
    X, Y = np.meshgrid(sigma_vals, T_vals)
    st.subheader("Meshgrid: Volatility (σ) vs Time to Maturity (T)")

    st.plotly_chart(plot_greeks_surface(X, Y, delta(S, K, Y, X, r, q, option_type), "σ", "T", "Delta"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, gamma(S, K, Y, X, r, q), "σ", "T", "Gamma"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, vega(S, K, Y, X, r, q), "σ", "T", "Vega"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, theta(S, K, Y, X, r, q, option_type), "σ", "T", "Theta"), use_container_width=True)
    st.plotly_chart(plot_greeks_surface(X, Y, rho(S, K, Y, X, r, q, option_type), "σ", "T", "Rho"), use_container_width=True)


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
    expiry = st.text_input("Expiry Date (YYYY-MM-DD)", value="2025-07-19")
    show_chain = st.button("Fetch & Analyze Chain")

    if show_chain:
        raw_df = fetch_chain(ticker, expiry)
        clean_df = clean_chain(raw_df)
        chain_with_greeks = add_greeks_to_chain(clean_df, r, q)

        greek = st.selectbox("Plot Greek vs Strike", ["Delta", "Gamma", "Theta", "Vega", "Rho"])
        fig_chain = plot_option_chain(chain_with_greeks, greek)
        st.plotly_chart(fig_chain, use_container_width=True)

        if st.checkbox("Overlay Theoretical Model"):
            K_vals = chain_with_greeks["strike"].values
            T_vals = chain_with_greeks["ttm"].values
            σ_vals = chain_with_greeks["iv"].values
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
    st.markdown("📈 Build a strategy and simulate its Greek exposures and PnL under changing market scenarios.")

    strategy_input = st.text_area("Enter strategy (JSON list)", 
        '[{"type": "call", "strike": 150, "position": 1, "expiry": 0.5}]')

    try:
        import json
        legs = json.loads(strategy_input)
        s_grid = np.linspace(50, 300, 30)
        vol_grid = np.linspace(0.05, 1.0, 30)
        S_mesh, V_mesh = np.meshgrid(s_grid, vol_grid)
        df_sim = simulate_strategy(legs, S_mesh, V_mesh, r, q)

        st.plotly_chart(plot_strategy_exposure(df_sim), use_container_width=True)
        st.plotly_chart(plot_pnl_surface(df_sim["pnl"].values.reshape(V_mesh.shape), s_grid, vol_grid), use_container_width=True)
    except Exception as e:
        st.error(f"Error parsing strategy input: {e}")

# =======================
# Tab 4: Trade Scanner
# =======================
with tabs[3]:
    st.header("Real-Time Trade Scanner")
    st.markdown("🔍 Sort by Vega/Gamma/Theta to identify top-risk contracts.")

    if "chain_with_greeks" in locals():
        metric = st.selectbox("Sort By", ["vega", "gamma", "theta"])
        min_iv = st.slider("Min IV", 0.0, 1.0, 0.1)
        min_vol = st.slider("Min Volume", 0, 5000, 100)

        filtered = chain_with_greeks[
            (chain_with_greeks["iv"] >= min_iv) &
            (chain_with_greeks["volume"] >= min_vol)
        ].sort_values(by=metric, ascending=False).head(15)

        plot_trade_scanner_table(filtered)
    else:
        st.warning("⚠️ Load option chain in Tab 2 before scanning.")

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

