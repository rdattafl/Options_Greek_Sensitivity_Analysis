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
import pandas as pd
import plotly.express as px
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

    # Inline interpretation
    greek_descriptions = {
        "Delta": """
    **Delta** measures **how much the option price moves when the underlying asset (S) changes by $1**.

    - For calls, Delta is positive (0 to 1); for puts, it’s negative (-1 to 0).
    - A Delta of 0.60 means the call option gains $0.60 if the stock rises $1.

    **It increases with:**
    - Higher spot price (S): you're more in-the-money, so the option acts more like the stock.
    - Shorter time to expiration (T): Delta tends to 1 or 0 as expiry nears.

    **It decreases with:**
    - Deeper out-of-the-money options or longer-dated options (higher T): less sensitive to small moves in S.
        """,

        "Gamma": """
    **Gamma** measures **how quickly Delta changes as the underlying asset (S) moves** — it’s the "Delta of Delta".

    - High Gamma means Delta shifts fast — important for hedging.
    - Gamma peaks **when the option is at-the-money**.

    **It increases with:**
    - Shorter time to maturity (T): gamma spikes near expiration.
    - ATM options: Gamma is max when S ≈ K.

    **It decreases with:**
    - Deep ITM or OTM options: their Deltas are already near 1 or 0.
    - Longer expiries: Delta changes more gradually.

    **Why it matters:** Traders with Delta-hedged portfolios still face risk from large Gamma — i.e., needing to rebalance Delta frequently as prices move.
        """,

        "Vega": """
    **Vega** measures **how sensitive the option price is to changes in implied volatility (σ)**.

    - A Vega of 0.10 means the option price changes by $0.10 for a 1% change in volatility.

    **It increases with:**
    - Longer time to maturity (T): more time = more exposure to volatility.
    - ATM options: they gain most from volatility increases.

    **It decreases with:**
    - Deep ITM/OTM options: their payouts are less affected by changes in volatility.
    - As T → 0: less time to benefit from volatility.

    **Why it matters:** When you buy options, you’re long Vega — higher volatility helps. When you sell options, you want volatility to stay low.
        """,

        "Theta": """
    **Theta** measures **how much value an option loses per day as time passes** — known as **time decay**.

    - A Theta of -0.05 means the option loses $0.05 in value per day (if everything else stays the same).

    **It increases (more negative) with:**
    - ATM options: they lose time value fastest.
    - Shorter time to expiry (T): decay accelerates near expiration.

    **It decreases (less negative or near 0) with:**
    - Deep ITM or OTM options: less time value left to decay.
    - Longer-dated options: decay is slower initially.

    **Why it matters:** Option buyers fight Theta decay every day. Sellers profit from it if nothing moves.
        """,

        "Rho": """
    **Rho** measures **how much the option price changes with a 1% change in interest rates (r)**.

    - A Rho of 0.10 means the option price changes by $0.10 for a 1% rise in rates.

    **For calls:**
    - Rho is positive — rising rates increase call value.

    **For puts:**
    - Rho is negative — rising rates decrease put value.

    **It increases with:**
    - Longer time to maturity (T): more exposure to discounting effect.
    - Deep ITM options: their intrinsic value dominates.

    **It’s small/negligible when:**
    - T is short (approaching expiry).
    - Option is near the money or low premium.

    **Why it matters:** Most relevant during macro shifts (e.g., Fed policy). Often ignored but can affect long-dated options significantly.
        """
    }

    st.markdown(greek_descriptions[selected_greek])

# =======================
# Tab 2: Chain Analyzer
# =======================
with tabs[1]:
    st.header("📈 Real Options Chain Analyzer")

    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()

    if ticker:
        try:
            ticker_obj = yf.Ticker(ticker)
            valid_expiries = ticker_obj.options
        except Exception:
            st.error("Could not fetch options data for this ticker.")
            st.stop()

        if not valid_expiries:
            st.error("No available option expirations for this ticker.")
            st.stop()

        expiry = st.selectbox("Select Expiry Date", valid_expiries, index=0)

        raw_df = fetch_chain(ticker, expiry)

        if raw_df.empty:
            st.warning("Fetched option chain is empty.")
            st.stop()

        df = raw_df.copy()

        # Fetch spot price
        spot_data = ticker_obj.history(period="1d")
        spot = spot_data["Close"].iloc[-1] if not spot_data.empty else None
        if spot is None:
            st.error("Could not fetch current spot price.")
            st.stop()

        st.write(f"Current Spot Price: **${spot:.2f}**")

        # Process chain
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        df["spread_pct"] = df["spread"] / df["mid"].replace(0, np.nan)
        df["ITM"] = ((df["option_type"] == "call") & (df["strike"] < spot)) | \
                    ((df["option_type"] == "put") & (df["strike"] > spot))
        df["dist_from_spot(%)"] = 100 * (df["strike"] - spot) / spot
        df["break_even"] = np.where(df["option_type"] == "call",
                                    df["strike"] + df["mid"],
                                    df["strike"] - df["mid"])
        df["weird"] = (df["mid"] <= 0) | (df["bid"] > df["ask"]) | (df["spread_pct"] > 0.5)

        # Section 1: Liquid contracts
        st.subheader("Most Liquid Contracts")
        st.markdown("""
        These are the **top 10 contracts** with high trading **volume and open interest**, and relatively **tight bid-ask spreads**.
        These contracts are easier to enter and exit without slippage and are typically favored by active traders.
        """)
        liquid = df[(df["volume"] > 0) & (df["openInterest"] > 0)].copy()
        liquid = liquid.sort_values(by=["volume", "spread_pct"], ascending=[False, True])
        st.dataframe(liquid[["contractSymbol", "option_type", "strike", "volume", "openInterest", "bid", "ask", "spread_pct"]].head(10))

        # Section 2: IV Skew
        st.subheader("Implied Volatility Skew")
        st.markdown("""
        **Implied Volatility (IV)** reflects how much the market expects the stock to move.  
        A **steep skew** in IV across strikes often indicates **hedging demand** or **directional bias**.

        - **Call IV > Put IV** ➝ bullish demand or call overwriting.
        - **Put IV > Call IV** ➝ bearish protection or crash hedging.
        """)
        if "impliedVolatility" in df.columns and df["impliedVolatility"].notnull().any():
            fig_iv = px.line(df, x="strike", y="impliedVolatility", color="option_type",
                             title="Implied Volatility vs Strike", markers=True)
            st.plotly_chart(fig_iv, use_container_width=True)
        else:
            st.warning("No implied volatility data found for this chain.")

        # Section 3: ITM vs OTM Comparison
        st.subheader("In-The-Money vs Out-of-The-Money Contracts")
        st.markdown("""
        **In-the-money (ITM)** options have intrinsic value and behave more like stock, while **out-of-the-money (OTM)** options
        are pure premium and are often used for **speculation** or **hedging**.

        This section shows how far the break-even point is from the current stock price,
        which is critical when evaluating whether a trade makes sense.
        """)
        itm_df = df[df["ITM"]].sort_values(by="strike")
        otm_df = df[~df["ITM"]].sort_values(by="strike")
        st.markdown("**ITM Contracts:**")
        st.dataframe(itm_df[["contractSymbol", "option_type", "strike", "mid", "break_even", "dist_from_spot(%)"]].head(10))
        st.markdown("**OTM Contracts:**")
        st.dataframe(otm_df[["contractSymbol", "option_type", "strike", "mid", "break_even", "dist_from_spot(%)"]].head(10))

        # Section 4: Weird Contracts
        st.subheader("Potentially Mispriced or Illiquid Contracts")
        st.markdown("""
        This identifies contracts with unusual pricing behavior:
        - **Negative or 0 mid-price**
        - **Bid > Ask** (data error or stale quote)
        - **Very wide bid-ask spread** (> 50%)

        These contracts are often **illiquid** or **data anomalies**, and should be **avoided** in serious strategies.
        """)
        weird_df = df[df["weird"]].sort_values(by="spread_pct", ascending=False)
        st.dataframe(weird_df[["contractSymbol", "option_type", "strike", "bid", "ask", "mid", "volume", "spread_pct"]].head(10))
