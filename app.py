# This web app is designed to allow users to explore how the Greeks (i.e., Delta, Gamma, Theta, Vega, and Rho) affect option pricing under the 
# Black-Scholes-Merton (BSM) model by adjusting input parameters (ex: strike price, time to maturity, volatility, interest rate, etc.) using sliders.
# The web app was designed using Streamlit and plotly.

import streamlit as st
import numpy as np
from greeks_calculator import price, delta, gamma, vega, theta, rho
from utils import fetch_chain, calculate_strategy_payoff
from plot_helpers import (
    plot_greeks_surface, plot_strategy_payoff
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
    - A Delta of 0.60 means the call option gains 60 cents if the stock rises 1 dollar.

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
    st.header("Real Options Chain Analyzer")

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

# =======================
# Tab 3: Strategy Simulator
# =======================
with tabs[2]:
    st.header("Options Strategy Simulator")

    st.markdown(
        "Build a custom multi-leg options strategy, add an underlying stock hedge, "
        "stress-test implied volatility, or start from a predefined template."
    )

    # -------------------------- basic inputs --------------------------
    ticker_sim = st.text_input("Ticker Symbol", value="AAPL")
    try:
        ticker_obj_sim = yf.Ticker(ticker_sim)
        expiries_sim = ticker_obj_sim.options
    except Exception:
        st.error("Could not load option expirations.")
        st.stop()

    expiry_sim = st.selectbox("Select Expiration", expiries_sim)

    # --------------------- 1️⃣ Strategy Templates ----------------------
    strategy_templates = {
        "Bull Call Spread": [
            {"type": "call", "action": "buy",  "strike": 100, "qty": 1, "price": 5, "expiry": expiry_sim},
            {"type": "call", "action": "sell", "strike": 110, "qty": 1, "price": 2, "expiry": expiry_sim},
        ],
        "Long Straddle": [
            {"type": "call", "action": "buy", "strike": 100, "qty": 1, "price": 4, "expiry": expiry_sim},
            {"type": "put",  "action": "buy", "strike": 100, "qty": 1, "price": 5, "expiry": expiry_sim},
        ],
        "Iron Condor": [
            {"type": "call", "action": "sell", "strike": 110, "qty": 1, "price": 2, "expiry": expiry_sim},
            {"type": "call", "action": "buy",  "strike": 115, "qty": 1, "price": 1, "expiry": expiry_sim},
            {"type": "put",  "action": "sell", "strike":  90, "qty": 1, "price": 2, "expiry": expiry_sim},
            {"type": "put",  "action": "buy",  "strike":  85, "qty": 1, "price": 1, "expiry": expiry_sim},
        ],
    }

    template_choice = st.selectbox(
        "Choose Strategy Template",
        ["None"] + list(strategy_templates.keys()),
        index=0,
        help="Auto-populate common multi-leg strategies."
    )

    # Initialise session state
    if "legs" not in st.session_state:
        st.session_state.legs = []

    # Apply template
    if template_choice != "None" and st.button("Load Template"):
        # overwrite current legs with a deep copy of template legs
        import copy
        st.session_state.legs = copy.deepcopy(strategy_templates[template_choice])

    st.subheader("Add / Edit Option Legs")

    # -------------------- manual leg entry form ----------------------
    with st.form("leg_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            opt_type = st.selectbox("Option Type", ["Call", "Put"])
        with col2:
            action = st.selectbox("Action", ["Buy", "Sell"])
        with col3:
            qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        strike = st.number_input("Strike Price", min_value=0.0, value=100.0, step=1.0)
        price = st.number_input("Premium (optional)", min_value=0.0, value=0.0, step=0.01)

        if st.form_submit_button("Add Leg"):
            st.session_state.legs.append(
                {
                    "type": opt_type.lower(),
                    "action": action.lower(),
                    "strike": strike,
                    "qty": qty,
                    "price": price,
                    "expiry": expiry_sim,
                }
            )

    # ---------------------------- payoff inputs -----------------------------
    if st.session_state.legs:
        st.subheader("Current Strategy Legs")
        st.dataframe(pd.DataFrame(st.session_state.legs))

        spot_price = st.number_input(
            "Current Spot Price", min_value=0.0, value=100.0, step=0.5
        )

        # 2️⃣ Underlying hedge
        underlying_qty = st.number_input(
            "Shares of Underlying ( + long / – short )",
            value=0,
            step=1,
            help="Add stock to hedge delta or create covered positions.",
        )

        # 3️⃣ IV stress-test
        iv_shift = st.slider(
            "Implied Volatility Shift (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=1,
            help="Re-price each option leg with shifted IV before payoff calc.",
        )

        # ------------------- calculate and plot strategy --------------------
        payoff_df, metrics = calculate_strategy_payoff(
            st.session_state.legs,
            spot_price,
            r,                 # risk-free
            underlying_qty,
            iv_shift
        )

        fig = plot_strategy_payoff(payoff_df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Metrics")
        st.write(metrics)

    if st.button("Reset Strategy"):
        st.session_state.legs = []