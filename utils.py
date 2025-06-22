# This file will contain methods for market data fetching, preprocessing and implied volatility (IV) cleaning, and strategy simulator
# helper functions.

import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Dict
from greeks_calculator import price, delta, gamma, vega, theta, rho
import streamlit as st
from datetime import datetime, timezone

def fetch_chain(ticker: str, expiry: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)
    chain = ticker_obj.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    calls["option_type"] = "call"
    puts["option_type"] = "put"

    df = pd.concat([calls, puts], ignore_index=True)
    df["expiry"] = expiry
    df["ticker"] = ticker
    return df

def calculate_strategy_payoff(
    legs: list,
    spot: float,
    r: float,
    underlying_qty: int | float = 0,
    iv_shift: float = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute mark-to-market P&L for a multi-leg option strategy plus an
    optional underlying-stock position, under a shifted implied-volatility
    scenario.

    Parameters
    ----------
    legs : list[dict]
        Each leg has keys: type, action, strike, qty, price (premium paid),
        expiry (string).
    spot : float
        Current underlying price.
    r : float
        Risk-free rate (kept for future Black-Scholes upgrade; unused here).
    underlying_qty : int | float, default 0
        +n for long shares, –n for short shares.
    iv_shift : float, default 0
        Percent change in IV (-50 … +50).  We proxy this by scaling premiums:
        new_premium = price * (1 + iv_shift/100).

    Returns
    -------
    payoff_df : pd.DataFrame
        "Spot" vs "Net P&L" over a 0.5×…1.5× spot sweep.
    metrics : dict
        Max profit, max loss, break-evens, risk-reward.
    """
    import numpy as np
    import pandas as pd

    # Sweep underlying price ±50 %
    s_range = np.linspace(spot * 0.5, spot * 1.5, 200)

    # ---------- Option legs ----------
    total_payoff = np.zeros_like(s_range)

    for leg in legs:
        K       = leg["strike"]
        qty     = leg["qty"]
        premium = leg["price"] * (1 + iv_shift / 100)   # IV shift proxy
        opt_t   = leg["type"].lower()
        action  = leg["action"].lower()

        intrinsic = np.where(
            opt_t == "call",
            np.maximum(s_range - K, 0),
            np.maximum(K - s_range, 0),
        )

        leg_payoff = intrinsic - premium      # long-leg P&L
        if action == "sell":
            leg_payoff *= -1                  # flip sign for short

        total_payoff += leg_payoff * qty

    # ---------- Underlying stock position ----------
    if underlying_qty != 0:
        total_payoff += underlying_qty * (s_range - spot)

    # ---------- Package results ----------
    payoff_df = pd.DataFrame({"Spot": s_range, "Net P&L": total_payoff})

    max_profit   = float(np.max(total_payoff))
    max_loss     = float(np.min(total_payoff))
    breakevens   = s_range[np.isclose(total_payoff, 0, atol=0.25)]

    metrics = {
        "Max Profit": round(max_profit, 2),
        "Max Loss": round(max_loss, 2),
        "Break-even Points": [round(b, 2) for b in breakevens],
        "Risk-Reward Ratio": round(abs(max_profit / max_loss), 2) if max_loss < 0 else "N/A",
    }

    return payoff_df, metrics


# -------------------------------------------
# caching wrappers
# -------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_option_chain(ticker: str, window: str) -> pd.DataFrame:
    """
    Fetch option chain for `ticker` and return a tidy DataFrame.
    `window` ∈ {"Next weekly", "Next monthly", "All ≤ 45 DTE"}.
    """
    tk = yf.Ticker(ticker)
    today = datetime.now(timezone.utc).date()

    def _dte(exp_str: str) -> int:
        return (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days

    # pick expiries
    all_exps = tk.options
    if window == "All ≤ 45 DTE":
        expiries = [e for e in all_exps if _dte(e) <= 45]
    elif window == "Next weekly":
        expiries = [all_exps[0]] if all_exps else []
    else:  # next monthly = 3rd Friday heuristic → pick first ≥ 21 dte
        expiries = []
        for e in all_exps:
            if _dte(e) >= 21:
                expiries.append(e)
                break

    frames = []
    for exp in expiries:
        oc = tk.option_chain(exp)
        for df, typ in [(oc.calls, "call"), (oc.puts, "put")]:
            dfe = (
                df.rename(columns=str.lower)
                .assign(expiry=exp, type=typ)
                .rename(columns={"impliedvolatility": "iv"})
            )
            dfe["iv"] = pd.to_numeric(dfe["iv"], errors="coerce")
            frames.append(dfe)

    if not frames:
        return pd.DataFrame()

    chain = pd.concat(frames, ignore_index=True)
    chain["dte"] = chain["expiry"].apply(lambda x: _dte(str(x)))
    # keep minimal cols
    keep = [
        "strike",
        "type",
        "dte",
        "iv",
        "lastprice",
        "bid",
        "ask",
        "volume",
        "openinterest",
    ]
    return chain[keep].rename(
        columns={
            "lastprice": "last",
            "openinterest": "oi",
        }
    )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_hist_prices(ticker: str, lookback: int = 60) -> pd.Series:
    """
    Download `lookback` trading days of historical prices; return close series.
    """
    df = yf.download(ticker, period=f"{lookback}d", progress=False, auto_adjust=True)

    # yfinance with auto_adjust=True gives only "Close"
    if "Close" in df.columns:
        price_ser = df["Close"]
    # fallback for older behaviour
    elif "Adj Close" in df.columns:
        price_ser = df["Adj Close"]
    else:
        raise ValueError("Price data missing Close/Adj Close columns")

    return price_ser


def calc_hv(price_ser: pd.Series, window: int = 20) -> float:
    """
    Annualised historical (realised) volatility using daily log-returns.
    """
    price_ser = price_ser.dropna()             # remove NaNs

    # need at least (window + 1) prices to compute `window` returns
    if len(price_ser) < window + 1:
        return np.nan

    log_r = np.log(price_ser / price_ser.shift()).dropna()
    hv_series = log_r.rolling(window).std(ddof=0).dropna()

    if hv_series.empty:
        return np.nan

    hv_scalar = hv_series.iloc[-1] * np.sqrt(252)   # annualise
    return float(hv_scalar)


def compute_skew(chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add column 'skew_z': z-score of IV within each expiry/type bucket.
    """
    if chain_df.empty:
        return chain_df

    def _z(grp):
        return (grp - grp.mean()) / grp.std(ddof=0)

    chain_df["skew_z"] = chain_df.groupby(["dte", "type"])["iv"].transform(_z)
    chain_df["abs_skew_z"] = chain_df["skew_z"].abs()
    return chain_df


def tag_signals(chain_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply thresholds & return only flagged rows.
    cfg keys: skew_z, iv_hv, vol_mult, hv
    """
    if chain_df.empty:
        return chain_df

    # simple placeholder for volume multiple: vol / (OI/10 + 1)
    chain_df["vol_mult"] = chain_df["volume"] / (chain_df["oi"] / 10 + 1)

    chain_df["iv_hv_ratio"] = chain_df["iv"] / cfg["hv"]

    def _flag(row):
        flags = []
        if row["abs_skew_z"] >= cfg["skew_z"]:
            flags.append("SK")
        if row["iv_hv_ratio"] >= cfg["iv_hv"]:
            flags.append("IV>HV")
        if row["vol_mult"] >= cfg["vol_mult"]:
            flags.append("VOLx")
        return ",".join(flags)

    chain_df["Flags"] = chain_df.apply(_flag, axis=1)
    chain_df = chain_df[chain_df["Flags"] != ""]  # keep only flagged rows
    return chain_df.reset_index(drop=True)