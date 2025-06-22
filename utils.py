# This file will contain methods for market data fetching, preprocessing and implied volatility (IV) cleaning, and strategy simulator
# helper functions.

import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Dict
from greeks_calculator import price, delta, gamma, vega, theta, rho

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