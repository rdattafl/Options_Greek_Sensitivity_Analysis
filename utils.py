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

# --- utils.py ---

def calculate_strategy_payoff(legs: list, spot: float, r: float) -> tuple[pd.DataFrame, dict]:
    import numpy as np
    s_range = np.linspace(spot * 0.5, spot * 1.5, 200)
    total_payoff = np.zeros_like(s_range)

    for leg in legs:
        K = leg["strike"]
        qty = leg["qty"]
        premium = leg["price"]
        opt_type = leg["type"]
        action = leg["action"]

        intrinsic = np.where(
            opt_type == "call",
            np.maximum(s_range - K, 0),
            np.maximum(K - s_range, 0)
        )

        leg_payoff = intrinsic - premium
        if action == "sell":
            leg_payoff *= -1

        total_payoff += leg_payoff * qty

    payoff_df = pd.DataFrame({"Spot": s_range, "Net P&L": total_payoff})

    max_profit = np.max(total_payoff)
    max_loss = np.min(total_payoff)
    breakeven_mask = np.isclose(total_payoff, 0, atol=0.25)
    breakevens = s_range[breakeven_mask]

    metrics = {
        "Max Profit": round(max_profit, 2),
        "Max Loss": round(max_loss, 2),
        "Break-even Points": [round(b, 2) for b in breakevens],
        "Risk-Reward Ratio": round(abs(max_profit / max_loss), 2) if max_loss < 0 else "N/A"
    }

    return payoff_df, metrics

