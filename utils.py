# This file will contain methods for market data fetching, preprocessing and implied volatility (IV) cleaning, and strategy simulator
# helper functions.

import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Dict
from greeks_calculator import price, delta, gamma, vega, theta, rho

def fetch_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Fetch the option chain for a given ticker and expiry."""
    ticker_obj = yf.Ticker(ticker)
    chain = ticker_obj.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    calls['option_type'] = 'call'
    puts['option_type'] = 'put'
    
    df = pd.concat([calls, puts], axis=0)
    df['expiry'] = expiry
    df['ticker'] = ticker
    df.reset_index(drop=True, inplace=True)
    return df

def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw option chain data and filter for usable rows."""
    df = df.copy()

    # Drop rows with missing crucial values
    df = df.dropna(subset=['strike', 'lastPrice', 'impliedVolatility', 'inTheMoney'])

    # Convert to numeric
    df['strike'] = pd.to_numeric(df['strike'])
    df['lastPrice'] = pd.to_numeric(df['lastPrice'])
    df['impliedVolatility'] = pd.to_numeric(df['impliedVolatility'])

    # Estimate time to maturity in years
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['days_to_expiry'] = (df['expiry'] - pd.Timestamp.today()).dt.days.clip(lower=1)
    df['T'] = df['days_to_expiry'] / 365.0

    # Remove very deep ITM/OTM (keep within +/- 30% moneyness)
    spot = df['lastPrice'].median() * 1.25  # crude spot proxy if not provided
    df = df[(df['strike'] > 0.7 * spot) & (df['strike'] < 1.3 * spot)]

    return df

def add_greeks_to_chain(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """Add Greek columns using Black-Scholes closed-form formulas."""
    df = df.copy()
    spot = df['lastPrice'].median() * 1.25  # again crude proxy if actual spot not passed

    S = np.full(len(df), spot)
    K = df['strike'].values
    T = df['T'].values
    sigma = df['impliedVolatility'].values
    opt_type = df['option_type'].values

    df['delta'] = [delta(S[i], K[i], T[i], sigma[i], r, q, opt_type[i]) for i in range(len(df))]
    df['gamma'] = [gamma(S[i], K[i], T[i], sigma[i], r, q) for i in range(len(df))]
    df['vega'] = [vega(S[i], K[i], T[i], sigma[i], r, q) for i in range(len(df))]
    df['theta'] = [theta(S[i], K[i], T[i], sigma[i], r, q, opt_type[i]) for i in range(len(df))]
    df['rho'] = [rho(S[i], K[i], T[i], sigma[i], r, q, opt_type[i]) for i in range(len(df))]

    return df

def simulate_strategy(
    legs: List[Dict],
    spot_paths: np.ndarray,
    vol_paths: np.ndarray,
    r: float,
    q: float
) -> pd.DataFrame:
    """
    Simulate the strategy over time across stochastic paths.
    Each leg is a dict: {'type': 'call'/'put', 'strike': K, 'position': n, 'expiry': T}
    Returns a DataFrame of cumulative Greeks and PnL per timestep.
    """
    assert spot_paths.shape == vol_paths.shape, "Shape mismatch in paths."

    time_steps = spot_paths.shape[1]
    results = {
        'delta': np.zeros(time_steps),
        'gamma': np.zeros(time_steps),
        'vega': np.zeros(time_steps),
        'theta': np.zeros(time_steps),
        'rho': np.zeros(time_steps),
        'pnl': np.zeros(time_steps)
    }

    for leg in legs:
        K = leg['strike']
        T = leg['expiry']
        n = leg['position']
        opt_type = leg['type']

        for t in range(time_steps):
            S_t = spot_paths[:, t]
            sigma_t = vol_paths[:, t]
            T_t = max(T - t / 252, 1e-6)

            p = price(S_t, K, T_t, sigma_t, r, q, opt_type)
            d = delta(S_t, K, T_t, sigma_t, r, q, opt_type)
            g = gamma(S_t, K, T_t, sigma_t, r, q)
            v = vega(S_t, K, T_t, sigma_t, r, q)
            th = theta(S_t, K, T_t, sigma_t, r, q, opt_type)
            rh = rho(S_t, K, T_t, sigma_t, r, q, opt_type)

            results['pnl'][t] += n * np.mean(p)
            results['delta'][t] += n * np.mean(d)
            results['gamma'][t] += n * np.mean(g)
            results['vega'][t] += n * np.mean(v)
            results['theta'][t] += n * np.mean(th)
            results['rho'][t] += n * np.mean(rh)

    return pd.DataFrame(results)
