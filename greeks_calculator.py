# This file will contain the methods used to compute the core Greek functions themselves (i.e., Delta, Gamma, Vega, and Theta.)

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, sigma, r, q):
    """Helper to compute d1 and d2 terms."""
    S, K, T, sigma = map(np.array, (S, K, T, sigma))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2, sqrtT


def price(S, K, T, sigma, r, q, option_type="call"):
    """
    Compute Black-Scholes-Merton option price.
    
    Parameters:
    - S: Spot price
    - K: Strike price
    - T: Time to maturity (in years)
    - sigma: Volatility
    - r: Risk-free rate
    - q: Dividend yield
    - option_type: "call" or "put"
    """
    d1, d2, _ = _d1_d2(S, K, T, sigma, r, q)
    S, K, T = map(np.array, (S, K, T))
    if option_type.lower() == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, sigma, r, q, option_type="call"):
    """
    Compute Delta of a call or put option.
    """
    d1, _, _ = _d1_d2(S, K, T, sigma, r, q)
    T = np.array(T)
    if option_type.lower() == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, sigma, r, q):
    """
    Compute Gamma of an option.
    """
    d1, _, sqrtT = _d1_d2(S, K, T, sigma, r, q)
    S, sigma, T = map(np.array, (S, sigma, T))
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * sqrtT + 1e-9)


def vega(S, K, T, sigma, r, q):
    """
    Compute Vega of an option.
    """
    d1, _, sqrtT = _d1_d2(S, K, T, sigma, r, q)
    S, T = map(np.array, (S, T))
    return S * np.exp(-q * T) * sqrtT * norm.pdf(d1)


def theta(S, K, T, sigma, r, q, option_type="call"):
    """
    Compute Theta of a call or put option (per year).
    """
    d1, d2, sqrtT = _d1_d2(S, K, T, sigma, r, q)
    S, K, T, sigma = map(np.array, (S, K, T, sigma))

    term1 = - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * sqrtT)
    if option_type.lower() == "call":
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2 + term3


def rho(S, K, T, sigma, r, q, option_type="call"):
    """
    Compute Rho of a call or put option.
    """
    _, d2, _ = _d1_d2(S, K, T, sigma, r, q)
    K, T = map(np.array, (K, T))
    if option_type.lower() == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
