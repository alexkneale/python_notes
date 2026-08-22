# ============================================================================
# QUANT options_pricing: BLACK-SCHOLES PDE, GREEKS, AND MONTE CARLO
# ============================================================================
# Target Audience: Quant Researchers, Derivatives Traders, Quant Devs.
# Topic: Black-Scholes Analytical Option Pricing, Greek Derivations, 
#        Geometric Brownian Motion, and Antithetic Variance Reduction.
# Compatibility: Python 3.10+, NumPy 1.22+, SciPy 1.8+
# ============================================================================

import numpy as np
import scipy.stats as si
import time

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is an Option?
   - An option is a financial derivative contract giving the buyer the right 
     (but not the obligation) to buy (Call option) or sell (Put option) an underlying 
     asset (like a stock) at a set price (Strike Price, K) on or before a certain 
     date (Maturity, T).
2. What is the Black-Scholes Model?
   - Black-Scholes is a mathematical model that values European-style options. 
     It assumes the stock price follows a Geometric Brownian Motion (GBM) with 
     constant drift and volatility, and that there are no arbitrage opportunities.
3. What are Option Greeks?
   - Greeks measure how sensitive an option's price is to changes in parameters:
     - Delta: Sensitivity to stock price.
     - Gamma: Sensitivity of Delta to stock price (curvature).
     - Vega: Sensitivity to volatility.
     - Theta: Sensitivity to time decay.
4. What is Monte Carlo Option Pricing?
   - Instead of using analytical formulas, Monte Carlo simulates thousands of potential 
     future price paths for the stock under random walk conditions. We calculate the 
     option's payout at maturity for each path, average them, and discount this average 
     back to the present day using the risk-free rate.
"""


# ============================================================================
# 1. ANALYTICAL BLACK-SCHOLES-MERTON ENGINE
# ============================================================================
"""
The price of a non-dividend paying European Call option is:
    C(S, t) = S * N(d1) - K * e^(-r*T) * N(d2)
Where:
    d1 = (ln(S / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N(x) is the cumulative standard normal distribution CDF.

Option Greeks:
- Delta (dC/dS): Measures directional risk.
    Delta_Call = N(d1)
    Delta_Put = N(d1) - 1
- Gamma (d^2C/dS^2): Measures curvature of Delta.
    Gamma = N'(d1) / (S * sigma * sqrt(T))  where N'(x) is the normal PDF.
- Vega (dC/d_sigma): Sensitivity to volatility.
    Vega = S * N'(d1) * sqrt(T)
- Theta (dC/dT): Rate of time decay (negative value).
"""

class BlackScholesEngine:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = S          # Spot Price of underlying
        self.K = K          # Strike Price
        self.T = T          # Time to Maturity (in years)
        self.r = r          # Risk-free interest rate
        self.sigma = sigma  # Volatility of underlying (annualized)
        
        # Calculate d1 and d2 parameters
        self.d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def price(self, option_type: str = "call") -> float:
        if option_type.lower() == "call":
            return self.S * si.norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2)
        elif option_type.lower() == "put":
            return self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2) - self.S * si.norm.cdf(-self.d1)
        raise ValueError("Invalid option_type. Must be 'call' or 'put'")

    def greeks(self) -> dict:
        # Calculate normal PDF of d1
        pdf_d1 = si.norm.pdf(self.d1)
        
        gamma = pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * pdf_d1 * np.sqrt(self.T)
        
        # Call Greeks
        delta_call = si.norm.cdf(self.d1)
        theta_call = - (self.S * pdf_d1 * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2)
        
        # Put Greeks
        delta_put = delta_call - 1.0
        theta_put = - (self.S * pdf_d1 * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2)
        
        return {
            "call_delta": delta_call,
            "put_delta": delta_put,
            "gamma": gamma,
            "vega": vega,
            "call_theta": theta_call,
            "put_theta": theta_put
        }


# ============================================================================
# 2. MONTE CARLO SIMULATION WITH ANTITHETIC VARIATE REDUCTION
# ============================================================================
"""
Geometric Brownian Motion SDE:
    dS_t = r * S_t * dt + sigma * S_t * dW_t
Discretized path step:
    S_T = S_0 * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z)
Where Z is a standard normal random variable.

Antithetic Variates (Variance Reduction):
For each normal random draw Z, we also evaluate the exact negative path using -Z. 
This guarantees perfectly symmetric draws, eliminating odd-moment sample biases, 
reducing standard error of our Monte Carlo estimation by up to 50% for the same 
number of path samples.
"""

def monte_carlo_european_price(S: float, K: float, T: float, r: float, sigma: float, 
                               num_paths: int = 100000, option_type: str = "call", 
                               use_antithetic: bool = True) -> tuple[float, float]:
    """
    Returns: (Estimated option price, Standard Error of estimation)
    """
    if use_antithetic:
        # Draw half the samples, then mirror them
        half_paths = num_paths // 2
        Z = np.random.standard_normal(half_paths)
        Z = np.concatenate([Z, -Z])
    else:
        Z = np.random.standard_normal(num_paths)
        
    # Simulate final asset price S_T
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)
    
    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("Invalid option type")
        
    # Discount payoffs back to present day
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    price_estimate = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs) / np.sqrt(num_paths)
    
    return price_estimate, standard_error


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What does Delta measure, and what does a Call option Delta of 0.5 mean?
   - Answer: Delta measures the rate of change of the option price with respect to 
     changes in the underlying asset's price (dC/dS). A Delta of 0.5 means that if the 
     stock price increases by $1.00, the option price is expected to increase by approximately 
     $0.50. It also represents a first-order risk metric and acts as a replication ratio 
     for delta-hedged portfolios.

2. MEDIUM: Explain Put-Call Parity. Write its formula and state its assumptions.
   - Answer: Put-Call Parity dictates a static replication relationship between Call and Put 
     options of the same underlying, strike, and maturity. 
     Formula: C + K * e^(-r*T) = P + S
     Assumptions: European style exercise, constant interest rates, liquid friction-free markets, 
     and no dividends paid during the option's lifetime. If violated, an arbitrageur can execute 
     long-short strategies to capture risk-free profit.

3. HARD: Derive or explain the Gamma Greek. What does its magnitude signify about 
   a delta-hedged trading book?
   - Answer: Gamma is the second-order derivative of the option price relative to stock price 
     (d^2C/dS^2). It measures the rate of change of Delta. For a delta-hedged book, 
     having a high positive Gamma means your delta-hedge becomes increasingly long as the stock rises 
     (and short as it falls), making rebalancing profitable (buy low, sell high). 
     Conversely, negative Gamma means rebalancing requires selling stock as it rises and buying 
     as it falls, incurring losses.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `calculate_implied_volatility(target_price, S, K, T, r, option_type)` 
   that solves for option Implied Volatility (IV) using the Newton-Raphson numerical root-finding 
   algorithm.
   - Equation: sigma_{n+1} = sigma_n - (BS_Price(sigma_n) - target_price) / Vega(sigma_n)
   - Code Template to execute:
"""

def calculate_implied_volatility(target_price: float, S: float, K: float, T: float, r: float, 
                                 option_type: str = "call", max_iter: int = 100, tolerance: float = 1e-6) -> float:
    # 1. Initialize guess using standard heuristic (e.g., 20% annualized volatility)
    sigma_guess = 0.20
    
    for i in range(max_iter):
        bs = BlackScholesEngine(S, K, T, r, sigma_guess)
        current_price = bs.price(option_type)
        greeks = bs.greeks()
        # Vega is identical for calls and puts (dC/d_sigma = dP/d_sigma)
        vega = greeks["vega"]
        
        # Avoid division by zero
        if abs(vega) < 1e-8:
            break
            
        difference = current_price - target_price
        
        # Stop if we are within acceptable numerical convergence boundaries
        if abs(difference) < tolerance:
            return sigma_guess
            
        # Newton-Raphson update step: x_next = x - f(x)/f'(x)
        sigma_guess -= difference / vega
        
    return sigma_guess


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING OPTIONS PRICING ENGINE & MONTE CARLO")
    print("="*80)

    # 1. Setup parameters
    S0 = 100.0   # Current stock price
    K = 105.0    # Strike price
    T = 1.0      # Time to maturity (1 year)
    r = 0.05     # 5% risk-free rate
    sigma = 0.20 # 20% annualized volatility

    # 2. Analytical Black-Scholes Evaluation
    bs = BlackScholesEngine(S0, K, T, r, sigma)
    call_price = bs.price("call")
    put_price = bs.price("put")
    greeks = bs.greeks()

    print(f"Analytical European Prices:")
    print(f"  Call Price: ${call_price:.4f} (Expected put parity value: ${call_price + K * np.exp(-r*T) - S0:.4f})")
    print(f"  Put Price:  ${put_price:.4f}")
    print(f"\nGreeks Sensitivity:")
    print(f"  Call Delta: {greeks['call_delta']:.4f}")
    print(f"  Put Delta:  {greeks['put_delta']:.4f}")
    print(f"  Gamma:      {greeks['gamma']:.4f}")
    print(f"  Vega:       {greeks['vega']:.4f}")

    # 3. Monte Carlo Comparison (with and without Antithetic Variate Reduction)
    print("\n" + "-"*50)
    print("MONTE CARLO SIMULATION COMPARISON")
    print("-"*50)
    
    t0 = time.perf_counter()
    mc_price_anti, mc_se_anti = monte_carlo_european_price(S0, K, T, r, sigma, num_paths=100000, use_antithetic=True)
    t_anti = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_price_raw, mc_se_raw = monte_carlo_european_price(S0, K, T, r, sigma, num_paths=100000, use_antithetic=False)
    t_raw = time.perf_counter() - t0

    print(f"Antithetic MC Call Price:  ${mc_price_anti:.4f} | Std Error: {mc_se_anti:.6f} | Time: {t_anti:.4f}s")
    print(f"Standard MC Call Price:    ${mc_price_raw:.4f}  | Std Error: {mc_se_raw:.6f}  | Time: {t_raw:.4f}s")
    print(f"Variance Reduction Factor: {mc_se_raw / mc_se_anti:.2f}x lower standard error!")

    # 4. Implied Volatility Newton-Raphson Test
    print("\n" + "-"*50)
    print("NEWTON-RAPHSON IMPLIED VOLATILITY CALCULATION")
    print("-"*50)
    
    # We want to solve for the volatility that yields the exact call price we got above
    solved_iv = calculate_implied_volatility(call_price, S0, K, T, r, option_type="call")
    print(f"Target Call Price: ${call_price:.4f}")
    print(f"Calculated IV:     {solved_iv * 100:.4f}% (Expected: {sigma * 100:.2f}%)")
