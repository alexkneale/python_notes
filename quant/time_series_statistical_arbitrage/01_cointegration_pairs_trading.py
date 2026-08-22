# ============================================================================
# QUANT statistical_arbitrage: COINTEGRATION & ORNSTEIN-UHLENBECK SDE
# ============================================================================
# Target Audience: Quant Researchers, Algorithmic Traders, Signal Engineers.
# Topic: Correlation vs Cointegration, Engle-Granger Method, Ornstein-Uhlenbeck 
#        SDE Discretization, Half-life of Mean Reversion, and Signal Entry.
# Compatibility: Python 3.10+, NumPy 1.22+, Statsmodels 0.13+
# ============================================================================

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Statistical Arbitrage (Pairs Trading)?
   - Pairs trading is a strategy that exploits temporary pricing anomalies 
     between two historically related assets. If stock A and stock B are highly 
     linked, and stock A suddenly rises while B falls, we sell A (short) and buy B 
     (long), betting that they will eventually snap back to their historical relationship.
2. What is Correlation vs. Cointegration?
   - Correlation: Measures short-term directional co-movements. However, two highly correlated 
     assets can drift infinitely apart (e.g. two tech stocks with different long-term growth), 
     causing large losses.
   - Cointegration: Asserts a long-term equilibrium connection. Even if both stocks drift 
     all over the place, a specific linear combination of their prices (the spread) stays 
     stationary (meaning it fluctuates around a constant average and never drifts to infinity).
3. What is the Ornstein-Uhlenbeck (OU) Process?
   - The OU process is a mathematical model for a random variable that is pulled back 
     towards a long-term average over time. It is the standard SDE used to model the behavior 
     of cointegrated stock spreads.
4. What is the Half-life of Mean Reversion?
   - The half-life is the expected time it takes for the spread to pull back halfway to its 
     historical long-term average. It helps quant researchers decide how long a pairs trade 
     is likely to remain open.
"""


# ============================================================================
# 1. THE ENGLE-GRANGER TWO-STEP COINTEGRATION METHOD
# ============================================================================
"""
If two time series $Y_t$ and $X_t$ are integrated of order 1 ($I(1)$), meaning their 
first differences are stationary, they are cointegrated if there exists a hedge 
ratio $\beta$ such that the linear combination:
    \epsilon_t = Y_t - \beta X_t - \alpha
Is stationary ($I(0)$).

Engle-Granger Steps:
1. Run Ordinary Least Squares (OLS) regression: $Y_t = \alpha + \beta X_t + \epsilon_t$
   - This estimates the optimal hedge ratio $\beta$.
2. Run the Augmented Dickey-Fuller (ADF) unit-root test on the residual series $\epsilon_t$.
   - If the ADF test statistic is lower than the critical values (rejecting the null 
     hypothesis of a unit root), the spread is stationary, confirming cointegration!
"""

def perform_engle_granger_test(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Returns: (ADF test statistic, critical value 5%, Spread residuals)
    """
    # 1. Regress Y on X to find hedge ratio
    X_with_const = sm.add_constant(x)
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    alpha = results.params[0]
    beta = results.params[1]
    
    # 2. Calculate residual spread
    spread = y - beta * x - alpha
    
    # 3. Perform ADF test on residuals
    # autolag='AIC' automatically selects optimal lags for serial correlation adjustment
    adf_results = ts.adfuller(spread, maxlag=1, autolag=None)
    
    adf_stat = adf_results[0]
    crit_5pct = adf_results[4]['5%']
    
    return adf_stat, crit_5pct, spread


# ============================================================================
# 2. ORNSTEIN-UHLENBECK SDE DISCRETIZATION
# ============================================================================
"""
The continuous Ornstein-Uhlenbeck (OU) stochastic process is:
    dx_t = \theta (\mu - x_t) dt + \sigma dW_t
Where:
    \theta: Speed of mean reversion (\theta > 0).
    \mu: Long-term equilibrium mean.
    \sigma: Instantaneous volatility.
    dW_t: Standard Brownian Motion increments.

Discretized AR(1) representation:
To estimate parameters from daily historical spread series $x_t$, we discretize the SDE:
    x_t - x_{t-1} = \theta (\mu - x_{t-1}) \Delta t + \epsilon_t
Rearranging into a standard linear regression model $Y = a + b \cdot X + \epsilon$:
    x_t = a + b \cdot x_{t-1} + \epsilon_t
Where:
    a = \theta \mu \Delta t
    b = 1 - \theta \Delta t  (or more precisely, b = e^{-\theta \Delta t})
    
From the regression coefficients $a$ and $b$ (with $\Delta t = 1.0$ for daily steps):
    \theta = -\ln(b) / \Delta t
    \mu = a / (1 - b)
The Half-life of Mean Reversion is:
    \tau_{1/2} = \frac{\ln(2)}{\theta}
"""


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Why can't we use standard Pearson Correlation to safely run a pairs trade?
   - Answer: Correlation only measures short-term directional co-movements and does 
     not establish any long-term error-correction equilibrium. Two correlated stocks 
     can diverge permanently due to structural business shifts. Cointegration establishes 
     a mean-reverting relationship, guaranteeing that pricing spreads will pull back 
     to a constant average over time, preventing catastrophic divergence.

2. MEDIUM: What does a high mean-reversion speed ($\theta$) tell you about a trading 
   pair compared to a low $\theta$?
   - Answer: A high $\theta$ implies that price divergences are pulled back rapidly 
     to the equilibrium mean, leading to a short half-life of mean reversion. This is 
     optimal for traders because capital is recycled quickly. A low $\theta$ indicates 
     very slow mean-reversion, requiring trades to remain open for long periods, 
     incurring high capital carry costs and increasing the risk of structural regime shifts.

3. HARD: Explain what happens if you run an OLS regression between two completely unrelated, 
   trending random walks. What is this phenomenon called?
   - Answer: This triggers the "Spurious Regression" phenomenon. If you regress two 
     non-stationary, independent random walks, OLS will often calculate a highly statistically 
     significant $\beta$ and a high R-squared ($R^2$), falsely suggesting a strong relationship. 
     This occurs because both series contain a cumulative temporal trend (integration), 
     violating classic IID residuals assumptions. This is why we must test residuals 
     for stationarity (via ADF) to verify actual cointegration.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `estimate_ou_parameters(spread_series)` that fits an 
   Ornstein-Uhlenbeck model on a historical spread array, calculating the mean-reversion 
   speed ($\theta$), long-term mean ($\mu$), and the expected Half-Life of mean reversion.
   - Code Template to execute:
"""

def estimate_ou_parameters(spread_series: np.ndarray, dt: float = 1.0) -> dict:
    # 1. Construct lag arrays: Y = x_t, X = x_{t-1}
    y_reg = spread_series[1:]
    x_reg = spread_series[:-1]
    
    # 2. Run OLS: Y = a + b * X
    X_with_const = sm.add_constant(x_reg)
    model = sm.OLS(y_reg, X_with_const)
    results = model.fit()
    
    a = results.params[0]
    b = results.params[1]
    
    # 3. Extract theta, mu, and half-life
    # If b >= 1.0, the process is non-stationary (does not mean revert)
    if b >= 1.0:
        return {"theta": 0.0, "mu": 0.0, "half_life": float('inf')}
        
    theta = -np.log(b) / dt
    mu = a / (1.0 - b)
    half_life = np.log(2.0) / theta
    
    return {
        "theta": theta,
        "mu": mu,
        "half_life": half_life
    }


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING STATISTICAL ARBITRAGE & MEAN REVERSION")
    print("="*80)

    # 1. Simulate a synthetic Cointegrated Pair
    # We generate Stock X as a random walk
    np.random.seed(42)
    n_steps = 500
    X_walk = np.cumsum(np.random.normal(0.0, 1.0, n_steps)) + 100.0
    
    # We generate a stationary OU Spread process (representing the relationship)
    # dx = theta*(mu - x)*dt + sigma*dW
    theta_true = 0.10
    mu_true = 2.0
    sigma_spread = 0.5
    
    spread_sim = np.zeros(n_steps)
    spread_sim[0] = mu_true
    for t in range(1, n_steps):
        # Discretized step
        noise = np.random.normal(0, sigma_spread)
        spread_sim[t] = spread_sim[t-1] + theta_true * (mu_true - spread_sim[t-1]) + noise
        
    # We construct Stock Y using the cointegrating equation: Y = 1.5 * X + Spread
    # This guarantees cointegration with a hedge ratio of 1.5!
    Y_walk = 1.5 * X_walk + spread_sim

    # 2. Test Cointegration using Engle-Granger
    adf_stat, crit_5pct, computed_spread = perform_engle_granger_test(Y_walk, X_walk)
    
    print("Engle-Granger Cointegration Analysis:")
    print(f"  ADF Test Statistic on Residuals: {adf_stat:.4f}")
    print(f"  5% Critical Value Threshold:     {crit_5pct:.4f}")
    
    if adf_stat < crit_5pct:
        print("  DECISION: ADF statistic is below critical threshold. Cointegration CONFIRMED!")
    else:
        print("  DECISION: Fail to reject unit root. Cointegration REJECTED.")

    # 3. Estimate Ornstein-Uhlenbeck parameters & Half-life
    ou_metrics = estimate_ou_parameters(computed_spread)
    
    print("\n" + "-"*50)
    print("ORNSTEIN-UHLENBECK SDE FIT RESULTS")
    print("-"*50)
    print(f"  Estimated Speed of Reversion (theta): {ou_metrics['theta']:.4f} (True: {theta_true:.2f})")
    print(f"  Estimated Equilibrium Mean (mu):     {ou_metrics['mu']:.4f} (True: {mu_true:.2f})")
    print(f"  Calculated Half-life of Reversion:    {ou_metrics['half_life']:.2f} trading days")
    
    # Let's plot the computed spread to visually inspect the stationarity
    print("\n[Visualizing Stationarity] Saved a spread simulation plot to 'pairs_spread.png'")
    plt.figure(figsize=(10, 4))
    plt.plot(computed_spread, label="Residual Spread", color="purple")
    plt.axhline(y=ou_metrics['mu'], color="red", linestyle="--", label="Estimated Mean (mu)")
    plt.title("Cointegrated Residual Spread: Mean-Reverting Stationary Path")
    plt.xlabel("Trading Days")
    plt.ylabel("Spread Price")
    plt.legend()
    plt.savefig("pairs_spread.png")
    plt.close()
