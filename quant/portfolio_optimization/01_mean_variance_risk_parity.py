# ============================================================================
# QUANT portfolio_optimization: MEAN-VARIANCE, SHRINKAGE, & RISK PARITY
# ============================================================================
# Target Audience: Quant Researchers, Portfolio Managers, Asset Allocators.
# Topic: Modern Portfolio Theory, Sharpe Maximization, Ledoit-Wolf Shrinkage, 
#        Risk Parity (Equal Risk Contribution), and SLSQP Optimization.
# Compatibility: Python 3.10+, NumPy 1.22+, SciPy 1.8+
# ============================================================================

import numpy as np
import scipy.optimize as sco
import scipy.linalg as sla

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Portfolio Optimization?
   - Portfolio optimization is the mathematical process of selecting the weights 
     (allocations) of various assets (like stocks, bonds, or commodities) to achieve 
     either the lowest possible risk, the highest possible return, or the best 
     risk-adjusted return (Sharpe Ratio).
2. What is Mean-Variance Optimization (Markowitz)?
   - Developed by Harry Markowitz, it balances the trade-off between the expected return 
     (mean) of a portfolio and its volatility (variance). The "Efficient Frontier" 
     represents portfolios that offer the highest expected return for a given level of risk.
3. What is the Sharpe Ratio?
   - The Sharpe Ratio measures risk-adjusted performance. It is the portfolio's expected excess 
     return (portfolio return minus risk-free rate) divided by its standard deviation. 
     Higher Sharpe ratios signify more return per unit of volatility.
4. What is Risk Parity?
   - Standard optimization allocates capital (dollars). Risk Parity allocates risk. 
     It constructs a portfolio where every asset contributes exactly the same amount of 
     volatility (risk) to the total portfolio risk, preventing volatile assets from 
     dominating the portfolio's behavior.
"""


# ============================================================================
# 1. LEDOIT-WOLF COVARIANCE SHRINKAGE MATHEMATICS
# ============================================================================
"""
In quantitative research, using a raw Sample Covariance matrix $\Sigma_{sample}$ 
leads to unstable optimization weights. This is because high-dimensional asset pools 
have extreme eigenvalues that represent sample noise rather than actual risk relationships.

Ledoit-Wolf shrinkage stabilizes the matrix by interpolating the sample covariance 
with a structured target matrix $F$ (usually a constant correlation matrix):
    \Sigma_{shrunk} = \delta F + (1 - \delta) \Sigma_{sample}
Where $\delta \in [0, 1]$ is the shrinkage intensity. This guarantees a positive-definite 
matrix that is highly robust to matrix inversion.
"""

def ledoit_wolf_shrinkage_emulation(returns_matrix: np.ndarray, shrinkage_intensity: float = 0.1) -> np.ndarray:
    """
    Computes a simplified shrunk covariance matrix.
    """
    # 1. Calculate sample covariance
    sample_cov = np.cov(returns_matrix, rowvar=False)
    
    # 2. Build target matrix: Constant Correlation target
    variances = np.diag(sample_cov)
    stds = np.sqrt(variances)
    
    # Average correlation
    correlation_matrix = sample_cov / np.outer(stds, stds)
    np.fill_diagonal(correlation_matrix, 1.0)
    avg_corr = (np.sum(correlation_matrix) - len(stds)) / (len(stds) * (len(stds) - 1))
    
    # Construct target matrix F
    F = np.ones_like(correlation_matrix) * avg_corr
    np.fill_diagonal(F, 1.0)
    F = F * np.outer(stds, stds)
    
    # 3. Shrink: Delta * F + (1 - Delta) * Sample_Cov
    shrunk_cov = shrinkage_intensity * F + (1.0 - shrinkage_intensity) * sample_cov
    return shrunk_cov


# ============================================================================
# 2. RISK PARITY (EQUAL RISK CONTRIBUTION) FORMULATION
# ============================================================================
"""
The total risk (volatility) of a portfolio with weights $w$ and covariance $\Sigma$ is:
    \sigma_p = \sqrt{w^T \Sigma w}
The Marginal Risk Contribution (MRC) of asset $i$ is:
    MRC_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}
The Total Risk Contribution (RC) of asset $i$ is:
    RC_i = w_i \cdot MRC_i = w_i \cdot \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}

Risk Parity (Equal Risk Contribution) demands that:
    RC_i = RC_j \quad \forall i, j
This is solved by minimizing the squared differences between risk contributions:
    \min_w \sum_{i=1}^N \sum_{j=1}^N \left( RC_i - RC_j \right)^2
Under constraints: $\sum w_i = 1.0$ and $w_i \ge 0$.
"""

def risk_parity_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    # 1. Calculate portfolio volatility
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    
    # 2. Calculate individual risk contributions
    mrc = np.dot(cov, weights) / portfolio_vol
    rc = weights * mrc
    
    # 3. Calculate sum of squared differences of risk contributions
    # We want to minimize the pairwise difference
    num_assets = len(weights)
    target_rc = portfolio_vol / num_assets
    
    squared_errors = np.sum((rc - target_rc) ** 2)
    return squared_errors

def optimize_risk_parity(cov: np.ndarray) -> np.ndarray:
    num_assets = len(cov)
    initial_weights = np.ones(num_assets) / num_assets
    
    # Constraints: sum of weights equals 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # Bounds: no short selling (weights between 0 and 1)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Optimize using SLSQP non-linear solver
    res = sco.minimize(
        fun=risk_parity_objective,
        x0=initial_weights,
        args=(cov,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9}
    )
    return res.x


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Why is expected return estimation harder than risk (covariance) estimation?
   - Answer: Expected asset returns are highly non-stationary and influenced by shifting 
     market regimes, sentiment, and macro economics, exhibiting extremely low signal-to-noise 
     ratios. Risk relationships (covariance and correlation), however, are governed by 
     structural corporate asset ties, market sectors, and leverage, making historical volatility 
     and covariances far more stable and predictable across time steps.

2. MEDIUM: What is the main structural weakness of Markowitz Mean-Variance Optimization 
   in production, and how do quants address it?
   - Answer: Markowitz is highly sensitive to input estimation errors. Small variations 
     in expected return predictions yield extreme, highly concentrated asset weights 
     (unstable allocations) that incur high transaction turnover costs. Quants address this by:
     (1) Using robust covariance estimators like Ledoit-Wolf shrinkage, 
     (2) Setting allocation bounds (e.g. max 15% per asset), or 
     (3) Utilizing the Black-Litterman model to blend market equilibrium returns with 
         subjective views.

3. HARD: Prove or explain why the Maximum Sharpe Ratio portfolio weights do not depend 
   on the absolute magnitude of the asset returns, but rather their relative proportions.
   - Answer: The Sharpe Ratio objective is scale-invariant. If we scale all expected returns 
     by a constant factor $c > 0$, the optimization objective $(w^T (c\mu) - r_f) / \sigma_p$ 
     retains the exact same argmax coordinate weights $w^*$. Thus, Sharpe maximization is 
     solely dependent on the relative cross-sectional returns and covariances, making it 
     robust to uniform shifting expectations.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `optimize_maximum_sharpe_ratio(expected_returns, cov, risk_free_rate)` 
   that solves for weights that maximize the Sharpe Ratio under no-short-sale constraints 
   using SciPy SLSQP.
   - Hint: Since optimizer minimizes functions, minimize the negative Sharpe Ratio!
   - Code Template to execute:
"""

def optimize_maximum_sharpe_ratio(expected_returns: np.ndarray, cov: np.ndarray, 
                                  risk_free_rate: float) -> np.ndarray:
    num_assets = len(expected_returns)
    
    # Objective function: Negative Sharpe Ratio
    def negative_sharpe(weights: np.ndarray) -> float:
        # Expected portfolio return: w^T * mu
        port_return = np.dot(weights, expected_returns)
        # Expected portfolio volatility: sqrt(w^T * Sigma * w)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        
        # Sharpe Ratio
        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe
        
    # Constraints: sum of weights = 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # Bounds: 0.0 to 1.0 (long-only)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    initial_weights = np.ones(num_assets) / num_assets
    
    res = sco.minimize(
        fun=negative_sharpe,
        x0=initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return res.x


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING PORTFOLIO OPTIMIZATION & COVARIANCE SHRINKAGE")
    print("="*80)

    # 1. Simulate asset historical returns (high dimensions: 3 assets, 20 observations)
    np.random.seed(109)
    # Asset 0: low risk, low return (bonds)
    # Asset 1: medium risk, medium return (large cap)
    # Asset 2: high risk, high return (tech)
    ret_bonds = np.random.normal(0.02, 0.04, 100)
    ret_stocks = np.random.normal(0.08, 0.15, 100)
    ret_tech = np.random.normal(0.15, 0.25, 100)
    
    returns_matrix = np.column_stack([ret_bonds, ret_stocks, ret_tech])
    expected_returns = np.mean(returns_matrix, axis=0)
    
    # Calculate sample and shrunk covariance
    sample_cov = np.cov(returns_matrix, rowvar=False)
    shrunk_cov = ledoit_wolf_shrinkage_emulation(returns_matrix, shrinkage_intensity=0.20)

    print("Expected Asset Returns: ", expected_returns)
    print("\nSample Covariance Matrix:\n", sample_cov)
    print("Shrunk Covariance Matrix (20% Shrinkage Intensity):\n", shrunk_cov)

    # 2. Risk Parity Allocation (Equal Risk Contribution)
    # Asset 2 has 25% volatility compared to Asset 0's 4% volatility.
    # Dollar-allocation would put equal money. Risk parity will allocate far more money 
    # to Asset 0 to keep risk contributions equal!
    rp_weights = optimize_risk_parity(shrunk_cov)
    
    # Verify Risk Contributions
    port_vol = np.sqrt(np.dot(rp_weights.T, np.dot(shrunk_cov, rp_weights)))
    rc = rp_weights * np.dot(shrunk_cov, rp_weights) / port_vol
    
    print("\n" + "-"*50)
    print("RISK PARITY PORTFOLIO ALLOCATION RESULTS")
    print("-"*50)
    print(f"  Optimized Weights:     {rp_weights}")
    print(f"  Risk Contributions ($): {rc} (Sum: {np.sum(rc):.4f} | Port Vol: {port_vol:.4f})")
    assert np.allclose(rc, rc[0])
    print("  SUCCESS: Risk contributions are perfectly balanced across all assets!")

    # 3. Maximum Sharpe Ratio Optimization
    print("\n" + "-"*50)
    print("MAXIMUM SHARPE RATIO OPTIMIZATION")
    print("-"*50)
    
    rf = 0.01 # 1% risk-free rate
    sharpe_weights = optimize_maximum_sharpe_ratio(expected_returns, shrunk_cov, risk_free_rate=rf)
    
    # Portfolio stats
    opt_return = np.dot(sharpe_weights, expected_returns)
    opt_vol = np.sqrt(np.dot(sharpe_weights.T, np.dot(shrunk_cov, sharpe_weights)))
    opt_sharpe = (opt_return - rf) / opt_vol
    
    print(f"  Optimized Sharpe Weights: {sharpe_weights}")
    print(f"  Portfolio Return:         {opt_return * 100:.2f}%")
    print(f"  Portfolio Volatility:     {opt_vol * 100:.2f}%")
    print(f"  Maximum Sharpe Ratio:     {opt_sharpe:.4f}")
