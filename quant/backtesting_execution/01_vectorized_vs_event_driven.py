# ============================================================================
# QUANT backtesting_execution: VECTORIZED BACKTEST & MAXIMUM DRAWDOWN
# ============================================================================
# Target Audience: Quant Developers, Algorithmic Engineers, Quant Researchers.
# Topic: Vectorized vs Event-Driven Backtesting, Slippage Cost Modeling, 
#        Sharpe/Sortino Metrics, and vectorized Maximum Drawdown.
# Compatibility: Python 3.10+, NumPy 1.22+, Pandas 1.4+
# ============================================================================

import pandas as pd
import numpy as np

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Backtesting?
   - Backtesting is the process of testing a trading strategy using historical data 
     to see how it would have performed. It helps quants evaluate profitability, 
     volatility, and risk metrics before putting real capital at risk.
2. What is a Vectorized Backtester?
   - A vectorized backtester processes the entire historical dataset at once using 
     fast array operations (like Pandas or NumPy). It is extremely fast and perfect 
     for testing simple ideas, but can suffer from "look-ahead bias" (accidentally 
     using future data to make past decisions).
3. What is an Event-Driven Backtester?
   - An event-driven backtester processes data sequentially, row-by-row (or tick-by-tick), 
     simulating real-time market events. It is much more realistic because it can model 
     order books, transaction fees, slippage, and latency, but it is slower to run.
4. What is Maximum Drawdown?
   - Maximum Drawdown (MDD) is the largest peak-to-trough decline in the value of a 
     portfolio's equity curve before a new peak is attained. It represents the worst-case 
     paper loss an investor would have experienced over the testing period.
"""


# ============================================================================
# 1. VECTORIZED VS. EVENT-DRIVEN ENGINE COMPREHENSION
# ============================================================================
"""
Architectural Trade-offs:

| Criterion | Vectorized Backtesting | Event-Driven Backtesting |
|-----------|------------------------|---------------------------|
| Execution Speed | Sub-second (Optimal for hyper-parameter search) | Minutes to hours (Tick-by-tick processing) |
| Code Complexity | Low (Few lines of Pandas/NumPy) | High (Requires State machines, Order queues, Portfolio trackers) |
| Look-Ahead Bias | High risk (Easy to leak future state via .shift()) | Low risk (Sequential iteration prevents looking forward) |
| Order Execution | Highly simplified (Assumes fill at Close or Open) | High realism (Simulates Limit/Market orders, Bid-Ask queues) |
| Capital Constraints | Hard to model margin calls or dynamic collateral | Natively tracks cash balances, leverage, and liquidations |
"""


# ============================================================================
# 2. RUNNABLE HIGH-PERFORMANCE VECTORIZED BACKTEST WITH TRANSACTION COSTS
# ============================================================================
"""
A common mistake in backtests is assuming "frictionless execution". In reality, 
each trade incurs costs:
    Friction Cost = Slippage (Market Impact) + Brokerage Commission (Fees)
    
Slippage models:
- Constant slippage: Fill price is worse by a fixed percentage or tick count.
- Linear execution cost model: 
    Cost_t = Size_t * (Spread_t / 2 + commission_rate)
"""

def run_vectorized_moving_average_crossover(prices_series: pd.Series, 
                                            fast_window: int = 10, 
                                            slow_window: int = 30, 
                                            commission: float = 0.001, # 0.1% fee
                                            slippage: float = 0.0005) -> pd.DataFrame:
    """
    Executes a vectorized SMA Crossover strategy:
    - Long when Fast SMA > Slow SMA
    - Flat when Fast SMA < Slow SMA
    """
    df = pd.DataFrame({"Close": prices_series})
    
    # 1. Calculate indicators
    df["Fast_SMA"] = df["Close"].rolling(window=fast_window).mean()
    df["Slow_SMA"] = df["Close"].rolling(window=slow_window).mean()
    
    # 2. Generate target positions: 1.0 (Long), 0.0 (Flat)
    # We shift target position by 1 day to prevent look-ahead bias! 
    # (We enter tomorrow based on today's indicators closing state)
    df["Signal"] = (df["Fast_SMA"] > df["Slow_SMA"]).astype(float)
    df["Position"] = df["Signal"].shift(1).fillna(0.0)
    
    # 3. Calculate log returns of the underlying asset
    df["Market_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # 4. Strategy returns (before costs)
    df["Strategy_Returns_Raw"] = df["Position"] * df["Market_Returns"]
    
    # 5. Model transaction frictions
    # Trades occur when the position changes: Position_t != Position_{t-1}
    df["Trades"] = df["Position"].diff().fillna(0.0).abs()
    
    # Apply trade friction penalty on day trade occurs
    total_friction_rate = commission + slippage
    df["Friction_Penalty"] = df["Trades"] * total_friction_rate
    
    # Net strategy returns
    df["Strategy_Returns_Net"] = df["Strategy_Returns_Raw"] - df["Friction_Penalty"]
    
    # 6. Cumulative Equity Curves
    df["Market_Equity"] = np.exp(df["Market_Returns"].cumsum())
    df["Strategy_Equity_Raw"] = np.exp(df["Strategy_Returns_Raw"].cumsum())
    df["Strategy_Equity_Net"] = np.exp(df["Strategy_Returns_Net"].cumsum())
    
    return df


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is "Look-Ahead Bias" in backtesting, and how do you prevent it?
   - Answer: Look-ahead bias occurs when a backtest utilizes data that would not have 
     been available at the moment of decision-making. For example, using today's close price 
     to determine today's position, or running rolling operations that leak future information. 
     Prevent it by always shifting indicators forward by 1 step (`df['Position'] = df['Signal'].shift(1)`) 
     so decisions are executed on subsequent market steps.

2. MEDIUM: Contrast the Sharpe Ratio with the Sortino Ratio. Under what circumstances 
   is the Sortino Ratio preferred?
   - Answer: The Sharpe Ratio penalizes *all* volatility (both upward price jumps and downward price 
     declines) in its denominator. The Sortino Ratio only penalizes *downside* volatility (returns 
     below a target hurdle rate, usually 0 or the risk-free rate). Sortino is preferred for 
     strategies with highly asymmetric return profiles, such as long options buying or trend-following 
     momentum, where large positive upward returns would otherwise skew Sharpe ratios downward.

3. HARD: Explain "Survivorship Bias" and "Market Impact" in backtesting platforms.
   - Answer: 
     - Survivorship Bias: Occurs when you backtest a stock strategy using only currently active 
       companies. It ignores companies that went bankrupt, merged, or got delisted during the historical 
       window, resulting in heavily inflated historical returns. Solve by using point-in-time, 
       survivorship-bias-free datasets.
     - Market Impact: Large trade orders shift prices. A backtest that assumes massive blocks can be 
       filled at the mid-market price without moving the bid-ask spread is unrealistic. Quants model 
       this using non-linear impact models (e.g. price impact scales with the square root of order size 
       relative to daily volume).

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `calculate_max_drawdown(equity_series)` that calculates 
   the Maximum Drawdown (MDD) of an equity curve without using Python loops (vectorized).
   - Drawdown Equation: DD_t = (Equity_t - Running_Peak_t) / Running_Peak_t
   - Code Template to execute:
"""

def calculate_max_drawdown(equity_series: pd.Series) -> tuple[float, pd.Series]:
    # 1. Calculate running historical peak at each step t using .cummax()
    running_peaks = equity_series.cummax()
    
    # 2. Calculate percentage drawdowns relative to current running peak
    drawdowns = (equity_series - running_peaks) / running_peaks
    
    # 3. Maximum Drawdown is the minimum value (most negative decline)
    max_dd = drawdowns.min()
    
    return max_dd, drawdowns


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING HIGH-PERFORMANCE VECTORIZED BACKTESTING")
    print("="*80)

    # 1. Generate synthetic trending stock price series (GBM with positive drift)
    np.random.seed(109)
    n_days = 250
    daily_returns = np.random.normal(0.0005, 0.01, n_days) # Drift +0.05% daily, Vol 1%
    stock_prices = 100.0 * np.exp(np.cumsum(daily_returns))
    prices_index = pd.date_range(start="2025-01-01", periods=n_days)
    prices_series = pd.Series(stock_prices, index=prices_index)

    # 2. Run vectorized backtest crossover
    fast = 5
    slow = 20
    comm = 0.001   # 0.1% broker fee
    slip = 0.0005  # 0.05% slippage
    
    results_df = run_vectorized_moving_average_crossover(prices_series, fast, slow, comm, slip)
    
    # 3. Compute Risk Metrics
    final_market_eq = results_df["Market_Equity"].iloc[-1]
    final_strategy_raw_eq = results_df["Strategy_Equity_Raw"].iloc[-1]
    final_strategy_net_eq = results_df["Strategy_Equity_Net"].iloc[-1]
    
    # Sharpe Ratio: Mean / Std (annualized)
    daily_sharpe = results_df["Strategy_Returns_Net"].mean() / results_df["Strategy_Returns_Net"].std()
    annualized_sharpe = daily_sharpe * np.sqrt(252)

    print(f"SMA Crossover Backtest Performance (Fast: {fast} | Slow: {slow}):")
    print(f"  Market Buy-and-Hold Cumulative Return: {((final_market_eq - 1.0) * 100):.2f}%")
    print(f"  Strategy RAW (Frictionless) Return:   {((final_strategy_raw_eq - 1.0) * 100):.2f}%")
    print(f"  Strategy NET (With Fees & Slip) Return: {((final_strategy_net_eq - 1.0) * 100):.2f}%")
    print(f"  Total Strategy Trades Executed:       {results_df['Trades'].sum():.0f}")
    print(f"  Annualized Strategy Sharpe Ratio:     {annualized_sharpe:.4f}")

    # 4. Calculate Maximum Drawdown
    max_dd, dd_series = calculate_max_drawdown(results_df["Strategy_Equity_Net"])
    
    print("\n" + "-"*50)
    print("MAXIMUM DRAWDOWN CALCULATION")
    print("-"*50)
    print(f"  Vectorized Maximum Drawdown (MDD): {max_dd * 100:.2f}%")
    
    # Find peak and valley index of the worst drawdown
    valley_idx = dd_series.idxmin()
    peak_before_valley = results_df["Strategy_Equity_Net"].loc[:valley_idx].max()
    valley_val = results_df["Strategy_Equity_Net"].loc[valley_idx]
    
    print(f"  Worst Decline Peak:  {peak_before_valley:.4f}")
    print(f"  Worst Decline Valley: {valley_val:.4f} (Occurred on {valley_idx.strftime('%Y-%m-%d')})")
