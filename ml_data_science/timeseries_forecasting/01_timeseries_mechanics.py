# ============================================================================
# TIME SERIES FORECASTING: ADVANCED FEATURE ENGINEERING & TEMPORAL VALIDATION
# ============================================================================
# Target Audience: Quantitative Developers, Kaggle Competitors, Data Scientists.
# Topic: Temporal Leakage, Purging & Embargoing, Fourier seasonal terms, 
#        Rolling EMA, and Tree Extrapolation limitations.
# Compatibility: Python 3.10+, NumPy 1.22+, Pandas 1.4+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Time Series Data?
   - Time series data is a sequence of observations recorded at regular time intervals 
     (e.g., hourly electricity usage or daily product sales). The order of rows is crucial 
     because current values depend heavily on historical trends.
2. What is a Lag Feature?
   - A lag feature is a way to feed historical context into machine learning models. It 
     simply shifts data backwards. For example, a "lag_1" column represents "yesterday's sales", 
     which is highly predictive of "today's sales".
3. What is a Rolling Feature?
   - A rolling feature calculates statistical aggregates (like a rolling mean or standard 
     deviation) over a moving window of past days (e.g., average sales over the last 7 days), 
     smoothing out short-term noise to capture weekly or monthly trends.
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 1. TEMPORAL VALIDATION MECHANICS: PURGING & EMBARGO
# ============================================================================
"""
Why is standard K-Fold cross-validation invalid for Time Series or Financial data?
1. Temporal Leakage (Look-Ahead Bias): Standard K-Fold randomizes data, meaning we 
   train on future data to predict the past.
2. Information Leakage from Serial Correlation (Overlapping Windows):
   In finance and retail (e.g., store sales), labels are often determined over 
   overlapping temporal windows (such as rolling 7-day sales or forward 5-day returns). 
   Because data points are highly autocorrelated, information from validation folds 
   leaks into adjacent training folds.

To solve this, we implement Marcos Lopez de Prado's 'Purging & Embargoing' framework:

A. PURGING:
   - Eliminates from the training set any labels whose evaluation windows overlap 
     with the test/validation set. This prevents direct information leakage 
     from overlapping metrics.

B. EMBARGO:
   - Since time-series effects (autoregression, momentum, AR(p) processes) persist 
     beyond the overlapping window, we discard training data immediately *after* 
     the test window. 
   - This 'embargo' acts as a temporal safety buffer, allowing the auto-regressive 
     memory of the system to decay before training resumes.
"""


# ============================================================================
# 2. RUNNABLE KAGGLE FEATURE ENGINEERING PIPELINE FOR STORE SALES
# ============================================================================
"""
Let's build a functional, leak-free feature engineering script demonstrating:
1. Fourier terms for multiple seasonality (weekly + yearly).
2. Historical Rolling Exponential Moving Averages (EMA).
3. Leak-free Lags.
"""

def generate_fourier_terms(dates: pd.Series, period: float, max_order_k: int) -> pd.DataFrame:
    """
    Generates Fourier sine and cosine terms to capture complex overlapping 
    seasonal frequencies (e.g., weekly period=7.0, yearly period=365.25).
    
    Formula:
        fourier_sin_k = sin(2 * pi * k * t / period)
        fourier_cos_k = cos(2 * pi * k * t / period)
    """
    fourier_df = pd.DataFrame(index=dates.index)
    # Convert dates to numerical day-count
    day_count = (dates - dates.min()).dt.days.astype(float)
    
    for k in range(1, max_order_k + 1):
        theta = 2.0 * np.pi * k * day_count / period
        fourier_df[f"fourier_sin_{period}_{k}"] = np.sin(theta)
        fourier_df[f"fourier_cos_{period}_{k}"] = np.cos(theta)
        
    return fourier_df


def build_kaggle_sales_features():
    print("\n" + "="*80)
    print("2. RUNNING KAGGLE STORE SALES FEATURE ENGINEERING PIPELINE")
    print("="*80)

    # 1. Create a dummy retail dataset of 3 years of daily store sales
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    n_days = len(dates)

    # Generate synthetic sales containing trend, weekly, and yearly seasonal cycles
    trend = np.linspace(100.0, 300.0, n_days)
    weekly_cycle = 30.0 * np.sin(2.0 * np.pi * dates.dayofweek / 7.0)
    yearly_cycle = 80.0 * np.sin(2.0 * np.pi * dates.dayofyear / 365.25)
    noise = np.random.normal(0.0, 15.0, n_days)

    sales = trend + weekly_cycle + yearly_cycle + noise

    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'store_id': np.random.choice([1, 2], size=n_days) # Two retail stores
    })

    print(f"Generated {len(df)} days of historical retail data.")
    print(df.head(3))

    # 2. GENERATE ADVANCED TEMPORAL FEATURES (SINE/COSINE FOURIER TERMS)
    # We model Weekly cycle (period=7) with order=1, and Yearly cycle (period=365.25) with order=3
    print("\n  Calculating Fourier terms for weekly and yearly periods...")
    f_weekly = generate_fourier_terms(df['date'], period=7.0, max_order_k=1)
    f_yearly = generate_fourier_terms(df['date'], period=365.25, max_order_k=3)

    df = pd.concat([df, f_weekly, f_yearly], axis=1)

    # 3. CALCULATING LEAK-FREE HISTORICAL FEATURES (LAGS & EMA)
    print("  Generating historical lags and Rolling Exponential Moving Averages...")
    
    # Store sales are highly autocorrelated. We shift our lag to ensure no data leakage 
    # during validation (e.g. if we want to forecast t+1, we can use lag 1. If we forecast 
    # a 14-day horizon, we must shift lag by 14!). Let's assume a 1-day forecast horizon.
    df['sales_lag_1'] = df.groupby('store_id')['sales'].shift(1)
    df['sales_lag_7'] = df.groupby('store_id')['sales'].shift(7)

    # Rolling Exponential Moving Average (EMA) of historical lag
    # Note: we apply EMA *on the shifted lag* to prevent current-step sales leaking!
    df['sales_ema_7'] = df.groupby('store_id')['sales_lag_1'].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df['sales_ema_30'] = df.groupby('store_id')['sales_lag_1'].transform(lambda x: x.ewm(span=30, adjust=False).mean())

    # Drop NaNs created by shifts
    df = df.dropna().reset_index(drop=True)

    print("\nFeature Engineering Completed. Final features sample columns:")
    features_list = ['date', 'sales', 'sales_lag_1', 'sales_lag_7', 'sales_ema_7', 'fourier_sin_7_1', 'fourier_cos_365.25_3']
    print(df[features_list].head(5))


# ============================================================================
# 3. INTERVIEW PITFALL: TREE EXTRAPOLATION LIMITATIONS
# ============================================================================
"""
A fundamental limitation of Tree-based models (XGBoost, LightGBM, Random Forest) 
is their INABILITY to extrapolate trends.

Why?
- Decision trees make predictions by partitioning the feature space into hyper-rectangles 
  and assigning a constant leaf value (the mean of the training samples in that leaf).
- If the feature value of a test sample lies outside the bounds of the training data 
  (e.g., a date index or lag feature that is larger than anything seen in training due 
  to a rising linear trend), the tree will route the sample to its outermost leaf.
- Consequently, the model will output a flat maximum constant prediction, completely 
  failing to capture the upward trend!

Solutions:
1. De-trending: Fit a linear regression model to the trend, subtract the trend from the 
   sales data, model the residual seasonalities with XGBoost, and then add the trend 
   back in during inference.
2. Differencing: Model first-order differences: Y'_t = Y_t - Y_{t-1} rather than 
   absolute values.
3. Neural/Statistical models: Use architectures like ARIMA, DeepAR, or Transformer-based 
   PatchTST which model auto-regressive scaling functions natively.
"""

def demonstrate_tree_extrapolation_failure():
    print("\n" + "="*80)
    print("3. EXPLAINING THE TREE EXTRAPOLATION LIMITATION")
    print("="*80)

    # Let's say we have an index t from 1 to 100
    t_train = np.arange(1, 81).reshape(-1, 1) # First 80 days
    y_train = 2.5 * t_train.flatten() + 10.0  # Linear trend: y = 2.5t + 10

    t_test = np.arange(81, 101).reshape(-1, 1) # Days 81 to 100
    y_test_ground_truth = 2.5 * t_test.flatten() + 10.0

    # Let's simulate what a decision tree (or ensemble of trees like LightGBM) does:
    # Any split will be of the form: if t > threshold: predict leaf_val
    # Since the maximum threshold the tree can ever learn during training is <= 80, 
    # any value of t in testing (81 to 100) will be routed to the rightmost leaf:
    #   if t > 79.5: predict mean(y_train[t_train > 79.5])
    
    # We calculate the prediction for t > 80:
    predicted_val = np.mean(y_train[t_train.flatten() > 75]) # Simulating the outer-leaf value
    predictions = np.ones_like(t_test.flatten()) * predicted_val

    print(f"Training Time Range: Day 1 to 80 (Sales values: {y_train[0]:.1f} to {y_train[-1]:.1f})")
    print(f"Testing Time Range:  Day 81 to 100 (Ground Truth sales: {y_test_ground_truth[0]:.1f} to {y_test_ground_truth[-1]:.1f})")
    print(f"\nDecision Tree Prediction for Day 81-100: {predictions[0]:.1f} (Flat constant!)")
    print(f"  Maximum Absolute Error at Day 100:      {np.abs(y_test_ground_truth[-1] - predictions[-1]):.2f} (Extreme underestimation!)")
    print("\nNOTE: Because GBDTs only predict averages of local training neighborhoods, they cannot extrapolate outside their training domain!")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Why is standard K-Fold cross-validation invalid for Time Series datasets?
   - Answer: Standard K-Fold shuffles the records randomly. In time series, this causes 
     severe temporal data leakage (look-ahead bias) where future records are present 
     in training folds to predict historical values in validation folds. This leads 
     to an artificial, overly optimistic validation score that collapses in production.

2. MEDIUM: What is the purpose of 'Purging' and 'Embargoing' in time series cross-validation?
   - Answer:
     - Purging: If labels are calculated over overlapping windows (e.g. 7-day rolling sales), 
       observations in the training set adjacent to the validation set will share overlapping 
       raw data, causing leakages. Purging removes any training labels whose evaluation 
       ranges overlap with the validation split.
     - Embargoing: Since serial correlation (momentum/autoregression) persists even after 
       purging, training data immediately following the validation set will still contain 
       remnant validation information. Embargoing discards a fixed duration of training 
       records immediately *after* the validation split to let autoregressive effects decay.

3. HARD: Why can't Gradient Boosted Trees (like XGBoost or LightGBM) extrapolate trends? 
   How do you resolve this?
   - Answer: Decision trees partition feature coordinates into regions and predict a constant 
     value (the training mean) in each leaf. When evaluating samples with time steps (or features) 
     exceeding the maximum training range, the sample is routed to the outermost leaf, 
     producing a flat, non-trending prediction.
     Solutions:
     (1) De-trending: Fit a Linear Regression on the trend, subtract it to make the target 
         stationary, train XGBoost on the residuals, and re-add the linear prediction at inference.
     (2) Differencing: Train the model to forecast differences ($Y_t - Y_{t-1}$) instead of 
         absolute values, which makes the target stationary and avoids monotonic coordinate splits.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a leak-free lag generator function `create_historical_lags(df, column, lags)` 
   that adds specified lag offsets to a pandas DataFrame, ensuring no future elements leak.
   - Code Template to execute:
"""

def create_historical_lags(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    df_out = df.copy()
    for lag in lags:
        # positive lag shifts indices forward in time, so index t gets value at t-lag (purely historical)
        df_out[f"{column}_lag_{lag}"] = df_out[column].shift(lag)
    return df_out


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    build_kaggle_sales_features()
    demonstrate_tree_extrapolation_failure()
    
    # Testing the lag generator
    print("\nTesting Custom Lag Generator:")
    test_df = pd.DataFrame({'sales': [100.0, 110.0, 120.0, 130.0]})
    lagged = create_historical_lags(test_df, 'sales', [1, 2])
    print(lagged)