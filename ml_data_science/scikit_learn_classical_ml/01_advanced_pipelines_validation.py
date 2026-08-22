# ============================================================================
# ADVANCED SCIKIT-LEARN: CUSTOM PIPELINES & ROBUST CROSS-VALIDATION
# ============================================================================
# Target Audience: Senior ML Engineers, Data Scientists.
# Topic: BaseEstimator/TransformerMixin, ColumnTransformer, Data Leakage, 
#        and GroupKFold/TimeSeriesSplit Validation.
# Compatibility: Python 3.10+, Scikit-Learn 1.2+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is a Machine Learning Pipeline?
   - Raw data must be cleaned, scaled, and encoded before it goes into a model. A Pipeline 
     packages all these pre-processing steps and the final model into a single, cohesive 
     object. This prevents human errors and guarantees that training transformations are 
     applied identically to test sets.
2. What is Data Leakage?
   - Data leakage occurs when information from outside the training dataset is accidentally 
     used to train the model. For example, if you calculate the average of a column across 
     the entire dataset (including validation data) to fill missing values, the model 
     "leaks" future validation statistics into its training phase, yielding fake high accuracy.
3. What is Cross-Validation?
   - To trust a model, we split our data into training and testing parts. Cross-validation 
     repeats this process multiple times (folds) across different subsets of data, ensuring 
     that our model is robust and performs well on any random split of data.
"""



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit
from sklearn.linear_model import Ridge

# ============================================================================
# 1. WRITING A CUSTOM TRANSFORMER (BASEESTIMATOR & TRANSFORMERMIXIN)
# ============================================================================
"""
Why write custom scikit-learn transformers?
To avoid Data Leakage, every data transformation (scaling, encoding, imputation) 
should be fit ONLY on the training split of a cross-validation fold, and then 
applied to the validation split. 
By wrapping our custom pre-processing logic into an object that implements the 
scikit-learn transformer interface, we can drop it directly into a `Pipeline`.

Rules for subclassing:
- Inherit from `BaseEstimator` and `TransformerMixin`.
- `BaseEstimator` provides automatic get_params() and set_params() for hyperparameter tuning.
  - Note: Do NOT use *args or **kwargs in your constructor (`__init__`). All parameters must 
    be explicitly declared with default values.
- `TransformerMixin` automatically implements `fit_transform` once `fit` and `transform` are written.
- `fit` must return `self`.
"""

class CyclicalDateTimeEncoder(BaseEstimator, TransformerMixin):
    """
    Transforms numerical time variables (e.g. day of week, month) into 
    sine and cosine representations. This captures cyclicality (e.g., Sunday 
    and Monday are adjacent, December and January are adjacent) without 
    imposing an artificial ordinal linear progression.
    
    Formula:
        x_sin = sin(2 * pi * x / max_value)
        x_cos = cos(2 * pi * x / max_value)
    """
    def __init__(self, column_max_values: dict = None):
        """
        column_max_values: Dictionary mapping column names to their maximum 
                           logical cyclical values (e.g., {'day_of_week': 6, 'month': 12})
        """
        self.column_max_values = column_max_values

    def fit(self, X, y=None):
        # Cyclical transformations are purely analytical; no parameters are fit.
        # However, we must implement fit and return self to respect the protocol.
        return self

    def transform(self, X):
        """
        X: Pandas DataFrame or NumPy array containing the target cyclical columns
        """
        # Ensure we are working with a Pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_out = X.copy()
        
        for col, max_val in self.column_max_values.items():
            if col in X_out.columns:
                # Calculate cyclical components
                radians = 2 * np.pi * X_out[col].astype(float) / max_val
                X_out[f"{col}_sin"] = np.sin(radians)
                X_out[f"{col}_cos"] = np.cos(radians)
                # Drop original column to avoid collinearity
                X_out = X_out.drop(columns=[col])
                
        return X_out


# ============================================================================
# 2. COMBINING CUSTOM TRANSFORMERS & COLUMNTRANSFORMERS IN PIPELINES
# ============================================================================

def build_advanced_pipeline() -> Pipeline:
    """
    Creates a robust ML pipeline that parses numerical, categorical, and cyclical
    features without leaking pre-processing parameters.
    """
    # Define our custom cyclical mapping (Day of week from 0-6, Month from 1-12)
    cyclical_map = {'day_of_week': 6.0, 'month': 12.0}
    
    # 1. Create specialized transformers for different feature categories
    cyclical_preprocessor = Pipeline([
        ('cyclical', CyclicalDateTimeEncoder(column_max_values=cyclical_map)),
        ('scaler', StandardScaler()) # Scale sine/cosine features
    ])

    categorical_preprocessor = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])

    # 2. Bundle preprocessors together using ColumnTransformer
    # ColumnTransformer routes columns to their correct transformation pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_preprocessor, ['sales_lag_1', 'temperature']),
            ('cat', categorical_preprocessor, ['store_type']),
            ('cyclical_dates', cyclical_preprocessor, ['day_of_week', 'month'])
        ],
        remainder='drop' # Drop columns not specified to avoid structural leakage
    )

    # 3. Create the final unified pipeline containing pre-processing and estimator
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    return full_pipeline


# ============================================================================
# 3. ADDRESSING DATA LEAKAGE & ROBUST VALIDATION STRATEGIES
# ============================================================================
"""
What is DATA LEAKAGE?
Data Leakage occurs when information from outside the training dataset is used 
to train the model. This leads to overly optimistic cross-validation scores, 
but catastrophic model failure in production.

Example of Leakage:
1. Fitting a StandardScaler on the entire dataset *before* running K-Fold.
   - Why is it leakage? The mean and standard deviation of the validation sets 
     influence the training scaling, leaking validation information.
2. Random train/test split on Grouped Data (e.g. patients, retail store IDs).
   - If a model is trained on patient A's blood pressure from Tuesday, and tested 
     on patient A's blood pressure from Thursday, the model memorizes patient A's 
     baseline rather than learning generalizable patient rules.
   - Solution: Use `GroupKFold` to ensure identical groups NEVER overlap between 
     train and validation folds.
3. Random train/test split on Time-Series Data.
   - If you shuffle time-series data, you predict past values using future values 
     (temporal leakage).
   - Solution: Use `TimeSeriesSplit` (rolling or expanding window).
"""

def demonstrate_advanced_validation():
    print("\n" + "="*80)
    print("3. TESTING PIPELINES AND ADVANCED VALIDATION METHODS")
    print("="*80)

    # Create dummy store sales dataframe
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'store_id': np.random.randint(1, 11, size=n_samples), # 10 different stores
        'month': np.random.randint(1, 13, size=n_samples),
        'day_of_week': np.random.randint(0, 7, size=n_samples),
        'sales_lag_1': np.random.exponential(scale=100.0, size=n_samples),
        'temperature': np.random.normal(loc=15.0, scale=8.0, size=n_samples),
        'store_type': np.random.choice(['A', 'B', 'C'], size=n_samples)
    })
    
    # Target value: sales
    data['sales'] = data['sales_lag_1'] * 0.8 + data['temperature'] * 2.5 + np.random.normal(0.0, 10.0, size=n_samples)

    X = data.drop(columns=['sales'])
    y = data['sales']
    groups = data['store_id'] # Store IDs represent logical grouping

    # Build our custom pipeline
    pipeline = build_advanced_pipeline()

    # Define hyperparameter grid search
    # Note double-underscore syntax to query parameters nested inside pipelines:
    # <step_name>__<transformer_or_estimator_parameter>
    param_grid = {
        'preprocessor__num__scaler__with_mean': [True, False],
        'regressor__alpha': [0.1, 1.0, 10.0]
    }

    # Scenario A: Evaluating on unseen groups (Out-Of-Sample, Out-Of-Group)
    # Using GroupKFold ensures stores in validation splits are completely unseen!
    print("Executing GroupKFold (group=store_id)...")
    gkf = GroupKFold(n_splits=5)
    grid_gkf = GridSearchCV(pipeline, param_grid, cv=gkf, scoring='r2')
    grid_gkf.fit(X, y, groups=groups)
    print(f"  Best GroupKFold R^2 score: {grid_gkf.best_score_:.4f}")
    print(f"  Best Parameters:           {grid_gkf.best_params_}")

    # Scenario B: Evaluating on temporal forecast splits
    # Using TimeSeriesSplit prevents temporal leakage!
    print("\nExecuting TimeSeriesSplit (rolling expanding window)...")
    tss = TimeSeriesSplit(n_splits=5)
    grid_tss = GridSearchCV(pipeline, param_grid, cv=tss, scoring='r2')
    grid_tss.fit(X, y)
    print(f"  Best TimeSeriesSplit R^2:  {grid_tss.best_score_:.4f}")
    print(f"  Best Parameters:           {grid_tss.best_params_}")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the benefit of inheriting from BaseEstimator and TransformerMixin in scikit-learn?
   - Answer: `BaseEstimator` automatically provides `get_params()` and `set_params()` 
     which are required for pipeline parameter sweeping inside `GridSearchCV`. 
     `TransformerMixin` automatically implements `fit_transform()` from your custom `fit()` 
     and `transform()` methods.

2. MEDIUM: Provide three ways data leakage can occur during pre-processing, and how to avoid them.
   - Answer: 
     (1) Fitting a Scaler (e.g. StandardScaler) on the entire dataset before splitting folds.
     (2) Computing target statistics (Target Encoding) using the full target vector.
     (3) Computing the median of columns for missing value imputation on the entire corpus.
     Avoidance: Package all transformers inside an sklearn `Pipeline` and run hyperparameter 
     search via `GridSearchCV(pipeline, ...)`. The pipeline guarantees that `.fit()` runs 
     exclusively on the training splits of each fold, and `.transform()` runs on the test split.

3. HARD: Explain the difference between KFold, GroupKFold, and TimeSeriesSplit. When is each used?
   - Answer:
     - KFold: Assumes IID (Independent and Identically Distributed) data. Elements can be 
       split completely at random.
     - GroupKFold: Used when data has underlying grouping (e.g., patient records over multiple 
       days, or retail store locations). It guarantees that a group (e.g., store 1) is never 
       split across both training and validation folds, verifying out-of-group generalization.
     - TimeSeriesSplit: Used when records have temporal dependencies. It uses a rolling forward 
       window (train on t_1, validation t_2; train on t_1 & t_2, validation t_3), ensuring 
       no look-ahead (future) leaks.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom scikit-learn transformer called `NullIndicatorImputer` 
   that imputes missing values in numerical columns with their median (learned during training), 
   but also automatically appends a binary helper column `{col_name}_is_null` indicating 
   where the missing values occurred.
   - Code Template to execute:
"""

class NullIndicatorImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        # Ensure we are in a pandas DataFrame
        df = pd.DataFrame(X)
        for col in df.columns:
            # Learn the median of the training split
            self.medians_[col] = df[col].median()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, val in self.medians_.items():
            if col in df.columns:
                # 1. Create the binary indicator column
                df[f"{col}_is_null"] = df[col].isnull().astype(float)
                # 2. Impute raw column with median
                df[col] = df[col].fillna(val)
        return df


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_advanced_validation()
    
    # Testing the custom imputer
    print("\nTesting Custom Null Indicator Imputer:")
    X_test = pd.DataFrame({'age': [25.0, np.nan, 35.0, 45.0]})
    imputer = NullIndicatorImputer()
    imputer.fit(X_test)
    transformed = imputer.transform(X_test)
    print(f"  Training Medians: {imputer.medians_} (Expected Median of age: 35.0)")
    print(f"  Transformed Output:\n{transformed}")