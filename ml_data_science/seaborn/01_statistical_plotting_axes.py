# ============================================================================
# SEABORN STATISTICAL GRAPHING: FIGURE-LEVEL VS AXES-LEVEL PLOTTING
# ============================================================================
# Target Audience: Data Scientists, Quantitative Researchers, AI Engineers.
# Topic: Figure-level FacetGrids, Axes-level subplots, KDE estimation, 
#        Regression Diagnostics (Residual plotting), and Matplotlib integration.
# Compatibility: Python 3.10+, Seaborn 0.11+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Seaborn?
   - Seaborn is a data visualization library built on top of Matplotlib. It makes it 
     easy to generate beautiful, informative, and structurally complex statistical plots 
     with minimal code.
2. What is Matplotlib?
   - Matplotlib is the underlying, low-level plotting engine in Python. It provides 
     objects like Figures (the overall window/canvas) and Axes (the individual plots/grids 
     inside the canvas).
3. What is a Subplot?
   - Subplots allow you to draw multiple separate axes (plots) next to each other inside 
     a single figure, which is necessary for comparing different features of a dataset 
     side-by-side.
"""



import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to ensure execution safety in CLI/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ============================================================================
# 1. UNDER THE HOOD OF SEABORN: FIGURE-LEVEL VS AXES-LEVEL FUNCTIONS
# ============================================================================
"""
Seaborn splits its plotting ecosystem into two distinct families of functions. 
Understanding this separation is essential for customizing multi-plot layouts:

1. AXES-LEVEL FUNCTIONS (e.g. `sns.scatterplot`, `sns.boxplot`, `sns.histplot`):
   - These draw directly onto a pre-existing Matplotlib Axes (`ax`) object.
   - They accept an optional `ax` parameter. If omitted, they plot onto the 
     currently active axes (`plt.gca()`).
   - They are highly flexible and integrate perfectly into custom multi-panel 
     subplots created via `plt.subplots(rows, cols)`.

2. FIGURE-LEVEL FUNCTIONS (e.g. `sns.relplot`, `sns.catplot`, `sns.displot`, `sns.lmplot`):
   - These manage their own Matplotlib Figure and do NOT accept an `ax` parameter.
   - Under the hood, they instantiate and return a Seaborn `FacetGrid` object.
   - They are specifically designed to support "faceting" (splitting a dataset 
     across multiple subplots based on categorical factors like `col='Store'`).
"""

def demonstrate_axes_level_subplots():
    print("\n" + "="*80)
    print("1. DEMONSTRATING AXES-LEVEL SUBPLOTS")
    print("="*80)

    # Create dummy data
    df = pd.DataFrame({
        "X": np.random.randn(100),
        "Y": np.random.randn(100),
        "Group": np.random.choice(["A", "B"], size=100)
    })

    # Create a 1-row, 2-column Matplotlib layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot onto ax1 and ax2 using AXES-LEVEL functions
    sns.scatterplot(data=df, x="X", y="Y", hue="Group", ax=ax1)
    ax1.set_title("Axes 1: Scatterplot")

    sns.boxplot(data=df, y="X", x="Group", ax=ax2)
    ax2.set_title("Axes 2: Boxplot")

    # Save figure
    fig.tight_layout()
    fig.savefig("axes_level_subplots.png")
    plt.close(fig)
    print("  [SUCCESS] Plotted and saved 'axes_level_subplots.png'")


# ============================================================================
# 2. STATISTICAL REGRESSION & DIAGNOSTIC RESIDUAL PLOTS
# ============================================================================
"""
In quantitative finance and classical machine learning, plotting regressions 
requires evaluating prediction errors (residuals):
- Residual: $e_i = y_i - \hat{y}_i$.
- Homoscedasticity: Ideally, residuals should be distributed randomly around 
  a horizontal baseline ($e=0$) with constant variance across all values of $X$.
- Heteroscedasticity: If residual spread forms a funnel shape (variance scales with $X$), 
  standard linear regression assumptions are violated, and standard errors are biased!
- Non-linearity: If residuals form a parabolic curve, there is an unmodeled non-linear 
  relationship.
"""

def demonstrate_residual_diagnostics():
    print("\n" + "="*80)
    print("2. REGRESSION DIAGNOSTICS & RESIDUALS")
    print("="*80)

    # 1. Generate data with Heteroscedasticity (increasing variance)
    np.random.seed(42)
    X = np.linspace(1, 10, 150)
    # Variance increases as X grows (multiplied by X)
    noise = np.random.randn(150) * (X * 0.3)
    y = 2.0 * X + noise

    df = pd.DataFrame({"X": X, "y": y})

    # Plot joint regression using sns.jointplot (Figure-Level)
    # It shows both bivariate scatter/regression line and marginal univariate distributions
    joint_grid = sns.jointplot(data=df, x="X", y="y", kind="reg", height=6)
    joint_grid.savefig("joint_regression.png")
    plt.close()
    print("  [SUCCESS] Plotted and saved 'joint_regression.png'")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the primary difference between Figure-Level and Axes-Level functions in Seaborn?
   - Answer: Figure-level functions (`relplot`, `displot`, `catplot`) control and build 
     their own multi-panel Matplotlib figures (returning `FacetGrid` objects) and cannot be passed 
     to an existing `ax`. Axes-level functions (`scatterplot`, `boxplot`) draw directly onto 
     an existing Matplotlib Axes object (`ax`), letting you compose them in standard custom grids.

2. MEDIUM: Explain what "Heteroscedasticity" looks like on a residual plot. Why is it a problem?
   - Answer: Heteroscedasticity appears on a residual plot when the vertical spread of residuals 
     is non-constant (e.g., forming a widening funnel shape as the predicted value increases). 
     This indicates that the model's prediction variance depends on the input values, 
     violating ordinary least squares (OLS) regression assumptions and biasing hypothesis testing.

3. HARD: How does Seaborn's `pairplot` compile bivariate distributions under the hood?
   - Answer: `pairplot` instantiates a `PairGrid` object containing an $N \times N$ grid of axes 
     where $N$ is the number of numerical features. It iterates over all coordinate pairs:
     - Off-diagonal axes (row != col): Plots pairwise bivariate relationships (e.g. `scatterplot`).
     - Diagonal axes (row == col): Plots univariate marginal distributions (e.g. `histplot` or `kdeplot`).

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `compute_and_plot_residuals(X, y_true, y_pred)` that 
   calculates residuals ($e = y_{true} - y_{pred}$), instantiates a Matplotlib figure, 
   plots a scatterplot of predictions vs residuals, draws a horizontal baseline at $y=0$, 
   and saves the output.
   - Code Template to execute:
"""

def compute_and_plot_residuals(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> np.ndarray:
    # 1. Calculate residuals: e = y_true - y_pred
    residuals = y_true - y_pred
    
    # 2. Initialize Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 3. Draw scatterplot of y_pred vs residuals using axes-level scatterplot
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.7)
    
    # 4. Draw horizontal baseline at y=0
    ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
    
    # Set styling details
    ax.set_title("Residual Analysis Plot")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (True - Pred)")
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    
    return residuals


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_axes_level_subplots()
    demonstrate_residual_diagnostics()
    
    # Testing residual calculation and plot
    print("\nTesting Custom Residual Plotter:")
    X_test = np.linspace(1, 10, 50)
    y_true = 3.0 * X_test + 2.0 + np.random.randn(50) # Linear trend with normal noise
    y_pred = 3.0 * X_test + 2.0                     # Optimal regression prediction
    
    res = compute_and_plot_residuals(X_test, y_true, y_pred, "test_residuals.png")
    print(f"  Calculated Mean Residual: {np.mean(res):.4f} (Expected near 0.0)")
    print("  [SUCCESS] Plotted and saved 'test_residuals.png'")