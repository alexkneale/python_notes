# ============================================================================
# MATHEMATICAL MECHANICS: GBDTS & CONSTRAINED OPTIMIZATION
# ============================================================================
# Target Audience: Applied Mathematicians, Quantitative Developers, ML Researchers.
# Topic: Tree Split Mathematics, Second-Order Taylor GBDT Engine (XGBoost Math),
#        and Constrained Optimization (Quadratic Programming via SciPy).
# Compatibility: Python 3.10+, NumPy 1.22+, SciPy 1.8+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is a Decision Tree?
   - A decision tree is a flowchart-like model that splits data into smaller groups 
     based on simple yes/no questions (e.g., "Is age > 30?"). It does this until it 
     reaches the leaves, which provide the final prediction.
2. What is Gradient Boosting?
   - Gradient Boosting is an ensemble method. Instead of training one complex model, it 
     trains many simple decision trees sequentially. Each new tree is trained to predict 
     and correct the errors (residuals) made by the combination of all previous trees.
3. What is Constrained Optimization?
   - Optimization is finding the best inputs to maximize or minimize a function (like profit 
     or risk). Constrained optimization forces the solver to obey specific rules (like 
     "sum of portfolio investments must equal exactly 100%").
"""



import numpy as np
import scipy.optimize as opt

# ============================================================================
# 1. THE MATHEMATICAL MECHANICS OF GRADIENT BOOSTING (XGBOOST ENGINE)
# ============================================================================
"""
Modern Gradient Boosting frameworks (XGBoost, LightGBM) optimize a custom objective 
using a second-order Taylor expansion approximation.

Let the regularized objective at iteration 't' be:
    Obj^(t) = sum_{i=1}^N L(y_i, y_hat_i^(t-1) + f_t(x_i)) + Omega(f_t)

where f_t is the new tree we add to the ensemble, and Omega(f_t) is the complexity 
regularization penalty:
    Omega(f_t) = gamma * T + 0.5 * lambda * sum_{j=1}^T w_j^2
(T is the number of leaves, and w_j is the score/weight of leaf j).

Taking the second-order Taylor expansion of the loss L:
    Obj^(t) approx sum_{i=1}^N [ L(y_i, y_hat_i^(t-1)) + g_i * f_t(x_i) + 0.5 * h_i * f_t(x_i)^2 ] + Omega(f_t)

where:
    g_i = d(L(y_i, y_hat_i^(t-1))) / d(y_hat_i^(t-1))   <-- First-order gradient
    h_i = d^2(L(y_i, y_hat_i^(t-1))) / d(y_hat_i^(t-1))^2 <-- Second-order Hessian

By grouping the sums by leaf node j (letting I_j be the set of sample indices in leaf j):
    Obj^(t) approx sum_{j=1}^T [ (sum_{i in I_j} g_i) * w_j + 0.5 * (sum_{i in I_j} h_i + lambda) * w_j^2 ] + gamma * T

Finding Optimal Leaf Weights (w_j*):
By setting the derivative of Obj^(t) with respect to w_j to 0, we solve the optimal weight:
    w_j* = - (sum_{i in I_j} g_i) / (sum_{i in I_j} h_i + lambda)

Substituting w_j* back into the objective, we get the optimal loss reduction score (quality score) of a structure:
    Obj*_opt = -0.5 * sum_{j=1}^T [ (sum_{i in I_j} g_i)^2 / (sum_{i in I_j} h_i + lambda) ] + gamma * T

Deriving the Split Gain:
When evaluating a potential split on a leaf into a Left (L) and Right (R) node:
    Gain = 0.5 * [ (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - (G_Parent^2 / (H_Parent + lambda)) ] - gamma
where:
    G_L = sum_{i in I_L} g_i ,  H_L = sum_{i in I_L} h_i
This formula is the core metric used by XGBoost to determine optimal splits!
"""

class MathCustomGBDTSplitter:
    """
    Python demonstration class showing how XGBoost calculates optimal split gain 
    manually from raw gradients and Hessians.
    """
    def __init__(self, reg_lambda: float = 1.0, min_split_gain_gamma: float = 0.1):
        self.reg_lambda = reg_lambda
        self.gamma = min_split_gain_gamma

    def calculate_leaf_weight(self, sum_gradients: float, sum_hessians: float) -> float:
        """w_j* = - G_j / (H_j + lambda)"""
        return - sum_gradients / (sum_hessians + self.reg_lambda)

    def calculate_structure_score(self, sum_gradients: float, sum_hessians: float) -> float:
        """Structure score = G^2 / (H + lambda)"""
        return (sum_gradients ** 2) / (sum_hessians + self.reg_lambda)

    def evaluate_split(self, grad_left: np.ndarray, hess_left: np.ndarray, 
                       grad_right: np.ndarray, hess_right: np.ndarray) -> dict:
        """
        Evaluates the potential split gain.
        """
        G_L, H_L = np.sum(grad_left), np.sum(hess_left)
        G_R, H_R = np.sum(grad_right), np.sum(hess_right)
        
        # Parent node totals
        G_P, H_P = G_L + G_R, H_L + H_R
        
        # Calculate structure scores
        score_left = self.calculate_structure_score(G_L, H_L)
        score_right = self.calculate_structure_score(G_R, H_R)
        score_parent = self.calculate_structure_score(G_P, H_P)
        
        # XGBoost Split Gain Formula
        gain = 0.5 * (score_left + score_right - score_parent) - self.gamma
        
        return {
            "gain": gain,
            "weight_left": self.calculate_leaf_weight(G_L, H_L),
            "weight_right": self.calculate_leaf_weight(G_R, H_R),
            "is_split_viable": gain > 0.0
        }


def demonstrate_gbdt_math():
    print("\n" + "="*80)
    print("1. THE MATHEMATICAL MECHANICS OF XGBOOST SPLIT GAINS")
    print("="*80)

    # Let's say we are performing binary classification with Cross-Entropy Loss
    # L(y, p) = - [ y*log(p) + (1-y)*log(1-p) ]
    # Gradient g_i = p_i - y_i
    # Hessian  h_i = p_i * (1 - p_i)
    
    # 5 samples reach a node: 3 belonging to Class 1, 2 to Class 0
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.85, 0.1, 0.6]) # Predictions from prior trees

    gradients = p - y
    hessians = p * (1 - p)

    splitter = MathCustomGBDTSplitter(reg_lambda=1.0, min_split_gain_gamma=0.05)

    # Scenario 1: A potential split that groups high-probability together (Good Split)
    # Left Child: first 3 samples (mostly class 1)
    # Right Child: last 2 samples (mostly class 0)
    split_info_good = splitter.evaluate_split(
        grad_left=gradients[:3], hess_left=hessians[:3],
        grad_right=gradients[3:], hess_right=hessians[3:]
    )
    print("Good Split Evaluation (Separating classes correctly):")
    print(f"  Calculated Split Gain: {split_info_good['gain']:.6f}")
    print(f"  Is split viable?        {split_info_good['is_split_viable']}")
    print(f"  Left leaf weight (w_L): {split_info_good['weight_left']:.4f}")
    print(f"  Right leaf weight(w_R): {split_info_good['weight_right']:.4f}")

    # Scenario 2: A bad split that scrambles instances randomly
    # Left Child: indices [0, 3]
    # Right Child: indices [1, 2, 4]
    split_info_bad = splitter.evaluate_split(
        grad_left=gradients[[0, 3]], hess_left=hessians[[0, 3]],
        grad_right=gradients[[1, 2, 4]], hess_right=hessians[[1, 2, 4]]
    )
    print("\nBad Split Evaluation (Random mixture of labels):")
    print(f"  Calculated Split Gain: {split_info_bad['gain']:.6f}")
    print(f"  Is split viable?        {split_info_bad['is_split_viable']} (Rejected due to Gamma regularization constraint!)")


# ============================================================================
# 2. CONSTRAINED NUMERICAL OPTIMIZATION (SCIPY MINIMIZE)
# ============================================================================
"""
In operations research and quantitative finance, we solve constrained optimization problems.
Let's implement a Markowitz Portfolio Optimization to solve for asset weights 'w' that 
minimize the overall Portfolio Variance (Risk) under specific linear constraints:

Objective:
    minimize  0.5 * w^T * Sigma * w
where Sigma is the Covariance Matrix of Asset Returns.

Constraints:
1. Linear Equality Constraint: Weights must sum to 1: sum(w_i) = 1
2. Bounds (No short-selling): 0 <= w_i <= 1 for all i.
"""

def solve_portfolio_optimization():
    print("\n" + "="*80)
    print("2. CONSTRAINED NUMERICAL OPTIMIZATION (PORTFOLIO VARIANCE MINIMIZATION)")
    print("="*80)

    # 1. Define asset covariance matrix Sigma (3 assets: e.g. Tech, Bonds, Gold)
    # Bonds and Gold have lower variance and negative correlation with Tech
    Sigma = np.array([[0.09, -0.01, -0.03],
                      [-0.01, 0.04,  0.01],
                      [-0.03, 0.01,  0.05]])
    
    asset_names = ["Tech Stocks", "Bonds", "Gold"]

    # 2. Objective Function: w^T * Sigma * w
    def objective_func(w):
        return 0.5 * np.dot(w.T, np.dot(Sigma, w))

    # 3. Define Constraints
    # Scipy constraints are dictionaries:
    # 'type': 'eq' (for equality constraints c(x)=0) or 'ineq' (for inequalities c(x)>=0)
    # 'fun': callable function calculating the constraint
    def weight_sum_constraint(w):
        return np.sum(w) - 1.0 # sum(w) - 1 = 0

    constraints = ({'type': 'eq', 'fun': weight_sum_constraint})

    # 4. Bounds: 0 <= w_i <= 1 for each of the 3 assets
    bounds = [(0.0, 1.0) for _ in range(3)]

    # 5. Starting point guess (Equal weight allocation)
    w0 = np.array([1.0/3, 1.0/3, 1.0/3])

    # 6. Execute Optimization (using Sequential Least Squares Programming - SLSQP)
    result = opt.minimize(
        objective_func,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    print("Optimization Status:", result.message)
    print("Success:", result.success)
    
    optimized_weights = result.x
    print("\nOptimal Risk-Minimized Portfolio Weights:")
    for name, weight in zip(asset_names, optimized_weights):
        print(f"  {name:12s}: {weight * 100:.2f}%")
        
    portfolio_variance = result.fun * 2.0
    print(f"\nMinimum Portfolio Variance: {portfolio_variance:.6f}")
    print(f"Portfolio Standard Deviation (Volatility): {np.sqrt(portfolio_variance) * 100:.2f}%")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is Gini Impurity, and how is it used to find tree splits?
   - Answer: Gini Impurity measures the probability of misclassifying a randomly chosen 
     element if it were labeled according to the distribution of labels in the node.
     Formula: $I_G(p) = 1 - \sum p_i^2$. At each candidate split, we search for the boundary 
     that maximizes the reduction in Gini Impurity (Gain = Parent_Gini - [Left_Weight * Left_Gini 
     + Right_Weight * Right_Gini]).

2. MEDIUM: Why does XGBoost use a second-order Taylor expansion, while standard GBM uses first-order?
   - Answer: Standard Gradient Boosting Machines (GBM) approximate the loss function 
     solely using first-order gradients (acting as a functional gradient descent). XGBoost uses 
     a second-order Taylor expansion containing both the gradient $g_i$ and the Hessian $h_i$. 
     This allows XGBoost to incorporate curvature, calculating the optimal leaf values 
     ($w_j* = -G / (H + \lambda)$) and evaluating split structures in a single cohesive pass. 
     This behaves like a Newton-Raphson optimization step, accelerating convergence.

3. HARD: How does SLSQP (Sequential Least Squares Programming) handle constraints?
   - Answer: SLSQP solves non-linear optimization problems under equality and inequality constraints. 
     It operates by solving a sequence of Quadratic Programming (QP) subproblems. At each step:
     (1) The objective is approximated by a quadratic function (using Hessian estimates).
     (2) Non-linear constraints are linearized around the current iteration point using first-order derivatives.
     (3) A least-squares solver is used to calculate the search direction that satisfies the constraints.
     (4) A line-search is executed to ensure convergence and update the optimization state.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `calculate_gini_impurity(labels)` that calculates the 
   Gini Impurity of a 1D NumPy array of binary labels (0 or 1).
   - Code Template to execute:
"""

def calculate_gini_impurity(labels: np.ndarray) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    
    # Calculate probability of class 1
    p_1 = np.sum(labels == 1) / n
    p_0 = 1.0 - p_1
    
    # Gini = 1 - (p_0^2 + p_1^2)
    return 1.0 - (p_0**2 + p_1**2)


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_gbdt_math()
    solve_portfolio_optimization()
    
    # Testing Gini
    print("\nTesting Custom Gini Impurity:")
    mixed = np.array([1, 1, 0, 0]) # 50% split -> Gini should be 1 - (0.5^2 + 0.5^2) = 0.5
    pure = np.array([1, 1, 1, 1])   # 100% split -> Gini should be 0
    print(f"  Gini (mixed): {calculate_gini_impurity(mixed)} (Expected: 0.5)")
    print(f"  Gini (pure):  {calculate_gini_impurity(pure)} (Expected: 0.0)")