"""
================================================================================
COMPREHENSIVE TUTORIAL: TIME & SPACE COMPLEXITY IN ALGORITHMIC INTERVIEWS
Special Focus: The 0/1 Knapsack Problem Progression & Classic Interview Patterns
================================================================================

TABLE OF CONTENTS:
1. CORE PRINCIPLES OF TIME & SPACE COMPLEXITY
   - Big-O Defined
   - Auxiliary Space vs. Total Space
   - The Call Stack & Recursion Depth
   - Why Time and Space Often Diverge (The Time-Space Trade-off)

2. DEEP DIVE: 0/1 KNAPSACK PROBLEM (Step-by-Step Evolution)
   - [Naive Approach] Recursion: O(2^n) Time | O(n) Space
   - [Better Approach 1] Top-Down DP (Memoization): O(n * W) Time | O(n * W) Space
   - [Better Approach 2] Bottom-Up DP (Tabulation): O(n * W) Time | O(n * W) Space
   - [Expected Approach] Bottom-Up DP (Space-Optimized): O(n * W) Time | O(W) Space
   - IN-DEPTH BREAKDOWN: Why Time & Space Change Between These Approaches

3. ESSENTIAL INTERVIEW PROBLEMS & TRADEOFF PATTERNS
   - Problem A: Fibonacci Numbers (O(2^n) -> O(n) -> O(1) Space)
   - Problem B: Two Sum (Time-Space Tradeoff: O(n^2)/O(1) vs. O(n)/O(n))
   - Problem C: DFS vs. BFS (Tree/Graph Traversal Memory Footprint)
   - Problem D: Longest Common Subsequence (LCS 2D to 1D Row Optimization)

4. INTERVIEW CHEAT SHEET & COMMON PITFALLS
   - Python-Specific Traps (Slicing, String Concatenation, Default Dicts)
   - Master Cheat Sheet for Stack vs. Heap Allocation

5. PRACTICE QUESTIONS & SELF-TEST (With Detailed Explanations)
================================================================================
"""

from typing import List, Dict, Tuple


# ==============================================================================
# SECTION 1: CORE PRINCIPLES OF TIME & SPACE COMPLEXITY
# ==============================================================================
"""
TIME COMPLEXITY:
- Measures how the execution time grows as the input size (n) increases.
- We count primitive operations (comparisons, additions, memory writes), not clock
  time, because clock time varies by hardware and OS load.
- In Big-O, we only retain the fastest-growing term and drop constant multipliers:
    O(3n^2 + 5n + 100) -> O(n^2)

SPACE COMPLEXITY:
- Measures the *peak additional memory* required by an algorithm relative to input size.
- In interview settings:
    TOTAL SPACE = Input Space + Auxiliary Space
    AUXILIARY SPACE = Extra memory created by the algorithm (variables, arrays, call stack).
  *Unless specified otherwise, interviewers usually ask for AUXILIARY space.*

WHY TIME AND SPACE DIVERGE:
1. Dynamic Programming / Caching:
   - We precompute or store answers to subproblems in memory (hash map / array).
   - This drastically cuts TIME (from exponential to polynomial) by spending SPACE.
2. Call Stack vs. Work Done:
   - A recursive function might branch into billions of total operations (High Time),
     but if it only goes down a single path of depth n before returning, the peak stack
     depth is only n frames (Low Space).
3. Rolling Memory Buffers:
   - When calculating row `i` in DP, if we only need values from row `i-1`, we can
     discard rows `0` through `i-2`. Time remains identical, but space drops by an entire dimension.
"""


# ==============================================================================
# SECTION 2: THE 0/1 KNAPSACK PROBLEM
# ==============================================================================
"""
PROBLEM DEFINITION:
You are given `n` items, each with:
- a weight: weights[i]
- a value: values[i]
You have a bag with a maximum weight capacity `W`.
You can either take an item (1) or leave it (0). You cannot divide an item.
Goal: Maximize the total value without exceeding capacity `W`.
"""

# ------------------------------------------------------------------------------
# [Naive Approach] Using Recursion
# Time Complexity:  O(2^n)
# Space Complexity: O(n) (Auxiliary call-stack space)
# ------------------------------------------------------------------------------
def knapsack_recursive(weights: List[int], values: List[int], W: int, n: int) -> int:
    """
    At each item `n-1`, we have two mutually exclusive choices:
    1. Exclude the item: knapsack(n-1, current_W)
    2. Include the item (if weight <= current_W): values[n-1] + knapsack(n-1, current_W - weights[n-1])

    TIME EXPLANATION:
    - We have `n` items. At every item, the algorithm branches into 2 recursive calls.
    - The recursion tree has a height of `n`.
    - Total calls = 2^0 + 2^1 + 2^2 + ... + 2^n = 2^(n+1) - 1 = O(2^n).

    SPACE EXPLANATION:
    - No auxiliary tables or lists are allocated.
    - However, function calls are placed onto the SYSTEM CALL STACK.
    - Python must keep track of active function frames.
    - The maximum depth of the tree before reaching base cases is `n`.
    - Although there are 2^n total function calls, only at most `n` frames
      exist concurrently in memory at any single moment along any single branch.
    - Therefore, Peak Auxiliary Space = O(n).
    """
    # Base case: No items left or remaining capacity is 0
    if n == 0 or W == 0:
        return 0

    # If item is heavier than remaining capacity, it cannot be included
    if weights[n - 1] > W:
        return knapsack_recursive(weights, values, W, n - 1)

    # Return maximum of:
    # (a) Include the item
    # (b) Exclude the item
    include = values[n - 1] + knapsack_recursive(weights, values, W - weights[n - 1], n - 1)
    exclude = knapsack_recursive(weights, values, W, n - 1)
    return max(include, exclude)


# ------------------------------------------------------------------------------
# [Better Approach - 1] Using Top-Down DP (Memoization)
# Time Complexity:  O(n * W)
# Space Complexity: O(n * W) (Memo table) + O(n) (Call stack) = O(n * W)
# ------------------------------------------------------------------------------
def knapsack_memoization(weights: List[int], values: List[int], W: int) -> int:
    """
    WHY DID NAIVE RECURSION EXPLODE?
    The same subproblem `(n, W)` is computed over and over across different branches.
    Memoization caches the result of each unique state `(i, current_w)`.

    TIME EXPLANATION:
    - Unique states: `n` items x `(W + 1)` possible capacities = (n * W) states.
    - Each state is computed from scratch exactly ONCE.
    - Every subsequent call with the same `(i, current_w)` is an O(1) dictionary lookup.
    - Transitions per state involve 2 lookups and a max operation = O(1).
    - Total Time = (Number of unique states) * (Work per state) = O(n * W) * O(1) = O(n * W).

    SPACE EXPLANATION:
    - Auxiliary Memory has two components:
      1. Memo table / dictionary: Holds up to `n * W` entries -> O(n * W) on the heap.
      2. Call stack: Max recursion depth is still `n` -> O(n) on the stack.
    - Total Auxiliary Space = O(n * W) + O(n) = O(n * W) (dominant term).
    """
    memo: Dict[Tuple[int, int], int] = {}

    def helper(i: int, rem_w: int) -> int:
        if i == 0 or rem_w == 0:
            return 0
        if (i, rem_w) in memo:
            return memo[(i, rem_w)]

        if weights[i - 1] > rem_w:
            res = helper(i - 1, rem_w)
        else:
            include = values[i - 1] + helper(i - 1, rem_w - weights[i - 1])
            exclude = helper(i - 1, rem_w)
            res = max(include, exclude)

        memo[(i, rem_w)] = res
        return res

    return helper(len(weights), W)


# ------------------------------------------------------------------------------
# [Better Approach - 2] Using Bottom-Up DP (Tabulation)
# Time Complexity:  O(n * W)
# Space Complexity: O(n * W) (2D Grid, zero call stack)
# ------------------------------------------------------------------------------
def knapsack_tabulation(weights: List[int], values: List[int], W: int) -> int:
    """
    Iterative approach: Eliminate recursion completely by building a 2D table
    dp[i][w], representing the max value using the first `i` items with capacity `w`.

    TIME EXPLANATION:
    - Outer loop runs `n` times (from 1 to n).
    - Inner loop runs `W` times (from 1 to W).
    - Inside the inner loop: an if-check and an assignment = O(1).
    - Total operations: n * W iterations -> O(n * W) Time.

    SPACE EXPLANATION:
    - We allocate a 2D matrix of dimensions `(n + 1) x (W + 1)`.
    - Memory allocated = (n + 1) * (W + 1) * sizeof(int).
    - NO function call stack overhead because this is purely iterative.
    - Total Auxiliary Space = O(n * W).
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],                                # Exclude item i-1
                    values[i - 1] + dp[i - 1][w - weights[i - 1]] # Include item i-1
                )
            else:
                dp[i][w] = dp[i - 1][w]                          # Cannot include

    return dp[n][W]


# ------------------------------------------------------------------------------
# [Expected Approach] Using Bottom-Up DP (Space-Optimized)
# Time Complexity:  O(n * W)
# Space Complexity: O(W) (Single 1D array)
# ------------------------------------------------------------------------------
def knapsack_space_optimized(weights: List[int], values: List[int], W: int) -> int:
    """
    KEY INSIGHT TO OPTIMIZE SPACE:
    Notice the recurrence relation:
        dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])

    To compute row `i`, we ONLY ever look at row `i - 1`!
    We never look back at row `i - 2`, `i - 3`, etc.
    Therefore, keeping all `n` rows in memory is completely redundant.

    WHY REVERSE LOOP FOR CAPACITY?
    - If we iterate capacity `w` from 0 to W forwards using a single 1D array,
      `dp[w - weights[i-1]]` would already have been overwritten with the CURRENT
      item's result, effectively allowing the same item to be picked MULTIPLE times
      (which is Unbounded Knapsack, not 0/1 Knapsack!).
    - By iterating `w` in REVERSE (from W down to weights[i-1]), `dp[w - weights[i-1]]`
      still holds the value from the PREVIOUS item (the previous row `i - 1`).

    TIME EXPLANATION:
    - Outer loop still iterates `n` times.
    - Inner loop iterates `W` times.
    - Total iterations = n * W -> O(n * W) Time (Time is NOT reduced, only space is).

    SPACE EXPLANATION:
    - We only allocate a single 1D array `dp` of size `W + 1`.
    - No 2D array, no call stack.
    - Peak Auxiliary Space = O(W).
    """
    n = len(weights)
    dp = [0] * (W + 1)

    for i in range(n):
        curr_weight = weights[i]
        curr_value = values[i]
        # Iterate backwards down to curr_weight
        for w in range(W, curr_weight - 1, -1):
            dp[w] = max(dp[w], curr_value + dp[w - curr_weight])

    return dp[W]


# ==============================================================================
# SUMMARY TABLE: 0/1 KNAPSACK COMPLEXITY EVOLUTION
# ==============================================================================
"""
+-----------------------------+--------------+---------------+-----------------------------------+
| Approach                    | Time         | Aux Space     | Where the Space Lives             |
+-----------------------------+--------------+---------------+-----------------------------------+
| 1. Naive Recursion          | O(2^n)       | O(n)          | System Call Stack only            |
| 2. Top-Down (Memoization)   | O(n * W)     | O(n * W)      | Heap (memo dict) + Stack (depth n)|
| 3. Bottom-Up (Tabulation)   | O(n * W)     | O(n * W)      | Heap (2D table); Stack is O(1)    |
| 4. Space-Optimized DP       | O(n * W)     | O(W)          | Heap (1D array); Stack is O(1)    |
+-----------------------------+--------------+---------------+-----------------------------------+

CRITICAL INTERVIEW TAKEAWAY:
- "Why is naive recursion space O(n) while time is O(2^n)?"
  Because Time accounts for the SUM of all nodes visited in the tree (2^n work).
  Space accounts for the MAXIMUM CONCURRENT DEPTH active at any point in time (n frames).
- "Why does space-optimized DP maintain O(n*W) time?"
  Because you still have to evaluate every item against every weight subproblem.
  Space optimization changes where you STORE results, not how many transitions
  you have to compute.
"""


# ==============================================================================
# SECTION 3: OTHER CLASSIC INTERVIEW COMPLEXITY PATTERNS
# ==============================================================================

# ------------------------------------------------------------------------------
# PROBLEM A: FIBONACCI NUMBERS / CLIMBING STAIRS
# Progression: O(2^n) / O(n) -> O(n) / O(n) -> O(n) / O(1)
# ------------------------------------------------------------------------------
def fib_naive(n: int) -> int:
    """Time: O(2^n) | Space: O(n) (Call stack)"""
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

def fib_memo(n: int, memo: Dict[int, int] = None) -> int:
    """Time: O(n) | Space: O(n) (Dict of size n + Call stack of depth n)"""
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

def fib_constant_space(n: int) -> int:
    """
    Time: O(n) | Space: O(1)
    Only two variables `prev2` and `prev1` are kept in memory.
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1


# ------------------------------------------------------------------------------
# PROBLEM B: TWO SUM
# Showing Time-Space Tradeoffs between Brute Force, Two Pointers, and Hash Map
# ------------------------------------------------------------------------------
def two_sum_brute_force(nums: List[int], target: int) -> List[int]:
    """
    Time: O(n^2) - Nested loops comparing every pair.
    Space: O(1) - Only indices stored.
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_hash_map(nums: List[int], target: int) -> List[int]:
    """
    Time: O(n) - Single pass through the list with O(1) hash map lookups.
    Space: O(n) - Hash map storing up to `n` entries.

    INTERVIEW TRADEOFF NOTE:
    We traded O(n) auxiliary memory to bring time complexity down from O(n^2) to O(n).
    """
    seen: Dict[int, int] = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# ------------------------------------------------------------------------------
# PROBLEM C: BFS VS. DFS (GRAPH / TREE TRAVERSAL MEMORY PROFILES)
# ------------------------------------------------------------------------------
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def dfs_tree(root: TreeNode) -> None:
    """
    DFS uses the System Stack (or an explicit LIFO Stack).
    - Time:  O(V) where V is number of vertices/nodes.
    - Space: O(h) where `h` is the HEIGHT of the tree.
      * Best/Balanced Case: h = log(V) -> Space: O(log V)
      * Worst Case (Skewed tree / linked list): h = V -> Space: O(V)
    """
    if not root:
        return
    dfs_tree(root.left)
    dfs_tree(root.right)

def bfs_tree(root: TreeNode) -> None:
    """
    BFS uses a FIFO Queue.
    - Time:  O(V)
    - Space: O(w) where `w` is the maximum WIDTH of the tree.
      * For a full balanced binary tree, the last level has V/2 nodes.
      * Therefore, BFS Space on balanced tree = O(V/2) = O(V).

    INTERVIEW COMPARISON:
    - On a wide balanced tree: DFS is significantly more memory-efficient (O(log V) vs O(V)).
    - On a degenerate single-line tree: DFS uses O(V) stack, BFS uses O(1) queue.
    """
    from collections import deque
    if not root:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


# ------------------------------------------------------------------------------
# PROBLEM D: LONGEST COMMON SUBSEQUENCE (LCS)
# Demonstrating the 2-Row Space Optimization Pattern
# ------------------------------------------------------------------------------
def lcs_2d(text1: str, text2: str) -> int:
    """
    Standard Bottom-Up LCS:
    Time:  O(m * n) where m = len(text1), n = len(text2)
    Space: O(m * n) for 2D array
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def lcs_space_optimized(text1: str, text2: str) -> int:
    """
    Optimized LCS:
    Since row `i` only refers to row `i - 1`, we only need 2 rows in memory:
    `prev` and `curr`.

    Time:  O(m * n)
    Space: O(n) (or O(min(m, n)) by ensuring the shorter string forms the columns)
    """
    # Ensure text2 is the shorter string to minimize array width
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = max(prev[j], curr[j - 1])
        # Move current row to prev for next iteration
        prev = curr[:]

    return prev[n]


# ==============================================================================
# SECTION 4: INTERVIEW CHEAT SHEET & PYTHON COMPLEXITY TRAPS
# ==============================================================================
"""
CRITICAL PYTHON TRAPS TO WATCH OUT FOR IN INTERVIEWS:

1. SLICING CREATES A COPY:
   ```python
   def recurse(arr):
       if not arr: return
       recurse(arr[1:])  # TRAP! arr[1:] copies k elements -> O(k) time & O(k) space!
Fix: Pass integer pointers (index) instead of slicing arrays!

STRING CONCATENATION IN A LOOP:

s = ""
for ch in long_string:
    s += ch  # TRAP! Strings are immutable in Python. This creates a new string
             # each step -> O(n^2) total time!
Fix: Append to a list and use ''.join(list) -> O(n) time & O(n) space.

IN-PLACE VS OUT-OF-PLACE METHODS:

list.sort(): In-place, returns None, uses Timsort (O(n) worst-case auxiliary space).
sorted(list): Out-of-place, returns a brand new list -> O(n) memory.
RECURSION DEPTH LIMIT:

Python default recursion limit is typically 1000.
If recursion depth is O(n) and n = 10^5, it will crash with RecursionError.
In interviews, mention: "Tabulation avoids Python's recursion limit."
"""

# ==============================================================================
# SECTION 5: INTERVIEW QUESTIONS & PRACTICE EXERCISES
# ==============================================================================
"""
QUESTION 1:
Q: "Given a recursive algorithm with no loops, can it still exceed memory limits?"
A: YES. Every function call allocates a stack frame containing return address,
parameters, and local variables. If recursion depth reaches 10^5 without tail-call
optimization (which Python does not support), it will exhaust stack memory or
hit the recursion limit.

QUESTION 2:
Q: "In 0/1 Knapsack, why can't we optimize the space to O(1) like in Fibonacci?"
A: In Fibonacci, the transition depends only on the 2 immediately preceding numbers:
F(n) = F(n-1) + F(n-2).
In 0/1 Knapsack, state w depends on w - weight[i]. Because weight[i] can be
ANY value from 1 to W, any previous column might be needed. Hence, we must maintain
the entire row of length W in memory.

QUESTION 3:
Q: "What is the difference between 0/1 Knapsack and Unbounded Knapsack in terms of 1D DP?"
A: - 0/1 Knapsack: Inner loop runs BACKWARDS (W down to weight[i]) so each item is used at most once.

Unbounded Knapsack (Coin Change): Inner loop runs FORWARDS (weight[i] up to W) because
re-using the same item multiple times in the same iteration is permitted.
QUESTION 4:
Q: "Is O(n * W) considered Polynomial time?"
A: It is PSEUDO-POLYNOMIAL.
The input size of integer W is the number of BITS needed to represent it (log2 W).
If W doubles, its bit length increases by only 1, but the algorithm's running
time doubles. If W is astronomically large (e.g. 10^12), O(n * W) will TLE (Time Limit Exceeded).
"""

# ==============================================================================
# SECTION 6: VERIFICATION RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING 0/1 KNAPSACK COMPARISONS")
    print("=" * 70)

    # Test Data
    test_weights = [2, 3, 4, 5]
    test_values = [3, 4, 5, 6]
    test_capacity = 5
    num_items = len(test_weights)

    ans_rec = knapsack_recursive(test_weights, test_values, test_capacity, num_items)
    ans_memo = knapsack_memoization(test_weights, test_values, test_capacity)
    ans_tab = knapsack_tabulation(test_weights, test_values, test_capacity)
    ans_opt = knapsack_space_optimized(test_weights, test_values, test_capacity)

    print(f"Weights:  {test_weights}")
    print(f"Values:   {test_values}")
    print(f"Capacity: {test_capacity}\n")
    print(f"[Naive Recursion]        Result: {ans_rec} | Time: O(2^n)   | Space: O(n)")
    print(f"[Top-Down Memoization]   Result: {ans_memo} | Time: O(n*W)   | Space: O(n*W)")
    print(f"[Bottom-Up Tabulation]   Result: {ans_tab} | Time: O(n*W)   | Space: O(n*W)")
    print(f"[Bottom-Up 1D Optimized] Result: {ans_opt} | Time: O(n*W)   | Space: O(W)")

    assert ans_rec == ans_memo == ans_tab == ans_opt == 7, "All implementations must agree on 7!"
    print("\nAll Knapsack approaches verified successfully!")

    print("\n" + "=" * 70)
    print("VERIFYING OTHER PROBLEMS")
    print("=" * 70)

    # Fibonacci Verification
    assert fib_naive(10) == fib_memo(10) == fib_constant_space(10) == 55
    print("Fibonacci O(2^n) -> O(n) -> O(1) Space verified.")

    # Two Sum Verification
    nums = [2, 7, 11, 15]
    target = 9
    assert two_sum_brute_force(nums, target) == [0, 1]
    assert two_sum_hash_map(nums, target) == [0, 1]
    print("Two Sum O(n^2)/O(1) vs O(n)/O(n) verified.")

    # LCS Verification
    s1, s2 = "abcde", "ace"
    assert lcs_2d(s1, s2) == lcs_space_optimized(s1, s2) == 3
    print("LCS 2D O(m*n) to 2-Row O(min(m, n)) space verified.")

    print("\nEverything is working as expected. You are ready for interviews!")
