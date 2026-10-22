"""
================================================================================
DYNAMIC PROGRAMMING: FIBONACCI & 1D LINEAR SEQUENCES (TOP-DOWN & BOTTOM-UP)
================================================================================

TABLE OF CONTENTS:
1. RECAP & THE CORE INTUITION: THE 1D LINEAR SEQUENCE PATTERN
2. THE ARCHETYPE (FIBONACCI): RECURSION, MEMOIZATION, TABULATION & O(1) SPACE
3. THE PROGRESSION ENGINE: HOW SUBPROBLEMS COMBINE
4. VARIATION 1: CLIMBING STAIRS (COUNTING COMBINATIONS)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
5. VARIATION 2: MIN COST CLIMBING STAIRS (MINIMIZATION)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
6. VARIATION 3: HOUSE ROBBER (NON-CONSECUTIVE SELECTION / OPTIMIZATION)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
7. VARIATION 4: DECODE WAYS (BRANCHING VALIDITY & STRING PARSING)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
8. COMPARATIVE SUMMARY TABLE: TRANSITION EQUATIONS & BASE CASES
9. RUNNABLE TESTS & VERIFICATION
================================================================================
"""

from typing import List, Dict
import math


# ==============================================================================
# SECTION 1: THE CORE INTUITION: THE 1D LINEAR SEQUENCE PATTERN
# ==============================================================================
"""
THE 1D LINEAR PATTERN EXPLAINED:
--------------------------------
In 1D Linear Sequence DP, we are navigating along a single linear dimension
(an array index, a staircase rung, or a position along a string).

Key Characteristic:
The optimal solution to reach or evaluate state `i` depends STRICTLY on a small,
fixed sliding window of previous adjacent states—most frequently `i - 1` and `i - 2`.

Because state transitions only look back `k` steps (here, `k = 1` or `k = 2`):
1. Top-Down (Memoization) requires an array or hash map of size O(N).
2. Bottom-Up Tabulation requires an array of size O(N).
3. Space Optimization: Since `dp[i]` only depends on `dp[i - 1]` and `dp[i - 2]`,
   we do NOT need to preserve the entire array of history! We can discard older
   states and retain only two scalar variables (`prev1`, `prev2`), shrinking
   auxiliary space from O(N) down to O(1).
"""


# ==============================================================================
# SECTION 2: THE ARCHETYPE: FIBONACCI NUMBERS
# ==============================================================================
"""
Definition:
    F(0) = 0
    F(1) = 1
    F(n) = F(n - 1) + F(n - 2) for n >= 2
"""

# ------------------------------------------------------------------------------
# 2.1 Pure Recursion
# ------------------------------------------------------------------------------
def fib_recursive(n: int) -> int:
    """
    Pure Recursive Solution (No Memoization).
    Time:  O(2^n) - Branching binary tree creates redundant calls.
    Space: O(n)   - Maximum recursion call stack depth.
    """
    # BASE CASES
    if n == 0:
        return 0
    if n == 1:
        return 1

    # RECURSIVE STEP
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# ------------------------------------------------------------------------------
# 2.2 Top-Down DP (Memoization)
# ------------------------------------------------------------------------------
def fib_memoization(n: int) -> int:
    """
    Top-Down DP (Memoization).
    Time:  O(n) - Each state from 0 to n is computed exactly once.
    Space: O(n) - Call stack depth O(n) + hash map storage O(n).
    """
    memo: Dict[int, int] = {}

    def helper(k: int) -> int:
        if k == 0:
            return 0
        if k == 1:
            return 1

        if k in memo:
            return memo[k]

        memo[k] = helper(k - 1) + helper(k - 2)
        return memo[k]

    return helper(n)


# ------------------------------------------------------------------------------
# 2.3 Bottom-Up DP (Tabulation)
# ------------------------------------------------------------------------------
def fib_tabulation(n: int) -> int:
    """
    Bottom-Up DP (1D Table).
    Time:  O(n) - Single linear iteration.
    Space: O(n) - Table stores all intermediate answers.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1

    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


# ------------------------------------------------------------------------------
# 2.4 Bottom-Up DP (O(1) Space-Optimized Rolling Variables)
# ------------------------------------------------------------------------------
def fib_space_optimized(n: int) -> int:
    """
    Bottom-Up DP with Rolling Variables.
    Time:  O(n)
    Space: O(1) - Only two scalar values maintained in memory.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1

    prev2 = 0  # Represents F(i - 2)
    prev1 = 1  # Represents F(i - 1)

    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1


# ==============================================================================
# SECTION 3: THE PROGRESSION ENGINE: HOW SUBPROBLEMS COMBINE
# ==============================================================================
"""
All problems in this family share the same underlying graph structure (a DAG
where each node connects to the preceding 1 or 2 nodes). What changes is the
OPERATOR used to combine subproblem answers:

1. COUNTING PATHS (Summation):
   ways[i] = ways[i - 1] + ways[i - 2]
   Seen in: Climbing Stairs, Decode Ways.

2. MINIMIZATION / MAXIMIZATION (Selection):
   best[i] = min/max(cost[i] + best[i - 1], cost[i] + best[i - 2])
   Seen in: Min Cost Climbing Stairs, House Robber.

3. CONDITIONAL / GUARDED TRANSITIONS:
   Evaluating whether a transition is syntactically or logically valid
   (e.g., checking if 2 digits form a number between "10" and "26").
   Seen in: Decode Ways.
"""


# ==============================================================================
# SECTION 4: VARIATION 1: CLIMBING STAIRS
# ==============================================================================
"""
Problem:
You are climbing a staircase. It takes `n` steps to reach the top.
Each time you can either climb 1 or 2 steps.
In how many distinct ways can you climb to the top?

State Definition:
`ways(n)` = total number of distinct ways to reach step `n`.

Choice Diagram:
To arrive at step `n`, your final step must have originated from either:
  1. Step `n - 1` (by taking a 1-step leap)
  2. Step `n - 2` (by taking a 2-step leap)
Therefore:
  ways(n) = ways(n - 1) + ways(n - 2)
"""

def climb_stairs_rec(n: int) -> int:
    """Pure Recursion: O(2^n) time, O(n) space."""
    if n <= 1:
        return 1
    return climb_stairs_rec(n - 1) + climb_stairs_rec(n - 2)


def climb_stairs_memo(n: int) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, int] = {}

    def helper(k: int) -> int:
        if k <= 1:
            return 1
        if k in memo:
            return memo[k]
        memo[k] = helper(k - 1) + helper(k - 2)
        return memo[k]

    return helper(n)


def climb_stairs_dp(n: int) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def climb_stairs_optimized(n: int) -> int:
    """Bottom-Up Rolling Variables: O(n) time, O(1) space."""
    if n <= 1:
        return 1
    prev2, prev1 = 1, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1


# ==============================================================================
# SECTION 5: VARIATION 2: MIN COST CLIMBING STAIRS
# ==============================================================================
"""
Problem:
You are given an integer array `cost` where `cost[i]` is the cost of `i-th` step.
Once you pay the cost, you can climb either 1 or 2 steps.
You can start from step 0 or step 1.
Return the MINIMUM cost to reach the top of the floor (index `len(cost)`).

State Definition:
`min_cost(i)` = minimum cost required to reach step `i`.

Choice Diagram:
To arrive at step `i`:
  - Come from `i - 1`: Pay `cost[i - 1]`
  - Come from `i - 2`: Pay `cost[i - 2]`
Transition:
  min_cost(i) = min(min_cost(i - 1) + cost[i - 1], min_cost(i - 2) + cost[i - 2])
"""

def min_cost_climbing_stairs_rec(cost: List[int]) -> int:
    """Pure Recursion: O(2^n) time, O(n) space."""
    def helper(i: int) -> int:
        # Base cases: Starting at index 0 or 1 incurs no prior cost to stand there
        if i <= 1:
            return 0
        from_one_step = helper(i - 1) + cost[i - 1]
        from_two_step = helper(i - 2) + cost[i - 2]
        return min(from_one_step, from_two_step)

    return helper(len(cost))


def min_cost_climbing_stairs_memo(cost: List[int]) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, int] = {}

    def helper(i: int) -> int:
        if i <= 1:
            return 0
        if i in memo:
            return memo[i]

        res = min(
            helper(i - 1) + cost[i - 1],
            helper(i - 2) + cost[i - 2]
        )
        memo[i] = res
        return res

    return helper(len(cost))


def min_cost_climbing_stairs_dp(cost: List[int]) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    n = len(cost)
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 0

    for i in range(2, n + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

    return dp[n]


def min_cost_climbing_stairs_optimized(cost: List[int]) -> int:
    """Bottom-Up Rolling Variables: O(n) time, O(1) space."""
    prev2 = 0
    prev1 = 0

    for i in range(2, len(cost) + 1):
        curr = min(prev1 + cost[i - 1], prev2 + cost[i - 2])
        prev2 = prev1
        prev1 = curr

    return prev1


# ==============================================================================
# SECTION 6: VARIATION 3: HOUSE ROBBER
# ==============================================================================
"""
Problem:
You are a professional robber planning to rob houses along a street.
Each house has a certain amount of money stashed.
Adjacent houses have connected security systems that trigger alarms if TWO
adjacent houses are broken into on the same night.
Given `nums` representing the money of each house, return the MAXIMUM amount
of money you can rob without alerting the police.

State Definition:
`rob(i)` = maximum loot possible considering houses from index `0` up to `i`.

Choice Diagram for House `i`:
- Option 1 (ROB HOUSE i):
    Gain `nums[i]`. Because adjacent houses cannot be robbed, the next available
    house to rob must be at index `i - 2`.
    Total = nums[i] + rob(i - 2)
- Option 2 (SKIP HOUSE i):
    Gain 0 from house `i`. Can freely rob up to house `i - 1`.
    Total = rob(i - 1)

Transition:
  rob(i) = max(nums[i] + rob(i - 2), rob(i - 1))
"""

def house_robber_rec(nums: List[int]) -> int:
    """Pure Recursion: O(2^n) time, O(n) space."""
    def helper(i: int) -> int:
        if i < 0:
            return 0
        if i == 0:
            return nums[0]

        rob_curr = nums[i] + helper(i - 2)
        skip_curr = helper(i - 1)
        return max(rob_curr, skip_curr)

    return helper(len(nums) - 1)


def house_robber_memo(nums: List[int]) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, int] = {}

    def helper(i: int) -> int:
        if i < 0:
            return 0
        if i == 0:
            return nums[0]
        if i in memo:
            return memo[i]

        rob_curr = nums[i] + helper(i - 2)
        skip_curr = helper(i - 1)
        res = max(rob_curr, skip_curr)

        memo[i] = res
        return res

    return helper(len(nums) - 1)


def house_robber_dp(nums: List[int]) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(nums[i] + dp[i - 2], dp[i - 1])

    return dp[-1]


def house_robber_optimized(nums: List[int]) -> int:
    """Bottom-Up Rolling Variables: O(n) time, O(1) space."""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(nums[i] + prev2, prev1)
        prev2 = prev1
        prev1 = curr

    return prev1


# ==============================================================================
# SECTION 7: VARIATION 4: DECODE WAYS
# ==============================================================================
"""
Problem:
A message containing letters from A-Z can be encoded into numbers using the mapping:
'A' -> "1", 'B' -> "2", ..., 'Z' -> "26".
Given a string `s` containing only digits, return the NUMBER of ways to decode it.

State Definition:
`decode(i)` = number of valid decodings for suffix starting at index `i`.

Choice Diagram at Index `i`:
- Option 1 (Single Digit):
    If `s[i] != '0'`, we can decode `s[i]` as a single letter.
    Contribution = decode(i + 1)
- Option 2 (Double Digit):
    If `s[i:i+2]` forms a valid two-digit number between 10 and 26:
    Contribution = decode(i + 2)

Transition:
  ways = 0
  if s[i] != '0': ways += decode(i + 1)
  if 10 <= int(s[i:i+2]) <= 26: ways += decode(i + 2)
"""

def decode_ways_rec(s: str) -> int:
    """Pure Recursion: O(2^n) time, O(n) space."""
    def helper(i: int) -> int:
        # If we successfully parsed the entire string, we found 1 valid decoding path
        if i == len(s):
            return 1
        # Leading zero cannot be decoded into any character
        if s[i] == '0':
            return 0

        # Decode single digit
        ways = helper(i + 1)

        # Decode two digits if valid
        if i + 1 < len(s) and (s[i] == '1' or (s[i] == '2' and s[i + 1] in "0123456")):
            ways += helper(i + 2)

        return ways

    return helper(0)


def decode_ways_memo(s: str) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, int] = {}

    def helper(i: int) -> int:
        if i == len(s):
            return 1
        if s[i] == '0':
            return 0
        if i in memo:
            return memo[i]

        ways = helper(i + 1)
        if i + 1 < len(s) and (s[i] == '1' or (s[i] == '2' and s[i + 1] in "0123456")):
            ways += helper(i + 2)

        memo[i] = ways
        return ways

    return helper(0)


def decode_ways_dp(s: str) -> int:
    """
    Bottom-Up Tabulation: O(n) time, O(n) space.
    Let dp[i] be the number of ways to decode prefix s[0:i].
    Length of dp array is len(s) + 1.
    """
    if not s or s[0] == '0':
        return 0

    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty prefix has 1 valid way
    dp[1] = 1  # First digit is guaranteed non-zero

    for i in range(2, n + 1):
        one_digit = int(s[i - 1 : i])
        two_digit = int(s[i - 2 : i])

        if 1 <= one_digit <= 9:
            dp[i] += dp[i - 1]
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]

    return dp[n]


def decode_ways_optimized(s: str) -> int:
    """Bottom-Up Rolling Variables: O(n) time, O(1) space."""
    if not s or s[0] == '0':
        return 0

    prev2 = 1  # Corresponds to dp[i - 2]
    prev1 = 1  # Corresponds to dp[i - 1]

    for i in range(2, len(s) + 1):
        curr = 0
        one_digit = int(s[i - 1 : i])
        two_digit = int(s[i - 2 : i])

        if 1 <= one_digit <= 9:
            curr += prev1
        if 10 <= two_digit <= 26:
            curr += prev2

        prev2 = prev1
        prev1 = curr

    return prev1


# ==============================================================================
# SECTION 8: COMPARATIVE SUMMARY TABLE
# ==============================================================================
"""
+-----------------------------+-------------------------------+-------------------------------------------+
| Problem                     | Subproblem Dependence         | Recurrence Relation                       |
+-----------------------------+-------------------------------+-------------------------------------------+
| Fibonacci                   | dp[i - 1], dp[i - 2]          | dp[i] = dp[i - 1] + dp[i - 2]             |
| Climbing Stairs             | dp[i - 1], dp[i - 2]          | dp[i] = dp[i - 1] + dp[i - 2]             |
| Min Cost Climbing Stairs    | dp[i - 1], dp[i - 2]          | dp[i] = min(dp[i-1]+cost[i-1],            |
|                             |                               |             dp[i-2]+cost[i-2])            |
| House Robber                | dp[i - 1], dp[i - 2]          | dp[i] = max(dp[i - 1], nums[i] + dp[i - 2)|
| Decode Ways                 | dp[i - 1], dp[i - 2] (Guarded)| dp[i] = (valid_1 ? dp[i-1] : 0) +         |
|                             |                               |         (valid_2 ? dp[i-2] : 0)           |
+-----------------------------+-------------------------------+-------------------------------------------+
"""


# ==============================================================================
# SECTION 9: RUNNABLE TESTS & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("TESTING 1D LINEAR SEQUENCE DP: RECURSION vs TOP-DOWN vs TABULATION vs O(1)")
    print("=" * 75)

    # 1. Fibonacci
    n_fib = 10
    fib_rec = fib_recursive(n_fib)
    fib_mem = fib_memoization(n_fib)
    fib_tab = fib_tabulation(n_fib)
    fib_opt = fib_space_optimized(n_fib)
    print(f"Fibonacci({n_fib}):")
    print(f"  Rec: {fib_rec} | Memo: {fib_mem} | Tab: {fib_tab} | Opt: {fib_opt}")
    assert fib_rec == fib_mem == fib_tab == fib_opt == 55
    print("  [SUCCESS] All Fibonacci variants returned 55.\n")

    # 2. Climbing Stairs
    n_stairs = 5
    st_rec = climb_stairs_rec(n_stairs)
    st_mem = climb_stairs_memo(n_stairs)
    st_tab = climb_stairs_dp(n_stairs)
    st_opt = climb_stairs_optimized(n_stairs)
    print(f"Climbing Stairs({n_stairs}):")
    print(f"  Rec: {st_rec} | Memo: {st_mem} | Tab: {st_tab} | Opt: {st_opt}")
    assert st_rec == st_mem == st_tab == st_opt == 8
    print("  [SUCCESS] All Climbing Stairs variants returned 8.\n")

    # 3. Min Cost Climbing Stairs
    cost = [10, 15, 20]
    mc_rec = min_cost_climbing_stairs_rec(cost)
    mc_mem = min_cost_climbing_stairs_memo(cost)
    mc_tab = min_cost_climbing_stairs_dp(cost)
    mc_opt = min_cost_climbing_stairs_optimized(cost)
    print(f"Min Cost Climbing Stairs (cost={cost}):")
    print(f"  Rec: {mc_rec} | Memo: {mc_mem} | Tab: {mc_tab} | Opt: {mc_opt}")
    # Cheaper to start at index 1 (pay 15) and leap 2 steps to top -> total 15
    assert mc_rec == mc_mem == mc_tab == mc_opt == 15
    print("  [SUCCESS] All Min Cost Climbing Stairs variants returned 15.\n")

    # 4. House Robber
    houses = [2, 7, 9, 3, 1]
    hr_rec = house_robber_rec(houses)
    hr_mem = house_robber_memo(houses)
    hr_tab = house_robber_dp(houses)
    hr_opt = house_robber_optimized(houses)
    print(f"House Robber (nums={houses}):")
    print(f"  Rec: {hr_rec} | Memo: {hr_mem} | Tab: {hr_tab} | Opt: {hr_opt}")
    # Optimal: rob house 0 (2), house 2 (9), house 4 (1) -> 2 + 9 + 1 = 12
    assert hr_rec == hr_mem == hr_tab == hr_opt == 12
    print("  [SUCCESS] All House Robber variants returned 12.\n")

    # 5. Decode Ways
    test_str = "226"
    dw_rec = decode_ways_rec(test_str)
    dw_mem = decode_ways_memo(test_str)
    dw_tab = decode_ways_dp(test_str)
    dw_opt = decode_ways_optimized(test_str)
    print(f"Decode Ways (s='{test_str}'):")
    print(f"  Rec: {dw_rec} | Memo: {dw_mem} | Tab: {dw_tab} | Opt: {dw_opt}")
    # Valid combinations: "2, 2, 6" (BBF), "22, 6" (VF), "2, 26" (BZ) -> 3 ways
    assert dw_rec == dw_mem == dw_tab == dw_opt == 3
    print("  [SUCCESS] All Decode Ways variants returned 3.\n")

    # Edge cases for Decode Ways
    assert decode_ways_dp("06") == 0
    assert decode_ways_dp("10") == 1
    assert decode_ways_dp("2101") == 1
    print("  [SUCCESS] Decode Ways edge cases (leading zero, valid internal zeros) verified.")

    print("\n" + "=" * 75)
    print("[ALL TESTS PASSED] 1D Linear Sequence DP module verified successfully!")
    print("=" * 75)