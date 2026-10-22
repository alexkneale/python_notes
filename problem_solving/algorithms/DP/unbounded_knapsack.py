"""
================================================================================
DYNAMIC PROGRAMMING: UNBOUNDED KNAPSACK, TOP-DOWN & BOTTOM-UP EXPLAINED
================================================================================

TABLE OF CONTENTS:
1. RECAP & THE CORE DIFFERENCE: 0/1 VS. UNBOUNDED KNAPSACK
2. THE CHOICE DIAGRAM & PURE RECURSION
3. TOP-DOWN DYNAMIC PROGRAMMING (MEMOIZATION)
   - Plain English Explanation
   - Step-by-Step Code Implementation
4. BOTTOM-UP DYNAMIC PROGRAMMING (TABULATION & 1D SPACE OPTIMIZATION)
   - Plain English Explanation (The 2D Grid vs. 1D Rolling Array)
   - Step-by-Step Code Implementation
5. DIRECT COMPARISON: 0/1 KNAPSACK VS. UNBOUNDED KNAPSACK
6. UNBOUNDED KNAPSACK EXTENSIONS & VARIATIONS
   - Variation 1: Rod Cutting Problem (Value Maximization)
   - Variation 2: Coin Change II (Count Number of Ways to Make Sum)
   - Variation 3: Coin Change I (Minimum Coins to Make Sum)
7. RUNNABLE TESTS & VERIFICATION
================================================================================
"""

from typing import List, Dict, Tuple
import math


# ==============================================================================
# SECTION 1: THE CORE DIFFERENCE (0/1 VS. UNBOUNDED)
# ==============================================================================
"""
In 0/1 Knapsack:
- You either take an item (1) or ignore it (0).
- Once an item is evaluated (taken or rejected), you MUST move on to the next
  item: `n -> n - 1`. Its availability is consumed.

In UNBOUNDED Knapsack:
- You have an INFINITE supply of each item.
- If you REJECT an item: You discard it permanently (`n -> n - 1`).
- If you TAKE an item: Its weight is deducted, its value is gained, BUT you
  remain at the EXACT SAME item (`n -> n`) so you can potentially pick it again!

Summary of Index Transitions on "TAKE":
  - 0/1 Knapsack:       take = val[n-1] + solve(capacity - wt[n-1], n - 1)
  - Unbounded Knapsack: take = val[n-1] + solve(capacity - wt[n-1], n)
"""


# ==============================================================================
# SECTION 2: THE RECURSIVE BASELINE & THE CHOICE DIAGRAM
# ==============================================================================
"""
THE CHOICE DIAGRAM FOR ITEM (n - 1):
-----------------------------------
We inspect item `n-1`:

Is weights[n-1] <= remaining capacity?
      /                          \\
    YES                           NO
   /   \\                           \\
TAKE   DON'T TAKE                 DON'T TAKE (Too heavy!)

- TAKE:
    Value gained = vals[n-1]
    Remaining capacity shrinks: bag_weight - weights[n-1]
    Next item index: STAYS AT `n` (Item can be taken again!)
- DON'T TAKE:
    Value gained = 0
    Remaining capacity remains: bag_weight
    Next item index: MOVES TO `n - 1` (Discarded forever)
"""

def unbounded_knapsack_recursive(weights: List[int], vals: List[int], bag_weight: int, n: int) -> int:
    """
    Pure Recursive Solution (No DP).
    Time:  Exponential O(2^(bag_weight / min_weight))
    Space: O(bag_weight / min_weight) - Maximum call stack depth.
    """
    # BASE CONDITION:
    # 1. No items left to pick (n == 0) -> Cannot earn value
    # 2. Bag has no remaining capacity (bag_weight == 0) -> Cannot hold anything
    if n == 0 or bag_weight == 0:
        return 0

    # CHOICE 1: Item is too heavy to fit into remaining capacity
    if weights[n - 1] > bag_weight:
        # Move to next item (n - 1)
        return unbounded_knapsack_recursive(weights, vals, bag_weight, n - 1)

    # CHOICE 2: Item fits. Explore both options:
    # Option A: Take item -> index stays at `n` because supply is infinite!
    take = vals[n - 1] + unbounded_knapsack_recursive(weights, vals, bag_weight - weights[n - 1], n)
    # Option B: Leave item -> permanently discard, move to `n - 1`
    leave = unbounded_knapsack_recursive(weights, vals, bag_weight, n - 1)

    return max(take, leave)


# ==============================================================================
# SECTION 3: TOP-DOWN DYNAMIC PROGRAMMING (MEMOIZATION)
# ==============================================================================
"""
MEMOIZATION: CACHING OVERLAPPING SUBPROBLEMS
--------------------------------------------
Because we can pick the same item multiple times across different branches,
the recursion tree produces massive numbers of identical calls:
    `helper(n, remaining_weight)`

We store each unique pair `(n, w)` in a hash table (our "notebook") so that
every subproblem is solved exactly once.
"""

def unbounded_knapsack_memoization(weights: List[int], vals: List[int], bag_weight: int) -> int:
    """
    Top-Down DP (Recursion + Memoization).
    Time:  O(n * W) - Every unique state (n, w) is solved once.
    Space: O(n * W) - Memo table + recursion call stack.
    """
    memo: Dict[Tuple[int, int], int] = {}

    def helper(n: int, w: int) -> int:
        if n == 0 or w == 0:
            return 0

        if (n, w) in memo:
            return memo[(n, w)]

        if weights[n - 1] > w:
            result = helper(n - 1, w)
        else:
            # Crucial detail: helper(n, w - weights[n - 1]) maintains `n`
            take = vals[n - 1] + helper(n, w - weights[n - 1])
            leave = helper(n - 1, w)
            result = max(take, leave)

        memo[(n, w)] = result
        return result

    return helper(len(weights), bag_weight)


# ==============================================================================
# SECTION 4: BOTTOM-UP DYNAMIC PROGRAMMING (TABULATION)
# ==============================================================================
"""
PLAIN ENGLISH EXPLANATION: 2D TABLE & THE 1D SPACE OPTIMIZATION
--------------------------------------------------------------
In 0/1 Knapsack, the transition was:
    `dp[i][w] = max(vals[i-1] + dp[i - 1][w - weights[i-1]], dp[i - 1][w])`
                                  ^^^^^
                        (Looked at previous row i - 1)

In Unbounded Knapsack, since we can reuse item `i`:
    `dp[i][w] = max(vals[i-1] + dp[i][w - weights[i-1]], dp[i - 1][w])`
                                  ^^^
                        (Looks at CURRENT row i!)

SPACE OPTIMIZATION TO 1D ARRAY:
Because the transition depends on:
1. `dp[i - 1][w]` (the value at the same capacity from the previous item)
2. `dp[i][w - weights[i-1]]` (the value at smaller capacity on the CURRENT item)

If we iterate `w` FORWARD from `weights[i-1]` up to `W`, the updated value
`dp[w - weights[i-1]]` will ALREADY reflect the current item's choice!
Contrast this with 0/1 knapsack, where we had to loop backwards to avoid reuse.
"""

def unbounded_knapsack_tabulation_2d(weights: List[int], vals: List[int], bag_weight: int) -> int:
    """
    Bottom-Up DP (2D Table).
    Time:  O(n * W)
    Space: O(n * W)
    """
    n = len(weights)
    W = bag_weight
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            curr_weight = weights[i - 1]
            curr_val = vals[i - 1]

            if curr_weight <= w:
                # Look at SAME row `i` for `take`, look at PREVIOUS row `i - 1` for `leave`
                take = curr_val + dp[i][w - curr_weight]
                leave = dp[i - 1][w]
                dp[i][w] = max(take, leave)
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]


def unbounded_knapsack_tabulation_1d(weights: List[int], vals: List[int], bag_weight: int) -> int:
    """
    Bottom-Up DP (Space-Optimized 1D Array).
    Time:  O(n * W)
    Space: O(W)
    """
    W = bag_weight
    dp = [0 for _ in range(W + 1)]

    for i in range(len(weights)):
        curr_weight = weights[i]
        curr_val = vals[i]
        # Iterate FORWARD from curr_weight to W allows multiple uses of the same item
        for w in range(curr_weight, W + 1):
            dp[w] = max(dp[w], curr_val + dp[w - curr_weight])

    return dp[W]


# ==============================================================================
# SECTION 5: DIRECT COMPARISON: 0/1 VS. UNBOUNDED KNAPSACK
# ==============================================================================
"""
+-------------------------+-----------------------------------+-----------------------------------+
| Feature                 | 0/1 Knapsack                      | Unbounded Knapsack                |
+-------------------------+-----------------------------------+-----------------------------------+
| Item Availability       | Exactly once (0 or 1)             | Infinite times (0, 1, 2, ...)     |
| Recursive Take Step     | helper(n - 1, w - wt[n - 1])      | helper(n, w - wt[n - 1])          |
| 2D Table Take Step      | dp[i - 1][w - wt[i - 1]]          | dp[i][w - wt[i - 1]]              |
| 1D Array Loop Direction | Backwards: W down to wt[i]        | Forwards: wt[i] up to W           |
| Primary Archetypes      | Subset Sum, Partition Equal Sum   | Rod Cutting, Coin Change I & II   |
+-------------------------+-----------------------------------+-----------------------------------+
"""


# ==============================================================================
# SECTION 6: UNBOUNDED KNAPSACK EXTENSIONS & VARIATIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# VARIATION 1: ROD CUTTING PROBLEM
# ------------------------------------------------------------------------------
"""
Problem: Given a rod of length `N` inches and an array of prices that includes
prices of all pieces of size smaller than or equal to `N`. Determine the
maximum value obtainable by cutting up the rod and selling the pieces.

HOW IT MAPS TO UNBOUNDED KNAPSACK:
- `rod_length N` === `bag_weight W`
- `piece length (1 to N)` === `weights`
- `piece price` === `vals`
- You can cut multiple pieces of the same length === `Unbounded supply`
"""

def rod_cutting_memo(price: List[int], n: int) -> int:
    """
    Top-Down Memoization for Rod Cutting.
    `price[i]` corresponds to rod cut of length `i + 1`.
    """
    memo: Dict[Tuple[int, int], int] = {}

    def helper(cut_idx: int, remaining_len: int) -> int:
        if cut_idx == 0 or remaining_len == 0:
            return 0

        if (cut_idx, remaining_len) in memo:
            return memo[(cut_idx, remaining_len)]

        piece_len = cut_idx  # length of cut is cut_idx (1-based)
        piece_val = price[cut_idx - 1]

        if piece_len <= remaining_len:
            # We can take piece_len and stay at cut_idx
            take = piece_val + helper(cut_idx, remaining_len - piece_len)
            leave = helper(cut_idx - 1, remaining_len)
            result = max(take, leave)
        else:
            result = helper(cut_idx - 1, remaining_len)

        memo[(cut_idx, remaining_len)] = result
        return result

    return helper(n, n)


def rod_cutting_dp(price: List[int], n: int) -> int:
    """
    Bottom-Up Tabulation (1D Space-Optimized) for Rod Cutting.
    Time:  O(n^2)
    Space: O(n)
    """
    dp = [0 for _ in range(n + 1)]

    # dp[length] = max revenue achievable from a rod of given length
    for length in range(1, n + 1):
        max_val = -math.inf
        for cut in range(1, length + 1):
            # cut-1 accesses the 0-indexed price array
            max_val = max(max_val, price[cut - 1] + dp[length - cut])
        dp[length] = max_val

    return dp[n]


# ------------------------------------------------------------------------------
# VARIATION 2: COIN CHANGE II (COUNT NUMBER OF WAYS)
# ------------------------------------------------------------------------------
"""
Problem: Given an integer array `coins` representing coins of different
denominations and an integer `amount`, return the NUMBER OF COMBINATIONS that
make up that amount. You have an infinite number of each kind of coin.

HOW IT MAPS TO UNBOUNDED KNAPSACK:
- `coins` === `weights`
- `amount` === `bag_weight`
- Instead of maximizing value with `max()`, we add the combinations: `take + leave`.
- Ordering does NOT matter (combinations, not permutations), so we iterate coin by coin.
"""

def coin_change_ways_rec(coins: List[int], amount: int, n: int) -> int:
    """Pure recursive combination counter."""
    if amount == 0:
        return 1  # Exactly one way to make 0 amount: use empty set
    if n == 0:
        return 0  # No coins left, cannot form non-zero amount

    if coins[n - 1] <= amount:
        # Take coin (stay at index n) + Leave coin (move to n - 1)
        take = coin_change_ways_rec(coins, amount - coins[n - 1], n)
        leave = coin_change_ways_rec(coins, amount, n - 1)
        return take + leave
    else:
        return coin_change_ways_rec(coins, amount, n - 1)


def coin_change_ways_memo(coins: List[int], amount: int) -> int:
    """Top-Down Memoization for Coin Change II."""
    memo: Dict[Tuple[int, int], int] = {}

    def helper(n: int, a: int) -> int:
        if a == 0:
            return 1
        if n == 0:
            return 0

        if (n, a) in memo:
            return memo[(n, a)]

        if coins[n - 1] <= a:
            take = helper(n, a - coins[n - 1])
            leave = helper(n - 1, a)
            res = take + leave
        else:
            res = helper(n - 1, a)

        memo[(n, a)] = res
        return res

    return helper(len(coins), amount)


def coin_change_ways_dp(coins: List[int], amount: int) -> int:
    """
    Bottom-Up Tabulation (1D Space-Optimized) for Coin Change II.
    Time:  O(n * amount)
    Space: O(amount)
    """
    dp = [0 for _ in range(amount + 1)]
    dp[0] = 1  # 1 way to form amount 0: pick no coins

    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]

    return dp[amount]


# ------------------------------------------------------------------------------
# VARIATION 3: COIN CHANGE I (MINIMUM COINS TO MAKE SUM)
# ------------------------------------------------------------------------------
"""
Problem: Given an integer array `coins` and an integer `amount`, return the
FEWEST NUMBER of coins needed to make up that amount. If that amount cannot be
made up by any combination of the coins, return -1.

HOW IT MAPS TO UNBOUNDED KNAPSACK:
- `coins` === `weights`
- `amount` === `bag_weight`
- Instead of maximizing value, we MINIMIZE item count!
- Taking a coin adds `1` coin to our count: `1 + solve(amount - coin)`.
- Base cases require handling infinity because an impossible subproblem must
  not be selected by `min()`.
"""

from math import inf

def min_coins_rec(coins: List[int], amount: int, curr_index: int):

    # end cases
    if amount == 0:
        return 0
    if curr_index == len(coins):
        return inf

    leave = min_coins_rec(coins=coins, amount=amount, curr_index=curr_index+1)
    if coins[curr_index] > amount:
        return leave
    else:
        take = 1+ min_coins_rec(coins=coins, amount=amount-coins[curr_index], curr_index=curr_index)
        return min(take, leave)

def min_coins_memo(coins: List[int], amount: int):


    memo : Dict[Tuple[int,int], int] = {}

    def helper(amount: int, curr_index: int):

        if amount == 0:
            return 0
        if curr_index == len(coins):
            return inf

        if (amount,curr_index) in memo:
            return memo[amount,curr_index]

        leave = helper(amount=amount, curr_index=curr_index+1)
        if coins[curr_index] > amount:
            result = leave
        else:
            take = 1 + helper(amount=amount-coins[curr_index], curr_index=curr_index)

            result = min(take, leave)

        memo[amount,curr_index] = result
        return result
    helper_result = helper(amount=amount, curr_index=0)
    if helper_result== inf:
        return -1
    return helper_result

def min_coins_dp(coins: List[int], amount: int):

    dp = [[inf for _ in range(amount+1)] for _ in range(len(coins)+1)]

    for coin_index in range(1,len(coins)+1):
        for target_amount in range(1, amount+1):
            if coins[coin_index-1] > target_amount:
                dp[coin_index][target_amount] = dp[coin_index-1][target_amount]

            elif coins[coin_index-1] == target_amount:
                dp[coin_index][target_amount] = 1
            else:
                dp[coin_index][target_amount] = min(dp[coin_index-1][target_amount], 1 + dp[coin_index][target_amount-coins[coin_index-1]])

    if dp[len(coins)][amount] == inf:
        return -1
    return dp[len(coins)][amount]


# ==============================================================================
# SECTION 7: RUNNABLE TESTS & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("TESTING UNBOUNDED KNAPSACK: RECURSION vs TOP-DOWN vs BOTTOM-UP")
    print("=" * 75)

    values = [1, 4, 5, 7]
    weights = [1, 3, 4, 5]
    capacity = 8
    n = len(values)

    res_rec = unbounded_knapsack_recursive(weights, values, capacity, n)
    res_memo = unbounded_knapsack_memoization(weights, values, capacity)
    res_tab2d = unbounded_knapsack_tabulation_2d(weights, values, capacity)
    res_tab1d = unbounded_knapsack_tabulation_1d(weights, values, capacity)

    print(f"Items (Values):        {values}")
    print(f"Items (Weights):       {weights}")
    print(f"Bag Capacity:          {capacity}\n")
    print(f"1. Pure Recursion Result:          {res_rec}")
    print(f"2. Top-Down (Memo) Result:         {res_memo}")
    print(f"3. Bottom-Up (2D Table) Result:    {res_tab2d}")
    print(f"4. Bottom-Up (1D Optimized) Result:{res_tab1d}")

    # Optimal choice: Weight 3 (val 4) twice + Weight 1 (val 1) twice -> total wt = 8, val = 10
    # OR Weight 3 (val 4) once + Weight 5 (val 7) once -> total wt = 8, val = 11!
    assert res_rec == res_memo == res_tab2d == res_tab1d == 11
    print("\n[SUCCESS] All Unbounded Knapsack implementations match perfectly (Result: 11)!")

    print("\n" + "=" * 75)
    print("TESTING UNBOUNDED KNAPSACK VARIATIONS")
    print("=" * 75)

    # 1. Rod Cutting Problem
    rod_prices = [1, 5, 8, 9, 10, 17, 17, 20]
    rod_len = 8
    best_rod_memo = rod_cutting_memo(rod_prices, rod_len)
    best_rod_dp = rod_cutting_dp(rod_prices, rod_len)
    print(f"Rod Cutting: Max revenue for length {rod_len} -> {best_rod_dp}")
    assert best_rod_memo == best_rod_dp == 22  # Cuts of 2 and 6 -> 5 + 17 = 22

    # 2. Coin Change II (Count combinations)
    coins_ways = [1, 2, 5]
    target_amount = 5
    ways_rec = coin_change_ways_rec(coins_ways, target_amount, len(coins_ways))
    ways_memo = coin_change_ways_memo(coins_ways, target_amount)
    ways_dp = coin_change_ways_dp(coins_ways, target_amount)
    print(f"Coin Change II: Ways to make {target_amount} with {coins_ways} -> {ways_dp}")
    # Combinations: [5], [2,2,1], [2,1,1,1], [1,1,1,1,1] -> 4 ways
    assert ways_rec == ways_memo == ways_dp == 4

    # 3. Coin Change I (Minimum Coins)
    coins_min = [1, 2, 5]
    target_min = 11
    min_coins_memo = min_coins_memo(coins_min, target_min)
    min_coins_dp = min_coins_dp(coins_min, target_min)
    print(f"Coin Change I: Min coins to make {target_min} with {coins_min} -> {min_coins_dp}")
    # 5 + 5 + 1 = 11 -> 3 coins
    assert min_coins_memo == min_coins_dp == 3

    # Test an impossible amount
    impossible_coins = [2, 4, 6]
    target_impossible = 7
    imp_result = min_coins_dp(impossible_coins, target_impossible)
    print(f"Coin Change I (Impossible): Min coins for {target_impossible} with {impossible_coins} -> {imp_result}")
    assert imp_result == -1

    print("\n[SUCCESS] All variation problems tested and verified!")