"""
================================================================================
DYNAMIC PROGRAMMING: 0/1 KNAPSACK, TOP-DOWN & BOTTOM-UP EXPLAINED
================================================================================

TABLE OF CONTENTS:
1. RECAP & THE 3 TYPES OF KNAPSACK PROBLEMS
2. THE CHOICE DIAGRAM & PURE RECURSION (CLEANED UP & BUG-FIXED)
3. TOP-DOWN DYNAMIC PROGRAMMING (MEMOIZATION)
   - Plain English Explanation (The "Notebook" Analogy)
   - Step-by-Step Code Implementation
4. BOTTOM-UP DYNAMIC PROGRAMMING (TABULATION)
   - Plain English Explanation (The "Spreadsheet" Analogy)
   - How to Convert Recursion to an Iterative Table
   - Step-by-Step Code Implementation
5. DIRECT COMPARISON: TOP-DOWN VS. BOTTOM-UP
6. 0/1 KNAPSACK EXTENSIONS & VARIATIONS
   - Variation 1: Subset Sum Problem (Boolean DP)
   - Variation 2: Equal Sum Partition Problem
   - Variation 3: Count of Subsets with a Given Sum
7. RUNNABLE TESTS & VERIFICATION
================================================================================
"""

from typing import List, Dict, Tuple


# ==============================================================================
# SECTION 1: THE 3 TYPES OF KNAPSACK PROBLEMS
# ==============================================================================
"""
Before writing code, let's establish why 0/1 Knapsack is unique:

1. FRACTIONAL KNAPSACK:
   - You can break items into pieces (like flour, gold dust, or sugar).
   - Solution: GREEDY approach. Sort items by value-to-weight ratio (value / weight)
     in descending order, take as much of the top item as possible, then move on.
   - Dynamic Programming is NOT needed here because Greedy guarantees optimal profit.

2. 0/1 KNAPSACK:
   - Items are discrete, indivisible units (laptops, paintings, watches).
   - You either take the whole item (1) or leave it completely (0).
   - Greedy FAILS here! (Taking a heavy, slightly higher-value item might block
     two lighter items whose combined value is far superior).
   - You MUST explore choices -> Requires Dynamic Programming.

3. UNBOUNDED KNAPSACK:
   - Same as 0/1, BUT you have an unlimited supply of each item (e.g., Coin Change).
   - Once you take an item, you can take the exact same item again!
"""


# ==============================================================================
# SECTION 2: THE RECURSIVE BASELINE & THE CHOICE DIAGRAM
# ==============================================================================
"""
THE CHOICE DIAGRAM FOR ITEM (n - 1):
-----------------------------------
We inspect item `n-1` (working from the back of the list):

Is weight[n-1] <= remaining capacity?
      /                        \\
    YES                         NO
   /   \\                         \\
TAKE   DON'T TAKE               DON'T TAKE (Item is simply too heavy!)

- TAKE:
    Value gained = vals[n-1]
    Remaining capacity shrinks to: bag_weight - weight[n-1]
    Move to next item: n - 1
- DON'T TAKE:
    Value gained = 0
    Remaining capacity remains: bag_weight
    Move to next item: n - 1

"""

def knapsack_recursive(weights: List[int], vals: List[int], bag_weight: int, n: int) -> int:
    """
    Pure Recursive Solution (No DP).
    Time:  O(2^n) - Branches twice at every step.
    Space: O(n)   - Maximum height of the call stack.
    """
    # BASE CONDITION:
    # Think of the smallest valid inputs:
    # 1. No items left to pick (n == 0) -> Can't make any value
    # 2. Bag has no remaining capacity (bag_weight == 0) -> Can't hold anything
    if n == 0 or bag_weight == 0:
        return 0

    # CHOICE 1: Item is too heavy to fit into the remaining bag space
    if weights[n - 1] > bag_weight:
        # We have NO choice: skip this item
        return knapsack_recursive(weights, vals, bag_weight, n - 1)

    # CHOICE 2: Item CAN fit. We compare both options and take the maximum:
    # Option A: Take item (earn its value, deduct its weight, consider remaining items)
    take = vals[n - 1] + knapsack_recursive(weights, vals, bag_weight - weights[n - 1], n - 1)
    # Option B: Leave item (earn 0, keep capacity, consider remaining items)
    leave = knapsack_recursive(weights, vals, bag_weight, n - 1)

    return max(take, leave)


# ==============================================================================
# SECTION 3: TOP-DOWN DYNAMIC PROGRAMMING (MEMOIZATION)
# ==============================================================================
"""
PLAIN ENGLISH EXPLANATION: THE "NOTEBOOK" ANALOGY
-------------------------------------------------
Imagine you are solving a massive math quiz. A teacher asks you:
    "What is 387 * 492?"
You sit down, calculate for 5 minutes, and find the answer: 190,404.

If 10 minutes later the teacher asks you the exact same question:
    "What is 387 * 492?"
Do you spend another 5 minutes recalculating it?
Of course not! You open your notebook, look up the answer you wrote down, and
reply in 1 second.

TOP-DOWN (Memoization) is literally just normal recursion with a notebook!
- You start at the TOP (the original problem: `n` items, full capacity `W`).
- You branch downwards into smaller subproblems recursively.
- Before calculating ANY subproblem, you check your notebook (a dictionary or 2D list).
- If it's already there, return it immediately!
- If it's not there, compute it once, write it in the notebook, and then return it.

WHAT GOES INTO THE NOTEBOOK KEY?
Look at your recursive function parameters:
    `knapsack(weights, vals, bag_weight, n)`
Notice `weights` and `vals` never change.
The ONLY two variables changing on every call are:
    1. `n` (how many items we are considering)
    2. `bag_weight` (remaining capacity)
Therefore, our notebook key is simply the tuple: `(n, bag_weight)`.
"""

def knapsack_memoization(weights: List[int], vals: List[int], bag_weight: int) -> int:
    """
    Top-Down DP (Recursion + Memoization Cache).
    Time:  O(n * W) - Every unique pair (n, bag_weight) is calculated only once.
    Space: O(n * W) - To store up to n * W results in our memo dictionary.
    """
    # Our "notebook" to store results: (item_count, remaining_capacity) -> max_val
    memo: Dict[Tuple[int, int], int] = {}

    def helper(n: int, w: int) -> int:
        # Base condition
        if n == 0 or w == 0:
            return 0

        # Step 1: Check the notebook first!
        if (n, w) in memo:
            return memo[(n, w)]

        # Step 2: If not in notebook, calculate using choice diagram
        if weights[n - 1] > w:
            result = helper(n - 1, w)
        else:
            take = vals[n - 1] + helper(n - 1, w - weights[n - 1])
            leave = helper(n - 1, w)
            result = max(take, leave)

        # Step 3: Write down the answer in the notebook before returning
        memo[(n, w)] = result
        return result

    return helper(len(weights), bag_weight)


# ==============================================================================
# SECTION 4: BOTTOM-UP DYNAMIC PROGRAMMING (TABULATION)
# ==============================================================================
"""
PLAIN ENGLISH EXPLANATION: THE "SPREADSHEET" ANALOGY
---------------------------------------------------
In Top-Down, you start with the big question and dig down into smaller subproblems
using function calls.

In BOTTOM-UP (Tabulation), you throw away recursion entirely!
Instead, you build an Excel spreadsheet:
- You solve the tiniest, simplest problems FIRST (Row 0, Column 0).
- Then, you use those answers to solve slightly bigger problems.
- You build up row-by-row until the very bottom-right cell gives you the final answer.

HOW TO DESIGN THE 2D TABLE:
- Rows represent: `i` (number of items considered, from 0 up to `n`)
- Columns represent: `w` (capacity of bag, from 0 up to `bag_weight`)
- Value inside cell `dp[i][w]`: The maximum value using the first `i` items
  with a capacity limit of `w`.

CONVERTING RECURSION TO TABLE STEPS:
1. Initialize table of size `(n + 1) x (W + 1)` with zeros.
   - Row 0 means 0 items -> value is 0.
   - Col 0 means capacity 0 -> value is 0.
   (This directly mirrors our recursive base condition `if n == 0 or w == 0: return 0`).
2. Replace recursive calls with table lookups:
   - `helper(n - 1, w)`               becomes  `dp[i - 1][w]`
   - `helper(n - 1, w - weights[i-1])` becomes  `dp[i - 1][w - weights[i - 1]]`
"""

def knapsack_tabulation(weights: List[int], vals: List[int], bag_weight: int) -> int:
    """
    Bottom-Up DP (Tabulation - 2D Table).
    Time:  O(n * W) - Two nested loops: outer runs n times, inner runs W times.
    Space: O(n * W) - A 2D array of size (n + 1) x (bag_weight + 1).
    """
    n = len(weights)
    W = bag_weight

    # Create table filled with 0s
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Fill table iteratively:
    # i represents considering the first i items (1 to n)
    # w represents current test capacity (1 to W)
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            curr_weight = weights[i - 1]
            curr_val = vals[i - 1]

            if curr_weight <= w:
                # Choice: max of (take item i-1, leave item i-1)
                take = curr_val + dp[i - 1][w - curr_weight]
                leave = dp[i - 1][w]
                dp[i][w] = max(take, leave)
            else:
                # Cannot take, copy answer from row above
                dp[i][w] = dp[i - 1][w]

    # The bottom-right cell contains the answer for all n items and full capacity W
    return dp[n][W]


# ==============================================================================
# SECTION 5: TOP-DOWN VS. BOTTOM-UP COMPARISON
# ==============================================================================
"""
+---------------------+-----------------------------------+-----------------------------------+
| Feature             | Top-Down (Memoization)            | Bottom-Up (Tabulation)            |
+---------------------+-----------------------------------+-----------------------------------+
| Starting Point      | Starts at original goal (n, W)    | Starts at smallest base case (0,0)|
| Control Flow        | Recursive (Function Call Stack)   | Iterative (Nested For-Loops)      |
| State Evaluation    | Only computes NEEDED states       | Computes ALL states systematically|
| Stack Overflow Risk | Yes (if recursion depth > 1000)   | None (no recursion used)          |
| Space Optimization  | Hard to optimize beyond O(n*W)    | Can easily reduce to 1D O(W) space|
| Intuition           | Easiest to write after recursion  | Requires designing table order    |
+---------------------+-----------------------------------+-----------------------------------+
"""


# ==============================================================================
# SECTION 6: 0/1 KNAPSACK EXTENSIONS & VARIATIONS
# ==============================================================================
"""
All the problems listed below look totally different on the surface,
but under the hood they are JUST 0/1 Knapsack with minor tweaks:

Item weight in knapsack       ===>   Number in array
Bag capacity W in knapsack    ===>   Target Sum
Taking an item (1) or not (0) ===>   Including a number in subset or not
"""

# ------------------------------------------------------------------------------
# VARIATION 1: SUBSET SUM PROBLEM
# ------------------------------------------------------------------------------
"""
Problem: Given an array of non-negative integers `arr` and a target integer `sum_target`,
is there a subset whose sum equals exactly `sum_target`? Return True/False.

HOW IT MAPS TO KNAPSACK:
- `weights` = `arr`
- `bag_weight` = `sum_target`
- We don't need `vals`! Instead of maximizing value, we check if exact sum is possible.
- `max(...)` becomes boolean `OR` (`take or leave`).
"""
def subset_sum_target_sum_rec(arr: List[int], sum_target: int, curr_index: int, curr_sum: int) -> bool:

    # base case
    if curr_sum == sum_target:
        return True
    if curr_sum > sum_target:
        return False
    
    if curr_index >= len(arr):
        return False

    take = subset_sum_target_sum_rec(arr=arr, sum_target=sum_target, curr_index=curr_index+1, curr_sum=curr_sum+arr[curr_index])

    leave = subset_sum_target_sum_rec(arr=arr, sum_target=sum_target, curr_index=curr_index+1, curr_sum=curr_sum)

    return take or leave

def subset_sum_target_sum_memo(arr: List[int], sum_target: int) -> bool:

    memo : Dict[Tuple[int,int], bool] = {}

    def helper(curr_index: int, curr_sum: int) -> bool:

        if curr_sum == sum_target:
            return True
        if curr_sum > sum_target:
            return False
        if curr_index >= len(arr):
            return False

        if (curr_index, curr_sum) in memo:
            return memo[curr_index, curr_sum]

        
        take = helper(curr_index=curr_index+1, curr_sum=curr_sum+arr[curr_index])
        leave = helper(curr_index=curr_index+1, curr_sum=curr_sum)

        result = take or leave

        memo[curr_index,curr_sum] = result
        return result
    return helper(0,0)

def subset_sum_target_sum_dp(arr: List[int], sum_target: int) -> bool:

    # each element (i,j) in matrix indicates whether or not it is possible to form sum of j using elements up to index i

    dp = [[False for _ in range(sum_target+1)] for _ in range(len(arr)+1)]

    # base case - if sum_target is 0, then always true
    for i in range(len(arr)+1):
        dp[i][0] = True

    for item_index in range(1,len(arr)+1):
        curr_val = arr[item_index-1]
    
        for target in range(1,sum_target+1):

            if curr_val <= target:
                # choice: include curr_val OR exclude curr_val
                dp[item_index][target] = dp[item_index-1][target-curr_val] or dp[item_index-1][target]
            else:
                # cannot include - if previous sums work out (item-index-1) then fine, otherwise it stays false
                dp[item_index][target] = dp[item_index-1][target]
    return dp[len(arr)][sum_target]            


# ------------------------------------------------------------------------------
# VARIATION 2: EQUAL SUM PARTITION
# ------------------------------------------------------------------------------
"""
Problem: Can an array be partitioned into two subsets with EQUAL sum?
Example: [1, 5, 11, 5] -> True (Subsets [1, 5, 5] and [11], both sum to 11).

THE REDUCTION TO SUBSET SUM:
1. Total sum of all numbers must be EVEN. If total sum is odd (e.g. 23),
   it is mathematically impossible to divide into two equal integers -> Return False.
2. If total sum is even, then each subset must equal: `total_sum // 2`.
3. The problem reduces to: "Does there exist a subset whose sum == total_sum // 2?"
   Which is EXACTLY Variation 1 (Subset Sum)!
"""
def can_partition_equal_sum(arr: List[int]) -> bool:
    total_sum = sum(arr)
    # If odd, impossible to split evenly
    if total_sum % 2 != 0:
        return False
    # Target is half of total sum
    return subset_sum_target_sum_dp(arr, total_sum // 2)


# ------------------------------------------------------------------------------
# VARIATION 3: COUNT OF SUBSET SUM
# ------------------------------------------------------------------------------
"""
Problem: Count how many subsets have a sum equal to `target`.
Example: arr = [1, 2, 3, 3], target = 6
Subsets: {1, 2, 3}, {1, 2, 3}, {3, 3} -> Returns 3.

THE MODIFICATION TO 0/1 KNAPSACK:
- In Knapsack we used `max(take, leave)`.
- In Subset Sum we used `take OR leave`.
- In Count of Subset Sum, we want TOTAL combinations, so we use `take + leave`!
"""
def count_subset_sum(arr: List[int], target: int) -> int:
    n = len(arr)
    dp = [[0 for _ in range(target + 1)] for _ in range(n + 1)]

    # Base case: Sum 0 can always be formed by 1 empty subset: []
    for i in range(n + 1):
        dp[i][0] = 1

    for i in range(1, n + 1):
        for s in range(1, target + 1):
            if arr[i - 1] <= s:
                # Add ways from taking item + ways from not taking item
                dp[i][s] = dp[i - 1][s - arr[i - 1]] + dp[i - 1][s]
            else:
                dp[i][s] = dp[i - 1][s]

    return dp[n][target]


# ==============================================================================
# SECTION 7: RUNNABLE TESTS & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING 0/1 KNAPSACK: RECURSION vs TOP-DOWN vs BOTTOM-UP")
    print("=" * 70)

    # Given data
    values = [1, 2, 3, 4]
    weights = [1, 1, 2, 3]
    capacity = 6
    n = len(values)

    res_rec = knapsack_recursive(weights, values, capacity, n)
    res_memo = knapsack_memoization(weights, values, capacity)
    res_tab = knapsack_tabulation(weights, values, capacity)

    print(f"Items (Values):  {values}")
    print(f"Items (Weights): {weights}")
    print(f"Bag Capacity:    {capacity}\n")
    print(f"1. Pure Recursion Result:      {res_rec}")
    print(f"2. Top-Down (Memo) Result:     {res_memo}")
    print(f"3. Bottom-Up (Table) Result:   {res_tab}")

    assert res_rec == res_memo == res_tab == 9, "All knapsack results should be 7!"
    print("\n[SUCCESS] All three 0/1 knapsack implementations match perfectly!")

    print("\n" + "=" * 70)
    print("TESTING 0/1 KNAPSACK VARIATIONS")
    print("=" * 70)

    # Test Subset Sum
    test_arr = [2, 3, 7, 8, 10]
    target_sum = 11
    has_sub = subset_sum_target_sum_dp(test_arr, target_sum)
    print(f"Subset Sum: Does {test_arr} have a subset summing to {target_sum}? -> {has_sub}")
    assert has_sub is True  # 3 + 8 = 11

    # Test Equal Sum Partition
    partition_arr1 = [1, 5, 11, 5]
    can_part1 = can_partition_equal_sum(partition_arr1)
    print(f"Equal Partition: Can {partition_arr1} be partitioned equally? -> {can_part1}")
    assert can_part1 is True  # [1, 5, 5] and [11]

    partition_arr2 = [1, 2, 3, 5]
    can_part2 = can_partition_equal_sum(partition_arr2)
    print(f"Equal Partition: Can {partition_arr2} be partitioned equally? -> {can_part2}")
    assert can_part2 is False  # sum is 11 (odd)

    # Test Count Subset Sum
    count_arr = [1, 2, 3, 3]
    count_target = 6
    ways = count_subset_sum(count_arr, count_target)
    print(f"Count Subsets: How many subsets in {count_arr} sum to {count_target}? -> {ways} ways")
    assert ways == 3  # [1, 2, 3_a], [1, 2, 3_b], [3_a, 3_b]

    print("\n[SUCCESS] All variation problems tested and verified!")
