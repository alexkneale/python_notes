"""
================================================================================
DYNAMIC PROGRAMMING: KADANE'S ALGORITHM & CONTIGUOUS SUBARRAY DP
================================================================================

TABLE OF CONTENTS:
1. RECAP & THE CORE INTUITION: THE CONTIGUOUS "EXTEND OR RESTART" PATTERN
2. THE ARCHETYPE (MAX SUBARRAY SUM): RECURSION, MEMO, TABULATION & O(1) SPACE
3. THE PROGRESSION ENGINE: LOCAL ENDING STATE VS. GLOBAL OPTIMUM
4. VARIATION 1: MAXIMUM PRODUCT SUBARRAY (DUAL MIN/MAX STATE TRACKING)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
5. VARIATION 2: MAXIMUM SUM CIRCULAR SUBARRAY (COMPLEMENTARY MIN-SUBARRAY DUALITY)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
6. VARIATION 3: LONGEST TURBULENT SUBARRAY (ALTERNATING PARITY STATE MACHINE)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
7. VARIATION 4: MAXIMUM SUBARRAY SUM WITH ONE DELETION (TWO-STATE TRANSITION)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(1) Space)
8. COMPARATIVE SUMMARY TABLE: TRANSITIONS & STATE DEFINITIONS
9. RUNNABLE TESTS & VERIFICATION
================================================================================
"""

from typing import Dict, Tuple, List


# ==============================================================================
# SECTION 1: THE CORE INTUITION: THE CONTIGUOUS "EXTEND OR RESTART" PATTERN
# ==============================================================================
"""
THE CONTIGUOUS SUBARRAY DP PATTERN EXPLAINED:
---------------------------------------------
Unlike subsequence problems where we can skip elements and still keep building
our sequence, a SUBARRAY must be strictly CONTIGUOUS (unbroken).

To enforce contiguity in Dynamic Programming, we change how we define our state:
Instead of defining `dp[i]` as "the best subarray anywhere in `nums[0...i]`",
we define `dp[i]` as:
    "The best subarray that MUST END EXACTLY AT index `i`."

Key Characteristics:
1. The "Extend or Restart" Decision:
   Because the subarray must end at index `i`, element `nums[i]` MUST be included.
   We only have two choices at index `i`:
     - EXTEND the best contiguous subarray ending at `i - 1` by appending `nums[i]`.
     - RESTART a brand new subarray beginning and ending at `nums[i]` alone
       (abandoning the previous prefix because it hurts our optimal value).

2. Local vs. Global Answer:
   - `dp[i]` holds the LOCAL optimum (best subarray anchored at index `i`).
   - The overall problem asks for the best subarray ending ANYWHERE.
   - Therefore, the GLOBAL answer is `max(dp[0], dp[1], ..., dp[n - 1])`.

3. Space Optimization:
   Because state `dp[i]` depends strictly on the immediately preceding state
   `dp[i - 1]`, a 1D array of size `O(n)` can always be compressed into `O(1)`
   auxiliary space using scalar rolling variables!
"""


# ==============================================================================
# SECTION 2: THE ARCHETYPE: MAXIMUM SUBARRAY SUM (KADANE'S ALGORITHM)
# ==============================================================================
"""
Problem:
Given an integer array `nums`, find the contiguous non-empty subarray which has
the largest sum, and return its sum.

State Definition:
`dp[i]` = maximum sum of a contiguous subarray ending at index `i`.

Recurrence:
dp[i] = max(
    nums[i],             (Restart: Start a fresh subarray at index i)
    dp[i - 1] + nums[i]  (Extend: Append nums[i] to the best subarray ending at i-1)
)

Global Answer:
max(dp[0], dp[1], ..., dp[n - 1])
"""

# ------------------------------------------------------------------------------
# 2.1 Pure Recursion
# ------------------------------------------------------------------------------
def max_subarray_recursive(nums: List[int]) -> int:
    """
    Pure Recursive Solution.
    At each index `i`, we compute the max subarray ending at `i` by recursively
    querying `i - 1`, and we track the global maximum across all indices.
    Time:  O(n) or O(2^n) depending on how branching is structured. Here we
           demonstrate unmemoized branching over (index, is_currently_taking).
    Space: O(n) - Maximum recursion stack depth.
    """
    def helper(i: int, must_pick: bool) -> int:
        # Base case: reached the end of the array
        if i == len(nums):
            return 0 if must_pick else -float("inf")

        if must_pick:
            # We are already inside a contiguous subarray:
            # Either stop the subarray right here (0), or continue extending to i + 1
            return max(0, nums[i] + helper(i + 1, True))

        # We haven't started our subarray yet:
        # Either skip index i (stay False), or start the subarray at index i (become True)
        return max(
            helper(i + 1, False),          # Skip nums[i], start later
            nums[i] + helper(i + 1, True)  # Start subarray at nums[i]
        )

    return int(helper(0, False))


# ------------------------------------------------------------------------------
# 2.2 Top-Down DP (Memoization)
# ------------------------------------------------------------------------------
def max_subarray_memoization(nums: List[int]) -> int:
    """
    Top-Down DP with Memoization.
    Defines `dp(i)` as the maximum subarray sum ending at index `i`.
    Time:  O(n) - Each index `i` is computed once.
    Space: O(n) - Recursion stack O(n) + memo table O(n).
    """
    memo: Dict[int, int] = {}

    def max_ending_at(i: int) -> int:
        if i == 0:
            return nums[0]
        if i in memo:
            return memo[i]

        # Choice: start fresh at nums[i] OR extend best subarray ending at i - 1
        memo[i] = max(nums[i], max_ending_at(i - 1) + nums[i])
        return memo[i]

    return max(max_ending_at(i) for i in range(len(nums)))


# ------------------------------------------------------------------------------
# 2.3 Bottom-Up DP (Tabulation)
# ------------------------------------------------------------------------------
def max_subarray_tabulation(nums: List[int]) -> int:
    """
    Bottom-Up DP (1D Table).
    Time:  O(n)
    Space: O(n)
    """
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    global_max = dp[0]

    for i in range(1, n):
        dp[i] = max(nums[i], dp[i - 1] + nums[i])
        global_max = max(global_max, dp[i])

    return global_max


# ------------------------------------------------------------------------------
# 2.4 Bottom-Up DP (Kadane's Space-Optimized O(1))
# ------------------------------------------------------------------------------
def max_subarray_kadane(nums: List[int]) -> int:
    """
    Classic Kadane's Algorithm (Rolling Scalar State).
    Time:  O(n)
    Space: O(1)
    """
    curr_max = nums[0]
    global_max = nums[0]

    for i in range(1, len(nums)):
        curr_max = max(nums[i], curr_max + nums[i])
        global_max = max(global_max, curr_max)

    return global_max


# ------------------------------------------------------------------------------
# 2.5 Bonus: Reconstructing the Actual Subarray Indices
# ------------------------------------------------------------------------------
def max_subarray_with_indices(nums: List[int]) -> Tuple[int, List[int]]:
    """
    Tracks start and end pointers whenever Kadane's algorithm restarts or
    updates the global maximum.
    Time:  O(n), Space: O(1) auxiliary.
    """
    curr_max = nums[0]
    global_max = nums[0]
    temp_start = 0
    best_start, best_end = 0, 0

    for i in range(1, len(nums)):
        if nums[i] > curr_max + nums[i]:
            curr_max = nums[i]
            temp_start = i  # Restart subarray here
        else:
            curr_max = curr_max + nums[i]  # Extend existing subarray

        if curr_max > global_max:
            global_max = curr_max
            best_start = temp_start
            best_end = i

    return global_max, nums[best_start : best_end + 1]


# ==============================================================================
# SECTION 3: THE PROGRESSION ENGINE: LOCAL ENDING STATE VS. GLOBAL OPTIMUM
# ==============================================================================
"""
How do we adapt Kadane's "Extend or Restart" engine to harder problems?
Whenever a single scalar `dp[i]` is not enough to make an optimal local decision
at index `i`, we expand the state tracked at index `i`:

1. STANDARD KADANE (Max Subarray Sum):
   - A negative running sum is ALWAYS useless for maximizing future sums.
   - Track 1 state at index `i`: `max_ending_here`.

2. SIGN-FLIPPING MULTIPLICATION (Maximum Product Subarray):
   - A large NEGATIVE product ending at `i - 1` can suddenly become a huge
     POSITIVE product if `nums[i]` is negative!
   - Track 2 states at index `i`: `(max_prod_ending_here, min_prod_ending_here)`.

3. WRAP-AROUND TOPOLOGY (Maximum Sum Circular Subarray):
   - A subarray can either be standard (non-wrapping) OR wrap around the ends.
   - Wrapping around the ends leaves a contiguous "hole" in the middle!
   - Maximizing a wrap-around subarray is equivalent to `Total_Sum - Min_Subarray_Sum`.
   - Track 2 states at index `i`: `(max_sum_ending_here, min_sum_ending_here)`.

4. DIRECTIONAL PARITY (Longest Turbulent Subarray):
   - Extending depends on whether the previous step was `<` or `>`.
   - Track 2 states at index `i`: `(up_len, down_len)`.

5. LIMITED EXCEPTIONS (Maximum Subarray Sum with One Deletion):
   - We can either have 0 deletions used so far, or 1 deletion used so far.
   - Track 2 states at index `i`: `(keep_all, delete_one)`.
"""


# ==============================================================================
# SECTION 4: VARIATION 1: MAXIMUM PRODUCT SUBARRAY
# ==============================================================================
"""
Problem:
Given an integer array `nums`, find a contiguous non-empty subarray within the
array that has the largest product, and return the product.

Why Standard Kadane Fails:
In addition, `[-2, 3, -4]` has prefix products: `-2`, `-6`, `+24`.
If we discarded `-6` at index 1 (because `3 > -6`), we would miss `-6 * -4 = 24`!
Negative numbers flip the roles of minimum and maximum.

State Definition:
- `max_dp[i]` = maximum product of a subarray ending at index `i`.
- `min_dp[i]` = minimum (most negative) product of a subarray ending at index `i`.

Recurrence:
candidates = (nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
max_dp[i] = max(candidates)
min_dp[i] = min(candidates)
"""

def max_product_rec(nums: List[int]) -> int:
    """Pure Recursion returning (max_ending_at_i, min_ending_at_i): O(n) depth."""
    def get_bounds_at(i: int) -> Tuple[int, int]:
        if i == 0:
            return nums[0], nums[0]

        prev_max, prev_min = get_bounds_at(i - 1)
        candidates = (nums[i], prev_max * nums[i], prev_min * nums[i])
        return max(candidates), min(candidates)

    return max(get_bounds_at(i)[0] for i in range(len(nums)))


def max_product_memo(nums: List[int]) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, Tuple[int, int]] = {}

    def get_bounds_at(i: int) -> Tuple[int, int]:
        if i == 0:
            return nums[0], nums[0]
        if i in memo:
            return memo[i]

        prev_max, prev_min = get_bounds_at(i - 1)
        candidates = (nums[i], prev_max * nums[i], prev_min * nums[i])
        memo[i] = (max(candidates), min(candidates))
        return memo[i]

    return max(get_bounds_at(i)[0] for i in range(len(nums)))


def max_product_dp(nums: List[int]) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    n = len(nums)
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = global_max = nums[0]

    for i in range(1, n):
        candidates = (nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        max_dp[i] = max(candidates)
        min_dp[i] = min(candidates)
        global_max = max(global_max, max_dp[i])

    return global_max


def max_product_optimized(nums: List[int]) -> int:
    """Bottom-Up O(1) Space Optimization: O(n) time, O(1) space."""
    curr_max = curr_min = global_max = nums[0]

    for i in range(1, len(nums)):
        x = nums[i]
        # Note: Must compute both simultaneously before overwriting curr_max!
        curr_max, curr_min = (
            max(x, curr_max * x, curr_min * x),
            min(x, curr_max * x, curr_min * x),
        )
        global_max = max(global_max, curr_max)

    return global_max


# ==============================================================================
# SECTION 5: VARIATION 2: MAXIMUM SUM CIRCULAR SUBARRAY
# ==============================================================================
"""
Problem:
Given a circular integer array `nums` of length `n` (where the end connects back
to the beginning), return the maximum possible sum of a non-empty subarray.

Insight (Duality of Circular Subarrays):
A maximum subarray in a circular array can take one of two forms:
  Case 1 (No Wrap): Lives completely inside `nums[0...n-1]`.
                    Solved by Standard Max-Kadane: `global_max`.
  Case 2 (Wraps Around): Takes a prefix `nums[0...i]` AND a suffix `nums[j...n-1]`.
                    Because the total sum of the array is fixed (`total_sum`),
                    maximizing the outer prefix + suffix is identical to
                    MINIMIZING the contiguous unchosen middle subarray `nums[i+1...j-1]`!
                    Solved by: `total_sum - global_min`.

Corner Case (All Negative Numbers):
If every element in `nums` is negative, `global_min` will select the ENTIRE array,
making `total_sum - global_min == 0` (which corresponds to an invalid empty subarray).
In that case, `global_max < 0`, so we simply return `global_max`.
"""

def max_circular_rec(nums: List[int]) -> int:
    """Pure Recursion checking both Max-Kadane and Min-Kadane: O(n) depth."""
    def max_ending_at(i: int) -> int:
        if i == 0:
            return nums[0]
        return max(nums[i], max_ending_at(i - 1) + nums[i])

    def min_ending_at(i: int) -> int:
        if i == 0:
            return nums[0]
        return min(nums[i], min_ending_at(i - 1) + nums[i])

    n = len(nums)
    total_sum = sum(nums)
    global_max = max(max_ending_at(i) for i in range(n))
    global_min = min(min_ending_at(i) for i in range(n))

    if global_max < 0:
        return global_max
    return max(global_max, total_sum - global_min)


def max_circular_memo(nums: List[int]) -> int:
    """Top-Down Memoization tracking (max_ending, min_ending): O(n) time & space."""
    memo: Dict[int, Tuple[int, int]] = {}

    def helper(i: int) -> Tuple[int, int]:
        if i == 0:
            return nums[0], nums[0]
        if i in memo:
            return memo[i]

        prev_max, prev_min = helper(i - 1)
        curr_max = max(nums[i], prev_max + nums[i])
        curr_min = min(nums[i], prev_min + nums[i])
        memo[i] = (curr_max, curr_min)
        return memo[i]

    total_sum = sum(nums)
    bounds = [helper(i) for i in range(len(nums))]
    global_max = max(b[0] for b in bounds)
    global_min = min(b[1] for b in bounds)

    return global_max if global_max < 0 else max(global_max, total_sum - global_min)


def max_circular_dp(nums: List[int]) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    n = len(nums)
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = global_max = global_min = nums[0]
    total_sum = nums[0]

    for i in range(1, n):
        total_sum += nums[i]
        max_dp[i] = max(nums[i], max_dp[i - 1] + nums[i])
        min_dp[i] = min(nums[i], min_dp[i - 1] + nums[i])
        global_max = max(global_max, max_dp[i])
        global_min = min(global_min, min_dp[i])

    if global_max < 0:
        return global_max
    return max(global_max, total_sum - global_min)


def max_circular_optimized(nums: List[int]) -> int:
    """Bottom-Up O(1) Space Optimization: O(n) time, O(1) space."""
    curr_max = curr_min = global_max = global_min = total_sum = nums[0]

    for i in range(1, len(nums)):
        x = nums[i]
        total_sum += x
        curr_max = max(x, curr_max + x)
        curr_min = min(x, curr_min + x)
        global_max = max(global_max, curr_max)
        global_min = min(global_min, curr_min)

    if global_max < 0:
        return global_max
    return max(global_max, total_sum - global_min)


# ==============================================================================
# SECTION 6: VARIATION 3: LONGEST TURBULENT SUBARRAY (ALTERNATING KADANE)
# ==============================================================================
"""
Problem:
Given an integer array `arr`, return the length of a maximum size turbulent
subarray of `arr`. A subarray is turbulent if the comparison sign flips between
every adjacent pair of elements in the subarray (e.g., `a < b > c < d > e`).

State Definition:
At index `i`, we track two Kadane-style contiguous lengths:
- `up[i]`   = longest turbulent subarray ending at `i` with `arr[i - 1] < arr[i]`.
- `down[i]` = longest turbulent subarray ending at `i` with `arr[i - 1] > arr[i]`.

Recurrence:
- If arr[i - 1] < arr[i]:
    up[i] = down[i - 1] + 1  (Extends a previous 'down' step)
    down[i] = 1              (Resets to single element)
- Else if arr[i - 1] > arr[i]:
    down[i] = up[i - 1] + 1  (Extends a previous 'up' step)
    up[i] = 1                (Resets to single element)
- Else (arr[i - 1] == arr[i]):
    up[i] = 1, down[i] = 1   (Equality breaks turbulence; both reset to 1)
"""

def max_turbulent_rec(arr: List[int]) -> int:
    """Pure Recursion returning (up_len, down_len) at index i."""
    def helper(i: int) -> Tuple[int, int]:
        if i == 0:
            return 1, 1
        prev_up, prev_down = helper(i - 1)
        if arr[i - 1] < arr[i]:
            return prev_down + 1, 1
        elif arr[i - 1] > arr[i]:
            return 1, prev_up + 1
        else:
            return 1, 1

    return max(max(helper(i)) for i in range(len(arr)))


def max_turbulent_memo(arr: List[int]) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, Tuple[int, int]] = {}

    def helper(i: int) -> Tuple[int, int]:
        if i == 0:
            return 1, 1
        if i in memo:
            return memo[i]

        prev_up, prev_down = helper(i - 1)
        if arr[i - 1] < arr[i]:
            res = (prev_down + 1, 1)
        elif arr[i - 1] > arr[i]:
            res = (1, prev_up + 1)
        else:
            res = (1, 1)

        memo[i] = res
        return res

    return max(max(helper(i)) for i in range(len(arr)))


def max_turbulent_dp(arr: List[int]) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    n = len(arr)
    up = [1] * n
    down = [1] * n
    max_len = 1

    for i in range(1, n):
        if arr[i - 1] < arr[i]:
            up[i] = down[i - 1] + 1
        elif arr[i - 1] > arr[i]:
            down[i] = up[i - 1] + 1
        max_len = max(max_len, up[i], down[i])

    return max_len


def max_turbulent_optimized(arr: List[int]) -> int:
    """Bottom-Up O(1) Space Optimization: O(n) time, O(1) space."""
    up = down = max_len = 1

    for i in range(1, len(arr)):
        if arr[i - 1] < arr[i]:
            up, down = down + 1, 1
        elif arr[i - 1] > arr[i]:
            down, up = up + 1, 1
        else:
            up, down = 1, 1
        max_len = max(max_len, up, down)

    return max_len


# ==============================================================================
# SECTION 7: VARIATION 4: MAXIMUM SUBARRAY SUM WITH ONE DELETION
# ==============================================================================
"""
Problem:
Given an array of integers `arr`, return the maximum sum for a non-empty
subarray with at most one element deletion.

State Definition:
At each index `i`, we track two states for subarrays ending at `i`:
- `keep[i]`   = max subarray sum ending at `i` with ZERO deletions.
- `delete[i]` = max subarray sum ending at `i` with AT MOST ONE deletion.

Recurrence:
1. `keep[i]`: Standard Kadane! Either restart at `arr[i]` or extend `keep[i - 1]`:
   keep[i] = max(arr[i], keep[i - 1] + arr[i])

2. `delete[i]`: Two possibilities:
   - Delete `arr[i]` right now: We take the best zero-deletion subarray ending
     at `i - 1` (`keep[i - 1]`). (Note: only valid for `i >= 1` so subarray isn't empty).
   - Keep `arr[i]` and rely on a deletion that already happened earlier:
     `delete[i - 1] + arr[i]`.
   delete[i] = max(keep[i - 1], delete[i - 1] + arr[i])
"""

def max_subarray_one_del_rec(arr: List[int]) -> int:
    """Pure Recursion returning (keep_at_i, delete_at_i)."""
    def helper(i: int) -> Tuple[int, float]:
        if i == 0:
            # At index 0, deleting arr[0] would leave an empty subarray (-inf)
            return arr[0], -float("inf")

        prev_keep, prev_del = helper(i - 1)
        curr_keep = max(arr[i], prev_keep + arr[i])
        curr_del = max(prev_keep, prev_del + arr[i])
        return curr_keep, curr_del

    return int(max(max(helper(i)) for i in range(len(arr))))


def max_subarray_one_del_memo(arr: List[int]) -> int:
    """Top-Down Memoization: O(n) time, O(n) space."""
    memo: Dict[int, Tuple[int, float]] = {}

    def helper(i: int) -> Tuple[int, float]:
        if i == 0:
            return arr[0], -float("inf")
        if i in memo:
            return memo[i]

        prev_keep, prev_del = helper(i - 1)
        curr_keep = max(arr[i], prev_keep + arr[i])
        curr_del = max(prev_keep, prev_del + arr[i])
        memo[i] = (curr_keep, curr_del)
        return memo[i]

    return int(max(max(helper(i)) for i in range(len(arr))))


def max_subarray_one_del_dp(arr: List[int]) -> int:
    """Bottom-Up Tabulation: O(n) time, O(n) space."""
    n = len(arr)
    keep = [0] * n
    delete = [0] * n
    keep[0] = arr[0]
    delete[0] = arr[0]  # Initialized to arr[0] so single-element arrays work
    global_max = arr[0]

    for i in range(1, n):
        keep[i] = max(arr[i], keep[i - 1] + arr[i])
        delete[i] = max(keep[i - 1], delete[i - 1] + arr[i])
        global_max = max(global_max, keep[i], delete[i])

    return global_max


def max_subarray_one_del_optimized(arr: List[int]) -> int:
    """Bottom-Up O(1) Space Optimization: O(n) time, O(1) space."""
    keep = arr[0]
    delete = arr[0]
    global_max = arr[0]

    for i in range(1, len(arr)):
        # Compute delete first (or simultaneously) since it depends on old `keep`
        keep, delete = (
            max(arr[i], keep + arr[i]),
            max(keep, delete + arr[i]),
        )
        global_max = max(global_max, keep, delete)

    return global_max


# ==============================================================================
# SECTION 8: COMPARATIVE SUMMARY TABLE
# ==============================================================================
"""
+-------------------------------+---------------------------+------------------------------------------------+
| Problem                       | Local State at Index `i`  | Transition (Extend vs. Restart)                |
+-------------------------------+---------------------------+------------------------------------------------+
| Maximum Subarray Sum          | curr_max                  | curr_max = max(x, curr_max + x)                |
| (Standard Kadane)             | (1 scalar)                |                                                |
+-------------------------------+---------------------------+------------------------------------------------+
| Maximum Product Subarray      | curr_max, curr_min        | curr_max = max(x, prev_max * x, prev_min * x)  |
|                               | (2 scalars: sign flip)    | curr_min = min(x, prev_max * x, prev_min * x)  |
+-------------------------------+---------------------------+------------------------------------------------+
| Maximum Circular Subarray     | curr_max, curr_min        | curr_max = max(x, curr_max + x)                |
|                               | (2 scalars: wrap duality) | curr_min = min(x, curr_min + x)                |
|                               |                           | Ans: max(global_max, total_sum - global_min)   |
+-------------------------------+---------------------------+------------------------------------------------+
| Longest Turbulent Subarray    | up, down                  | If arr[i-1] < arr[i]: up = down + 1, down = 1  |
|                               | (2 scalars: parity flip)  | If arr[i-1] > arr[i]: down = up + 1, up = 1    |
+-------------------------------+---------------------------+------------------------------------------------+
| Max Subarray w/ One Deletion  | keep, delete              | keep   = max(x, prev_keep + x)                 |
|                               | (2 scalars: 0 or 1 del)   | delete = max(prev_keep, prev_del + x)          |
+-------------------------------+---------------------------+------------------------------------------------+
"""


# ==============================================================================
# SECTION 9: RUNNABLE TESTS & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("TESTING KADANE'S & CONTIGUOUS SUBARRAY DP: REC vs MEMO vs TAB vs O(1) SPACE")
    print("=" * 78)

    # 1. Maximum Subarray Sum (Archetype)
    nums1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    # Best contiguous subarray is [4, -1, 2, 1] with sum 6
    ms_r = max_subarray_recursive(nums1)
    ms_m = max_subarray_memoization(nums1)
    ms_t = max_subarray_tabulation(nums1)
    ms_o = max_subarray_kadane(nums1)
    best_sum, best_sub = max_subarray_with_indices(nums1)
    print(f"1. Maximum Subarray Sum({nums1}):")
    print(f"   Rec: {ms_r} | Memo: {ms_m} | Tab: {ms_t} | Kadane O(1): {ms_o}")
    print(f"   Reconstructed Subarray: {best_sub} (Sum = {best_sum})")
    assert ms_r == ms_m == ms_t == ms_o == best_sum == 6
    assert best_sub == [4, -1, 2, 1]
    print("   [SUCCESS] All Maximum Subarray Sum variants returned 6.\n")

    # 2. Maximum Product Subarray
    nums2 = [2, 3, -2, 4, -1]
    # Whole array product: 2 * 3 * (-2) * 4 * (-1) = 48
    mp_r = max_product_rec(nums2)
    mp_m = max_product_memo(nums2)
    mp_t = max_product_dp(nums2)
    mp_o = max_product_optimized(nums2)
    print(f"2. Maximum Product Subarray({nums2}):")
    print(f"   Rec: {mp_r} | Memo: {mp_m} | Tab: {mp_t} | Opt O(1): {mp_o}")
    assert mp_r == mp_m == mp_t == mp_o == 48
    print("   [SUCCESS] All Maximum Product Subarray variants returned 48.\n")

    # 3. Maximum Sum Circular Subarray
    nums3 = [5, -3, 5]
    # Wrap-around subarray takes prefix [5] and suffix [5] -> sum = 10
    mc_r = max_circular_rec(nums3)
    mc_m = max_circular_memo(nums3)
    mc_t = max_circular_dp(nums3)
    mc_o = max_circular_optimized(nums3)
    # Also verify all-negative corner case
    nums3_neg = [-3, -2, -3]
    assert max_circular_optimized(nums3_neg) == -2
    print(f"3. Maximum Sum Circular Subarray({nums3}):")
    print(f"   Rec: {mc_r} | Memo: {mc_m} | Tab: {mc_t} | Opt O(1): {mc_o}")
    assert mc_r == mc_m == mc_t == mc_o == 10
    print("   [SUCCESS] All Maximum Circular Subarray variants returned 10 (and -2 for all-negative).\n")

    # 4. Longest Turbulent Subarray
    nums4 = [9, 4, 2, 10, 7, 8, 8, 1, 9]
    # Turbulent segment: [4, 2, 10, 7, 8] -> 4 > 2 < 10 > 7 < 8 (length 5)
    mt_r = max_turbulent_rec(nums4)
    mt_m = max_turbulent_memo(nums4)
    mt_t = max_turbulent_dp(nums4)
    mt_o = max_turbulent_optimized(nums4)
    print(f"4. Longest Turbulent Subarray({nums4}):")
    print(f"   Rec: {mt_r} | Memo: {mt_m} | Tab: {mt_t} | Opt O(1): {mt_o}")
    assert mt_r == mt_m == mt_t == mt_o == 5
    print("   [SUCCESS] All Longest Turbulent Subarray variants returned 5.\n")

    # 5. Maximum Subarray Sum with One Deletion
    nums5 = [1, -2, 0, 3]
    # Delete -2 from [1, -2, 0, 3] -> [1, 0, 3] has sum 4
    md_r = max_subarray_one_del_rec(nums5)
    md_m = max_subarray_one_del_memo(nums5)
    md_t = max_subarray_one_del_dp(nums5)
    md_o = max_subarray_one_del_optimized(nums5)
    print(f"5. Maximum Subarray Sum with One Deletion({nums5}):")
    print(f"   Rec: {md_r} | Memo: {md_m} | Tab: {md_t} | Opt O(1): {md_o}")
    assert md_r == md_m == md_t == md_o == 4
    print("   [SUCCESS] All Max Subarray with One Deletion variants returned 4.\n")

    print("=" * 78)
    print("[ALL TESTS PASSED] Kadane's Algorithm & Contiguous Subarray DP verified!")
    print("=" * 78)
