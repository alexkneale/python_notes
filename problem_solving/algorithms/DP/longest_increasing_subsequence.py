"""
================================================================================
DYNAMIC PROGRAMMING: LONGEST INCREASING SUBSEQUENCE (LIS) & PREDECESSOR CHAINS
================================================================================

TABLE OF CONTENTS:
1. RECAP & THE CORE INTUITION: THE 1D PREDECESSOR LOOKBACK PATTERN
2. THE ARCHETYPE (LIS): RECURSION, MEMOIZATION, TABULATION & O(N LOG N) PATIENCE SORT
3. THE PROGRESSION ENGINE: HOW PREDECESSOR MATCHING EXTENDS ACROSS PROBLEMS
4. VARIATION 1: MAXIMUM SUM INCREASING SUBSEQUENCE (MSIS)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation)
5. VARIATION 2: NUMBER OF LONGEST INCREASING SUBSEQUENCES (COUNTING PATHS)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Dual-Array Tabulation)
6. VARIATION 3: RUSSIAN DOLL ENVELOPES (2D SORTING REDUCTION TO LIS)
   - The Crucial Multi-Key Sort Insight, Bottom-Up O(N^2), Optimal O(N log N)
7. BONUS: RECONSTRUCTING THE ACTUAL LIS (BACKTRACKING & PARENT POINTERS)
8. COMPARATIVE SUMMARY TABLE: TRANSITIONS, STATES & COMPLEXITIES
9. RUNNABLE TESTS & VERIFICATION
================================================================================
"""

import bisect
from typing import Dict, List, Tuple


# ==============================================================================
# SECTION 1: THE CORE INTUITION: THE 1D PREDECESSOR LOOKBACK PATTERN
# ==============================================================================
"""
THE 1D PREDECESSOR LOOKBACK PATTERN EXPLAINED:
---------------------------------------------
Unlike two-sequence problems (LCS/Edit Distance) that scan a 2D coordinate grid (i, j),
subsequence-forming problems on a single 1D array typically use a "Lookback" pattern:

1. State Definition (Ending at index i):
   - Let `dp[i]` denote the optimal answer for a valid subsequence that *ends strictly
     at element nums[i]*.
   - By anchoring the subsequence endpoint to index `i`, we can make a local decision:
     an incoming element `nums[i]` can extend *any* valid prior subsequence ending at `j`
     (where 0 <= j < i) as long as the transition constraint (e.g., `nums[j] < nums[i]`)
     is satisfied.

2. The Lookback Dependency:
   To compute `dp[i]`, we inspect every candidate predecessor `j < i`:
     `dp[i] = 1 + max([dp[j] for j in range(i) if nums[j] < nums[i]] or [0])`

3. The Global Result:
   Because the optimal subsequence could end at any arbitrary index, the final answer
   is NOT simply `dp[n - 1]`. It is `max(dp[0...n-1])`.

4. The Dual Representation:
   - O(N^2) Tabulation: Ideal when weights, counts, or arbitrary local conditions are tracked.
   - O(N log N) Patience Sorting: Maintains the smallest possible tail values of all increasing
     subsequences of each length discovered so far. Binary search (`bisect_left`) locates where
     each new number fits.
"""


# ==============================================================================
# SECTION 2: THE ARCHETYPE: LONGEST INCREASING SUBSEQUENCE (LIS)
# ==============================================================================
"""
Problem:
Given an integer array `nums`, return the length of the longest strictly increasing
subsequence.

State Definition (Bottom-Up Tabulation):
`dp[i]` = length of the longest strictly increasing subsequence ending at index `i`.

Recurrence:
For each i from 0 to n - 1:
    dp[i] = 1  # Base case: single element
    For each j from 0 to i - 1:
        if nums[j] < nums[i]:
            dp[i] = max(dp[i], 1 + dp[j])

Answer: max(dp)
"""

# ------------------------------------------------------------------------------
# 2.1 Pure Recursion
# ------------------------------------------------------------------------------
def lis_recursive(nums: List[int]) -> int:
    """
    Pure Recursive Solution (Include / Exclude choice tree).
    Time:  O(2^n) - Binary branching at every index.
    Space: O(n)   - Maximum recursion stack depth.
    """
    def helper(curr_idx: int, prev_idx: int) -> int:
        # Base case: processed all elements
        if curr_idx == len(nums):
            return 0

        # Choice 1: Exclude nums[curr_idx]
        not_taken = helper(curr_idx + 1, prev_idx)

        # Choice 2: Include nums[curr_idx] (if strictly greater than previous chosen)
        taken = 0
        if prev_idx == -1 or nums[curr_idx] > nums[prev_idx]:
            taken = 1 + helper(curr_idx + 1, curr_idx)

        return max(taken, not_taken)

    return helper(0, -1)


# ------------------------------------------------------------------------------
# 2.2 Top-Down DP (Memoization)
# ------------------------------------------------------------------------------
def lis_memoization(nums: List[int]) -> int:
    """
    Top-Down DP with Hash Map / State Memoization.
    State: (curr_idx, prev_idx) where prev_idx is the index of the last included element.
    Time:  O(n^2) - Unique states: n * (n + 1).
    Space: O(n^2) - Memo table + recursion call stack.
    """
    memo: Dict[Tuple[int, int], int] = {}

    def helper(curr_idx: int, prev_idx: int) -> int:
        if curr_idx == len(nums):
            return 0

        state = (curr_idx, prev_idx)
        if state in memo:
            return memo[state]

        # Branch 1: Exclude
        res = helper(curr_idx + 1, prev_idx)

        # Branch 2: Include
        if prev_idx == -1 or nums[curr_idx] > nums[prev_idx]:
            res = max(res, 1 + helper(curr_idx + 1, curr_idx))

        memo[state] = res
        return res

    return helper(0, -1)


# ------------------------------------------------------------------------------
# 2.3 Bottom-Up DP (Tabulation - O(N^2))
# ------------------------------------------------------------------------------
def lis_tabulation(nums: List[int]) -> int:
    """
    Bottom-Up Tabulation (1D Predecessor Lookback).
    Time:  O(n^2) - Nested loop iterating over all pairs (j, i) with j < i.
    Space: O(n)   - Array of size n.
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # Every single element is an increasing subsequence of length 1

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], 1 + dp[j])

    return max(dp)


# ------------------------------------------------------------------------------
# 2.4 Bottom-Up DP (Optimal O(N log N) via Patience Sorting / Binary Search)
# ------------------------------------------------------------------------------
def lis_patience_sorting(nums: List[int]) -> int:
    """
    Patience Sorting / Monotonic Tails Array with Binary Search.
    `tails[k]` stores the smallest tail of all increasing subsequences of length (k + 1).

    For each number x in nums:
      - If x is larger than all tails, append it (we discovered a longer subsequence).
      - Otherwise, update the smallest tail that is >= x using bisect_left.
        (Greedy: smaller tails make it easier for future elements to extend the sequence).

    Time:  O(n log n) - Binary search takes O(log n) for each of the n elements.
    Space: O(n)       - `tails` array stores at most n elements.
    """
    tails: List[int] = []

    for x in nums:
        idx = bisect.bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x

    return len(tails)


# ==============================================================================
# SECTION 3: THE PROGRESSION ENGINE: HOW PREDECESSOR MATCHING EXTENDS
# ==============================================================================
"""
The 1D Lookback Engine:
  For each element at index i, look back at all elements at j (0 <= j < i).
  If a relationship condition holds (nums[j] < nums[i]), transition state from j to i.

How Variations Modify the Engine:

1. MAXIMUM SUM INCREASING SUBSEQUENCE (MSIS):
   - Goal: Instead of maximizing length (+1 per hop), maximize accumulated sum (+nums[i]).
   - Recurrence: `dp[i] = nums[i] + max([dp[j] for j < i if nums[j] < nums[i]] or [0])`.

2. NUMBER OF LONGEST INCREASING SUBSEQUENCES (COUNTING LIS):
   - Goal: Count how many distinct subsequences achieve the maximum length.
   - Dual State: Maintain `length[i]` AND `count[i]`.
     * If `length[j] + 1 > length[i]`: Reset `length[i]` and adopt `count[i] = count[j]`.
     * If `length[j] + 1 == length[i]`: Accumulate alternative paths: `count[i] += count[j]`.

3. RUSSIAN DOLL ENVELOPES (2D REDUCTION TO LIS):
   - Goal: Fit envelopes inside each other (both width and height must strictly increase).
   - Insight: Sort width ascending, and height DESCENDING for ties.
     Descending height prevents pairing envelopes of the same width.
   - Reduces directly to standard 1D LIS on the heights!
"""


# ==============================================================================
# SECTION 4: VARIATION 1: MAXIMUM SUM INCREASING SUBSEQUENCE (MSIS)
# ==============================================================================
"""
Problem:
Given an array of positive integers, find the maximum possible sum of an
increasing subsequence.

State Definition:
`dp[i]` = maximum sum of an increasing subsequence that ends strictly at index `i`.

Recurrence:
`dp[i] = nums[i] + max([dp[j] for j in range(i) if nums[j] < nums[i]] or [0])`
Global Answer = max(dp)
"""

def msis_rec(nums: List[int]) -> int:
    """Pure Recursion: O(2^n) time, O(n) space."""
    def helper(curr_idx: int, prev_idx: int) -> int:
        if curr_idx == len(nums):
            return 0

        # Choice 1: Exclude
        not_taken = helper(curr_idx + 1, prev_idx)

        # Choice 2: Include
        taken = 0
        if prev_idx == -1 or nums[curr_idx] > nums[prev_idx]:
            taken = nums[curr_idx] + helper(curr_idx + 1, curr_idx)

        return max(taken, not_taken)

    return helper(0, -1)


def msis_memo(nums: List[int]) -> int:
    """Top-Down Memoization: O(n^2) time, O(n^2) space."""
    memo: Dict[Tuple[int, int], int] = {}

    def helper(curr_idx: int, prev_idx: int) -> int:
        if curr_idx == len(nums):
            return 0

        state = (curr_idx, prev_idx)
        if state in memo:
            return memo[state]

        res = helper(curr_idx + 1, prev_idx)
        if prev_idx == -1 or nums[curr_idx] > nums[prev_idx]:
            res = max(res, nums[curr_idx] + helper(curr_idx + 1, curr_idx))

        memo[state] = res
        return res

    return helper(0, -1)


def msis_dp(nums: List[int]) -> int:
    """
    Bottom-Up Tabulation: O(n^2) time, O(n) space.
    Note: Patience sorting cannot directly apply here because optimal sum does not
    satisfy the monotonic tail property. Tabulation is the standard approach.
    """
    if not nums:
        return 0

    n = len(nums)
    dp = list(nums)  # Base case: each element alone forms a sum equal to its own value

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + nums[i])

    return max(dp)


# ==============================================================================
# SECTION 5: VARIATION 2: NUMBER OF LONGEST INCREASING SUBSEQUENCES (COUNTING LIS)
# ==============================================================================
"""
Problem:
Given an integer array `nums`, return the number of longest increasing subsequences.

Dual-Array State Definition:
- `lengths[i]`: length of LIS ending at index `i`.
- `counts[i]`: number of distinct LIS ending at index `i`.

Transition for each j < i where nums[j] < nums[i]:
  - If lengths[j] + 1 > lengths[i]:
      lengths[i] = lengths[j] + 1
      counts[i] = counts[j]           # Found a strictly longer chain; reset counts
  - Else if lengths[j] + 1 == lengths[i]:
      counts[i] += counts[j]          # Found an alternative path of the same length
"""

def find_number_of_lis_rec(nums: List[int]) -> int:
    """
    Recursive helper returning (max_len_from_here, count_of_that_len).
    O(2^n) time, O(n) call stack.
    """
    def helper(i: int) -> Tuple[int, int]:
        max_len = 1
        count = 1
        for j in range(i + 1, len(nums)):
            if nums[j] > nums[i]:
                length_j, count_j = helper(j)
                if length_j + 1 > max_len:
                    max_len = length_j + 1
                    count = count_j
                elif length_j + 1 == max_len:
                    count += count_j
        return max_len, count

    if not nums:
        return 0

    overall_max_len = 0
    total_count = 0
    for i in range(len(nums)):
        length_i, count_i = helper(i)
        if length_i > overall_max_len:
            overall_max_len = length_i
            total_count = count_i
        elif length_i == overall_max_len:
            total_count += count_i

    return total_count


def find_number_of_lis_memo(nums: List[int]) -> int:
    """Top-Down Memoization: O(n^2) time, O(n) space."""
    memo: Dict[int, Tuple[int, int]] = {}

    def helper(i: int) -> Tuple[int, int]:
        if i in memo:
            return memo[i]

        max_len = 1
        count = 1
        for j in range(i + 1, len(nums)):
            if nums[j] > nums[i]:
                length_j, count_j = helper(j)
                if length_j + 1 > max_len:
                    max_len = length_j + 1
                    count = count_j
                elif length_j + 1 == max_len:
                    count += count_j

        memo[i] = (max_len, count)
        return memo[i]

    if not nums:
        return 0

    overall_max_len = 0
    total_count = 0
    for i in range(len(nums)):
        length_i, count_i = helper(i)
        if length_i > overall_max_len:
            overall_max_len = length_i
            total_count = count_i
        elif length_i == overall_max_len:
            total_count += count_i

    return total_count


def find_number_of_lis_dp(nums: List[int]) -> int:
    """Bottom-Up Dual-Array Tabulation: O(n^2) time, O(n) space."""
    if not nums:
        return 0

    n = len(nums)
    lengths = [1] * n
    counts = [1] * n

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]

    max_len = max(lengths)
    return sum(counts[i] for i in range(n) if lengths[i] == max_len)


# ==============================================================================
# SECTION 6: VARIATION 3: RUSSIAN DOLL ENVELOPES (2D REDUCTION TO LIS)
# ==============================================================================
"""
Problem:
You are given a 2D array of integers `envelopes` where envelopes[i] = [w_i, h_i].
One envelope can fit inside another if and only if BOTH width and height are strictly
greater. Return the maximum number of envelopes you can Russian doll (nest).

Key Insight: Multi-Key Sorting Trick:
-------------------------------------
1. Sort envelopes by width ASCENDING.
2. For envelopes with identical widths, sort their heights in DESCENDING order!

Why descending height for ties?
- Suppose we have [3, 4] and [3, 5]. Neither can fit inside the other because their
  widths are equal.
- If we sorted heights ascending ([3, 4], [3, 5]), an LIS scan on heights would see
  4 < 5 and falsely consider them nestable!
- By sorting heights descending ([3, 5], [3, 4]), the number 5 comes before 4,
  making it impossible for 5 to precede 4 in an increasing subsequence of heights.

Once sorted by `(width ASC, height DESC)`, the problem reduces entirely to standard 1D LIS
on the height dimensions!
"""

def max_envelopes_tabulation(envelopes: List[List[int]]) -> int:
    """
    Bottom-Up Tabulation after sorting: O(n^2) time, O(n) space.
    (Note: LeetCode 354 gives TLE for O(N^2); optimal O(N log N) is required).
    """
    if not envelopes:
        return 0

    # Sort: width ascending, height descending for ties
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    heights = [h for _, h in envelopes]

    return lis_tabulation(heights)


def max_envelopes_optimal(envelopes: List[List[int]]) -> int:
    """
    Optimal O(N log N) using 2D sorting + Patience Sorting on heights.
    Time:  O(n log n) - Sorting takes O(n log n), binary search takes O(n log n).
    Space: O(n)       - Array of tails.
    """
    if not envelopes:
        return 0

    # Sort: width ascending, height descending for ties
    envelopes.sort(key=lambda x: (x[0], -x[1]))

    # Run patience sorting on heights
    tails: List[int] = []
    for _, h in envelopes:
        idx = bisect.bisect_left(tails, h)
        if idx == len(tails):
            tails.append(h)
        else:
            tails[idx] = h

    return len(tails)


# ==============================================================================
# SECTION 7: BONUS: RECONSTRUCTING THE ACTUAL LIS (BACKTRACKING / PARENT POINTERS)
# ==============================================================================
"""
Problem:
Instead of only returning the length, return one concrete Longest Increasing Subsequence.

Mechanism:
Maintain a `parent` array where `parent[i]` stores the index `j` of the element
that preceded `nums[i]` in the optimal chain.
"""

def reconstruct_lis(nums: List[int]) -> List[int]:
    """
    Tabulation with Parent Pointers: O(n^2) time, O(n) space.
    """
    if not nums:
        return []

    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    best_len = 1
    best_idx = 0

    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

        if dp[i] > best_len:
            best_len = dp[i]
            best_idx = i

    # Trace back through parent pointers
    sequence: List[int] = []
    curr = best_idx
    while curr != -1:
        sequence.append(nums[curr])
        curr = parent[curr]

    return sequence[::-1]


# ==============================================================================
# SECTION 8: COMPARATIVE SUMMARY TABLE: TRANSITIONS, STATES & COMPLEXITIES
# ==============================================================================
"""
+-----------------------------+-----------------------+-----------------------------------------------+---------------+
| Problem                     | State Representation  | Transition Relation                           | Time / Space  |
+-----------------------------+-----------------------+-----------------------------------------------+---------------+
| Longest Increasing Subseq   | dp[i]: length ending  | dp[i] = 1 + max(dp[j])                        | O(N^2) Tab    |
| (LIS Archetype)             | at index i            | for all j < i if nums[j] < nums[i]            | O(N log N) BS |
+-----------------------------+-----------------------+-----------------------------------------------+---------------+
| Maximum Sum Increasing      | dp[i]: max sum ending | dp[i] = nums[i] + max(dp[j])                  | O(N^2) / O(N) |
| Subsequence (MSIS)          | at index i            | for all j < i if nums[j] < nums[i]            |               |
+-----------------------------+-----------------------+-----------------------------------------------+---------------+
| Number of LIS               | len[i]: length        | If len[j] + 1 > len[i]  -> len[i]=len[j]+1;   | O(N^2) / O(N) |
| (Counting Paths)            | cnt[i]: count of LIS  |                           cnt[i]=cnt[j]       |               |
|                             | ending at index i     | If len[j] + 1 == len[i] -> cnt[i]+=cnt[j]     |               |
+-----------------------------+-----------------------+-----------------------------------------------+---------------+
| Russian Doll Envelopes      | Reduction to LIS      | Sort: key=(w ASC, h DESC)                     | O(N log N) /  |
|                             | on heights array      | Run patience sorting on heights array         | O(N)          |
+-----------------------------+-----------------------+-----------------------------------------------+---------------+
"""


# ==============================================================================
# SECTION 9: RUNNABLE TESTS & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING 1D LIS & PREDECESSOR CHAIN DP: RECURSION vs MEMO vs TABULATION vs O(N LOG N)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Longest Increasing Subsequence (LIS) Archetype
    # --------------------------------------------------------------------------
    nums_test = [10, 9, 2, 5, 3, 7, 101, 18]
    # One LIS is [2, 3, 7, 101] or [2, 5, 7, 101] -> length 4
    lis_r = lis_recursive(nums_test)
    lis_m = lis_memoization(nums_test)
    lis_t = lis_tabulation(nums_test)
    lis_o = lis_patience_sorting(nums_test)

    print(f"LIS({nums_test}):")
    print(f"  Rec: {lis_r} | Memo: {lis_m} | Tab: {lis_t} | Patience Sort O(N log N): {lis_o}")
    assert lis_r == lis_m == lis_t == lis_o == 4
    print("  [SUCCESS] All 4 LIS implementations returned length 4.\n")

    # --------------------------------------------------------------------------
    # 2. Maximum Sum Increasing Subsequence (MSIS)
    # --------------------------------------------------------------------------
    msis_test = [1, 101, 2, 3, 100, 4, 5]
    # LIS is [1, 2, 3, 4, 5] (length 5, sum 15)
    # But MSIS is [1, 2, 3, 100] -> sum = 106
    msis_r = msis_rec(msis_test)
    msis_m = msis_memo(msis_test)
    msis_t = msis_dp(msis_test)

    print(f"MSIS({msis_test}):")
    print(f"  Rec: {msis_r} | Memo: {msis_m} | Tab: {msis_t}")
    assert msis_r == msis_m == msis_t == 106
    print("  [SUCCESS] All MSIS variants returned sum 106.\n")

    # --------------------------------------------------------------------------
    # 3. Number of Longest Increasing Subsequences (Counting LIS)
    # --------------------------------------------------------------------------
    count_test = [1, 3, 5, 4, 7]
    # LIS length is 4. The 2 valid subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].
    cnt_r = find_number_of_lis_rec(count_test)
    cnt_m = find_number_of_lis_memo(count_test)
    cnt_t = find_number_of_lis_dp(count_test)

    print(f"Number of LIS({count_test}):")
    print(f"  Rec: {cnt_r} | Memo: {cnt_m} | Tab: {cnt_t}")
    assert cnt_r == cnt_m == cnt_t == 2

    # Edge test: duplicates (all equal)
    dup_test = [2, 2, 2, 2, 2]
    # LIS length is 1; each element individually forms an LIS -> count = 5
    assert find_number_of_lis_dp(dup_test) == 5
    print("  [SUCCESS] Number of LIS verified for standard and duplicate test cases.\n")

    # --------------------------------------------------------------------------
    # 4. Russian Doll Envelopes
    # --------------------------------------------------------------------------
    envelopes_test = [[5, 4], [6, 4], [6, 7], [2, 3]]
    # Optimal nesting: [2, 3] => [5, 4] => [6, 7] (length 3)
    # Notice [6, 4] cannot nest with [6, 7] because width is identical!
    env_t = max_envelopes_tabulation(list(envelopes_test))
    env_o = max_envelopes_optimal(list(envelopes_test))

    print(f"Russian Doll Envelopes({envelopes_test}):")
    print(f"  Tabulation O(N^2): {env_t} | Optimal O(N log N): {env_o}")
    assert env_t == env_o == 3
    print("  [SUCCESS] Russian Doll Envelopes returned 3.\n")

    # --------------------------------------------------------------------------
    # 5. Reconstructing the Actual LIS Sequence
    # --------------------------------------------------------------------------
    recon_input = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    recovered_seq = reconstruct_lis(recon_input)
    print(f"Reconstructed LIS for {recon_input}:")
    print(f"  Result: {recovered_seq} (Length: {len(recovered_seq)})")
    # Verify strict increase
    assert all(recovered_seq[i] < recovered_seq[i + 1] for i in range(len(recovered_seq) - 1))
    assert len(recovered_seq) == 6  # [10, 22, 33, 50, 60, 80] or similar
    print("  [SUCCESS] LIS reconstruction correctly recovered optimal chain.\n")

    print("=" * 80)
    print("[ALL TESTS PASSED] LIS & Predecessor Chain DP module verified successfully!")
    print("=" * 80)