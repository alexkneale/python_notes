"""
================================================================================
DYNAMIC PROGRAMMING: LONGEST COMMON SUBSEQUENCE (LCS) & TWO-SEQUENCE ALIGNMENT
================================================================================

TABLE OF CONTENTS:
1. RECAP & THE CORE INTUITION: THE 2D TWO-SEQUENCE GRID PATTERN
2. THE ARCHETYPE (LCS): RECURSION, MEMOIZATION, TABULATION & O(N) SPACE
3. THE PROGRESSION ENGINE: MATCH VS. MISMATCH BRANCHING
4. VARIATION 1: LONGEST COMMON SUBSTRING (CONTIGUOUS RESET PATTERN)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(N) Space)
5. VARIATION 2: EDIT DISTANCE / LEVENSHTEIN (INSERT, DELETE, REPLACE)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(N) Space)
6. VARIATION 3: SHORTEST COMMON SUPERSEQUENCE (LCS COMPLEMENT)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(N) Space)
7. VARIATION 4: LONGEST PALINDROMIC SUBSEQUENCE (REDUCTION TO LCS)
   - Naive Recursion, Top-Down Memoization, Bottom-Up DP (Tabulation & O(N) Space)
8. COMPARATIVE SUMMARY TABLE: TRANSITIONS & BASE CASES
9. RUNNABLE TESTS & VERIFICATION
================================================================================
"""

from typing import Dict, Tuple, List


# ==============================================================================
# SECTION 1: THE CORE INTUITION: THE 2D TWO-SEQUENCE GRID PATTERN
# ==============================================================================
"""
THE 2D TWO-SEQUENCE PATTERN EXPLAINED:
-------------------------------------
When comparing two sequences (strings or arrays) `s1` of length `m` and `s2`
of length `n`, we track two independent pointers: `i` for `s1` and `j` for `s2`.

Key Characteristics:
1. The state is represented as a 2-tuple: `(i, j)`.
   In prefix notation:
     `dp[i][j]` represents the answer considering `s1[0...i-1]` and `s2[0...j-1]`.
   In suffix notation:
     `dp(i, j)` represents the answer considering suffixes `s1[i...]` and `s2[j...]`.

2. The Grid Dependency:
   To compute state `(i, j)`, we typically look at three adjacent neighbors:
     - Diagonal `(i - 1, j - 1)`: Consume characters from both sequences simultaneously.
     - Top `(i - 1, j)`: Consume or delete from `s1`, leaving `s2` unchanged.
     - Left `(i, j - 1)`: Consume or delete from `s2`, leaving `s1` unchanged.

3. Space Optimization:
   Because row `i` depends strictly on row `i - 1` and row `i`:
   - Full Tabulation requires an `O(m * n)` 2D matrix.
   - We can compress auxiliary space to `O(n)` using two 1D rows (`prev_row`
     and `curr_row`), or even a single 1D row with a rolling diagonal variable.
"""


# ==============================================================================
# SECTION 2: THE ARCHETYPE: LONGEST COMMON SUBSEQUENCE (LCS)
# ==============================================================================
"""
Problem:
Given two strings `text1` and `text2`, return the length of their longest common
subsequence (a subsequence does not have to be contiguous, but must preserve order).

State Definition (Prefix):
`dp[i][j]` = length of LCS of `text1[0...i-1]` and `text2[0...j-1]`.

Recurrence:
If text1[i - 1] == text2[j - 1]:
    dp[i][j] = 1 + dp[i - 1][j - 1]     (Diagonal: Match found!)
Else:
    dp[i][j] = max(dp[i - 1][j],        (Top: Skip char in text1)
                   dp[i][j - 1])        (Left: Skip char in text2)
"""

# ------------------------------------------------------------------------------
# 2.1 Pure Recursion
# ------------------------------------------------------------------------------
def lcs_recursive(text1: str, text2: str) -> int:
    """
    Pure Recursive Solution (Suffix matching).
    Time:  O(2^(m + n)) - Worst-case binary branching without caching.
    Space: O(m + n)     - Maximum recursion stack depth.
    """
    def helper(i: int, j: int) -> int:
        # Base case: either string is exhausted
        if i == len(text1) or j == len(text2):
            return 0

        # Match: consume both characters
        if text1[i] == text2[j]:
            return 1 + helper(i + 1, j + 1)

        # Mismatch: branch by skipping one character from either string
        return max(helper(i + 1, j), helper(i, j + 1))

    return helper(0, 0)


# ------------------------------------------------------------------------------
# 2.2 Top-Down DP (Memoization)
# ------------------------------------------------------------------------------
def lcs_memoization(text1: str, text2: str) -> int:
    """
    Top-Down DP with Hash Map / Table Memoization.
    Time:  O(m * n) - Each unique pair (i, j) solved exactly once.
    Space: O(m * n) - Call stack O(m + n) + memoization store O(m * n).
    """
    memo: Dict[Tuple[int, int], int] = {}

    def helper(i: int, j: int) -> int:
        if i == len(text1) or j == len(text2):
            return 0

        state = (i, j)
        if state in memo:
            return memo[state]

        if text1[i] == text2[j]:
            res = 1 + helper(i + 1, j + 1)
        else:
            res = max(helper(i + 1, j), helper(i, j + 1))

        memo[state] = res
        return res

    return helper(0, 0)


# ------------------------------------------------------------------------------
# 2.3 Bottom-Up DP (Tabulation)
# ------------------------------------------------------------------------------
def lcs_tabulation(text1: str, text2: str) -> int:
    """
    Bottom-Up DP (2D Table).
    Time:  O(m * n)
    Space: O(m * n)
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


# ------------------------------------------------------------------------------
# 2.4 Bottom-Up DP (Space-Optimized to 2 Rows)
# ------------------------------------------------------------------------------
def lcs_space_optimized(text1: str, text2: str) -> int:
    """
    Bottom-Up DP with Rolling 1D Rows.
    Time:  O(m * n)
    Space: O(min(m, n)) - Maintaining two rows of size n + 1.
    """
    # Optimization: Ensure text2 is the shorter string to minimize space
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
        prev = curr[:]

    return prev[n]


# ==============================================================================
# SECTION 3: THE PROGRESSION ENGINE: MATCH VS. MISMATCH BRANCHING
# ==============================================================================
"""
All two-sequence problems navigate a 2D grid `(i, j)` where each cell decides
between diagonal traversal (pairing elements) or horizontal/vertical traversal
(skipping, deleting, or inserting elements):

1. MATCH VS MISMATCH (LCS):
   - Match:   Take diagonal + 1
   - Mismatch: max(top, left)

2. CONTIGUITY CONSTRAINTS (Longest Common Substring):
   - Match:   Take diagonal + 1
   - Mismatch: RESET to 0 (subarrays must be unbroken)

3. THREE-WAY COST MINIMIZATION (Edit Distance / Levenshtein):
   - Match:   Diagonal + 0 cost
   - Mismatch: 1 + min(Replace [diag], Delete [top], Insert [left])

4. PALINDROMIC ADAPTATION (LPS):
   - Reducible to LCS of string `s` and its reverse `reversed(s)`.
"""


# ==============================================================================
# SECTION 4: VARIATION 1: LONGEST COMMON SUBSTRING
# ==============================================================================
"""
Problem:
Given two strings `text1` and `text2`, find the length of the longest COMMON
SUBSTRING. Unlike subsequences, substrings MUST BE CONTIGUOUS in both strings.

State Definition:
`dp[i][j]` = length of the longest common suffix of `text1[0...i-1]` and `text2[0...j-1]`.

Recurrence:
If text1[i - 1] == text2[j - 1]:
    dp[i][j] = 1 + dp[i - 1][j - 1]
Else:
    dp[i][j] = 0   <-- RESET TO ZERO! Breaks contiguity.

The global answer is max(dp[i][j]) over all 1 <= i <= m, 1 <= j <= n.
"""

def lc_substring_rec(text1: str, text2: str) -> int:
    """Pure Recursion: O(2^(m+n)) time, O(m+n) space."""
    def helper(i: int, j: int, current_count: int) -> int:
        if i == len(text1) or j == len(text2):
            return current_count

        matched_count = current_count
        if text1[i] == text2[j]:
            matched_count = helper(i + 1, j + 1, current_count + 1)

        # Explore branches where we break contiguity
        skip_1 = helper(i + 1, j, 0)
        skip_2 = helper(i, j + 1, 0)

        return max(matched_count, skip_1, skip_2)

    return helper(0, 0, 0)


def lc_substring_memo(text1: str, text2: str) -> int:
    """Top-Down Memoization: O(m * n) time, O(m * n) space."""
    memo: Dict[Tuple[int, int], int] = {}
    max_len = 0

    def helper(i: int, j: int) -> int:
        nonlocal max_len
        if i == len(text1) or j == len(text2):
            return 0

        state = (i, j)
        if state in memo:
            return memo[state]

        curr = 0
        if text1[i] == text2[j]:
            curr = 1 + helper(i + 1, j + 1)
            max_len = max(max_len, curr)

        # Invoke search on branches so all states are visited
        helper(i + 1, j)
        helper(i, j + 1)

        memo[state] = curr
        return curr

    helper(0, 0)
    return max_len


def lc_substring_dp(text1: str, text2: str) -> int:
    """Bottom-Up Tabulation: O(m * n) time, O(m * n) space."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0  # Contiguity reset

    return max_len


def lc_substring_optimized(text1: str, text2: str) -> int:
    """Bottom-Up Rolling Array: O(m * n) time, O(min(m, n)) space."""
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    max_len = 0

    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = 1 + prev[j - 1]
                max_len = max(max_len, curr[j])
            else:
                curr[j] = 0
        prev = curr

    return max_len


# ==============================================================================
# SECTION 5: VARIATION 2: EDIT DISTANCE (LEVENSHTEIN DISTANCE)
# ==============================================================================
"""
Problem:
Given two strings `word1` and `word2`, return the MINIMUM number of operations
required to convert `word1` to `word2`.
Permitted operations:
  1. Insert a character
  2. Delete a character
  3. Replace a character

State Definition:
`dp[i][j]` = min operations to convert `word1[0...i-1]` to `word2[0...j-1]`.

Base Cases:
- dp[0][j] = j (Must insert `j` characters to match prefix of word2)
- dp[i][0] = i (Must delete `i` characters to reach empty string)

Transition:
If word1[i - 1] == word2[j - 1]:
    dp[i][j] = dp[i - 1][j - 1]  (Characters match, no operation needed)
Else:
    dp[i][j] = 1 + min(
        dp[i - 1][j - 1],        (Replace word1[i - 1] with word2[j - 1])
        dp[i - 1][j],            (Delete word1[i - 1])
        dp[i][j - 1]             (Insert word2[j - 1] into word1)
    )
"""

def edit_distance_rec(word1: str, word2: str) -> int:
    """Pure Recursion: O(3^(m+n)) time, O(m+n) space."""
    def helper(i: int, j: int) -> int:
        if i == 0:
            return j  # Insert remaining j characters
        if j == 0:
            return i  # Delete remaining i characters

        if word1[i - 1] == word2[j - 1]:
            return helper(i - 1, j - 1)

        return 1 + min(
            helper(i - 1, j - 1),  # Replace
            helper(i - 1, j),      # Delete
            helper(i, j - 1)       # Insert
        )

    return helper(len(word1), len(word2))


def edit_distance_memo(word1: str, word2: str) -> int:
    """Top-Down Memoization: O(m * n) time, O(m * n) space."""
    memo: Dict[Tuple[int, int], int] = {}

    def helper(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i

        state = (i, j)
        if state in memo:
            return memo[state]

        if word1[i - 1] == word2[j - 1]:
            res = helper(i - 1, j - 1)
        else:
            res = 1 + min(
                helper(i - 1, j - 1),  # Replace
                helper(i - 1, j),      # Delete
                helper(i, j - 1)       # Insert
            )

        memo[state] = res
        return res

    return helper(len(word1), len(word2))


def edit_distance_dp(word1: str, word2: str) -> int:
    """Bottom-Up Tabulation: O(m * n) time, O(m * n) space."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base case initialization
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # Replace
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1]       # Insert
                )

    return dp[m][n]


def edit_distance_optimized(word1: str, word2: str) -> int:
    """Bottom-Up Rolling Array: O(m * n) time, O(n) space."""
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        curr[0] = i
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(
                    prev[j - 1],  # Replace
                    prev[j],      # Delete
                    curr[j - 1]   # Insert
                )
        prev = curr

    return prev[n]


# ==============================================================================
# SECTION 6: VARIATION 3: SHORTEST COMMON SUPERSEQUENCE (SCS)
# ==============================================================================
"""
Problem:
Given two strings `str1` and `str2`, return the LENGTH of the shortest string
that has both `str1` and `str2` as subsequences.

Mathematical Equivalence:
When combining two strings into a minimal supersequence, any characters shared
in their Longest Common Subsequence (LCS) only need to be written ONCE.
All other non-shared characters from `str1` and `str2` must each be included.

Formula:
  Length(SCS) = len(str1) + len(str2) - Length(LCS(str1, str2))

Direct DP Recurrence:
If str1[i - 1] == str2[j - 1]:
    dp[i][j] = 1 + dp[i - 1][j - 1]
Else:
    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1])
"""

def scs_length_rec(str1: str, str2: str) -> int:
    """Pure Recursion: O(2^(m+n)) time, O(m+n) space."""
    def helper(i: int, j: int) -> int:
        # If one string is empty, we must take all remaining characters of the other
        if i == 0:
            return j
        if j == 0:
            return i

        if str1[i - 1] == str2[j - 1]:
            return 1 + helper(i - 1, j - 1)

        return 1 + min(helper(i - 1, j), helper(i, j - 1))

    return helper(len(str1), len(str2))


def scs_length_memo(str1: str, str2: str) -> int:
    """Top-Down Memoization: O(m * n) time, O(m * n) space."""
    memo: Dict[Tuple[int, int], int] = {}

    def helper(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i

        state = (i, j)
        if state in memo:
            return memo[state]

        if str1[i - 1] == str2[j - 1]:
            res = 1 + helper(i - 1, j - 1)
        else:
            res = 1 + min(helper(i - 1, j), helper(i, j - 1))

        memo[state] = res
        return res

    return helper(len(str1), len(str2))


def scs_length_dp(str1: str, str2: str) -> int:
    """Bottom-Up Tabulation: O(m * n) time, O(m * n) space."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def scs_length_optimized(str1: str, str2: str) -> int:
    """Bottom-Up Rolling Array: O(m * n) time, O(n) space."""
    m, n = len(str1), len(str2)
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        curr[0] = i
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1])
        prev = curr

    return prev[n]


def scs_reconstruct_string(str1: str, str2: str) -> str:
    """
    Reconstructs the actual SCS string by backtracking through the LCS table.
    Time: O(m * n), Space: O(m * n).
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtracking from dp[m][n] to dp[0][0]
    i, j = m, n
    result_chars: List[str] = []

    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            result_chars.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            result_chars.append(str1[i - 1])
            i -= 1
        else:
            result_chars.append(str2[j - 1])
            j -= 1

    # Append remaining characters
    while i > 0:
        result_chars.append(str1[i - 1])
        i -= 1
    while j > 0:
        result_chars.append(str2[j - 1])
        j -= 1

    return "".join(reversed(result_chars))


# ==============================================================================
# SECTION 7: VARIATION 4: LONGEST PALINDROMIC SUBSEQUENCE (LPS)
# ==============================================================================
"""
Problem:
Given a single string `s`, find the length of the longest palindromic subsequence.

Insight: Reduction to LCS:
A palindrome reads identical forwards and backwards.
Therefore, the Longest Palindromic Subsequence of `s` is IDENTICAL to:
    LCS(s, reversed(s))!

Alternatively, LPS can be solved directly on intervals (i, j):
If s[i] == s[j]:
    lps(i, j) = 2 + lps(i + 1, j - 1)
Else:
    lps(i, j) = max(lps(i + 1, j), lps(i, j - 1))
"""

def lps_rec(s: str) -> int:
    """Pure Recursion on interval (i, j): O(2^n) time, O(n) space."""
    def helper(i: int, j: int) -> int:
        if i > j:
            return 0
        if i == j:
            return 1

        if s[i] == s[j]:
            return 2 + helper(i + 1, j - 1)

        return max(helper(i + 1, j), helper(i, j - 1))

    return helper(0, len(s) - 1)


def lps_memo(s: str) -> int:
    """Top-Down Memoization: O(n^2) time, O(n^2) space."""
    memo: Dict[Tuple[int, int], int] = {}

    def helper(i: int, j: int) -> int:
        if i > j:
            return 0
        if i == j:
            return 1

        state = (i, j)
        if state in memo:
            return memo[state]

        if s[i] == s[j]:
            res = 2 + helper(i + 1, j - 1)
        else:
            res = max(helper(i + 1, j), helper(i, j - 1))

        memo[state] = res
        return res

    return helper(0, len(s) - 1)


def lps_dp(s: str) -> int:
    """
    Bottom-Up Tabulation via LCS Reduction:
    LPS(s) = LCS(s, reverse(s)).
    Time:  O(n^2)
    Space: O(n^2)
    """
    rev_s = s[::-1]
    return lcs_tabulation(s, rev_s)


def lps_optimized(s: str) -> int:
    """
    Bottom-Up Rolling Array via LCS Reduction:
    Time:  O(n^2)
    Space: O(n)
    """
    rev_s = s[::-1]
    return lcs_space_optimized(s, rev_s)


# ==============================================================================
# SECTION 8: COMPARATIVE SUMMARY TABLE
# ==============================================================================
"""
+-----------------------------+-------------------------------+-------------------------------------------+
| Problem                     | Subproblem Dependence         | Recurrence Relation                       |
+-----------------------------+-------------------------------+-------------------------------------------+
| Longest Common Subsequence  | dp[i-1][j-1], dp[i-1][j],     | Match:    1 + dp[i-1][j-1]                |
|                             | dp[i][j-1]                    | Mismatch: max(dp[i-1][j], dp[i][j-1])     |
+-----------------------------+-------------------------------+-------------------------------------------+
| Longest Common Substring    | dp[i-1][j-1]                  | Match:    1 + dp[i-1][j-1]                |
|                             |                               | Mismatch: 0 (RESET contiguity)            |
+-----------------------------+-------------------------------+-------------------------------------------+
| Edit Distance (Levenshtein) | dp[i-1][j-1], dp[i-1][j],     | Match:    dp[i-1][j-1]                    |
|                             | dp[i][j-1]                    | Mismatch: 1 + min(Replace, Del, Insert)   |
+-----------------------------+-------------------------------+-------------------------------------------+
| Shortest Common Superseq    | dp[i-1][j-1], dp[i-1][j],     | Match:    1 + dp[i-1][j-1]                |
|                             | dp[i][j-1]                    | Mismatch: 1 + min(dp[i-1][j], dp[i][j-1]) |
+-----------------------------+-------------------------------+-------------------------------------------+
| Longest Palindromic Subseq  | dp[i+1][j-1], dp[i+1][j],     | Match:    2 + dp[i+1][j-1]                |
|                             | dp[i][j-1]                    | Mismatch: max(dp[i+1][j], dp[i][j-1])     |
|                             |                               | (Or: LCS(s, reverse(s)))                  |
+-----------------------------+-------------------------------+-------------------------------------------+
"""


# ==============================================================================
# SECTION 9: RUNNABLE TESTS & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("TESTING 2D SEQUENCE DP: RECURSION vs TOP-DOWN vs TABULATION vs O(N) SPACE")
    print("=" * 75)

    # 1. Longest Common Subsequence (LCS)
    s1, s2 = "abcde", "ace"
    lcs_r = lcs_recursive(s1, s2)
    lcs_m = lcs_memoization(s1, s2)
    lcs_t = lcs_tabulation(s1, s2)
    lcs_o = lcs_space_optimized(s1, s2)
    print(f"LCS('{s1}', '{s2}'):")
    print(f"  Rec: {lcs_r} | Memo: {lcs_m} | Tab: {lcs_t} | Opt: {lcs_o}")
    assert lcs_r == lcs_m == lcs_t == lcs_o == 3  # "ace" has length 3
    print("  [SUCCESS] All LCS variants returned 3.\n")

    # 2. Longest Common Substring
    sub1, sub2 = "zxabcdezy", "yzabcdezx"
    # Longest common continuous segment is "abcdez" (length 6)
    sub_r = lc_substring_rec(sub1, sub2)
    sub_m = lc_substring_memo(sub1, sub2)
    sub_t = lc_substring_dp(sub1, sub2)
    sub_o = lc_substring_optimized(sub1, sub2)
    print(f"Longest Common Substring('{sub1}', '{sub2}'):")
    print(f"  Rec: {sub_r} | Memo: {sub_m} | Tab: {sub_t} | Opt: {sub_o}")
    assert sub_r == sub_m == sub_t == sub_o == 6
    print("  [SUCCESS] All Longest Common Substring variants returned 6.\n")

    # 3. Edit Distance (Levenshtein)
    w1, w2 = "horse", "ros"
    # horse -> rorse (replace 'h' with 'r') -> rose (remove 'r') -> ros (remove 'e') = 3 operations
    ed_r = edit_distance_rec(w1, w2)
    ed_m = edit_distance_memo(w1, w2)
    ed_t = edit_distance_dp(w1, w2)
    ed_o = edit_distance_optimized(w1, w2)
    print(f"Edit Distance('{w1}', '{w2}'):")
    print(f"  Rec: {ed_r} | Memo: {ed_m} | Tab: {ed_t} | Opt: {ed_o}")
    assert ed_r == ed_m == ed_t == ed_o == 3
    print("  [SUCCESS] All Edit Distance variants returned 3.\n")

    # 4. Shortest Common Supersequence (SCS)
    scs1, scs2 = "abac", "cab"
    scs_len_r = scs_length_rec(scs1, scs2)
    scs_len_m = scs_length_memo(scs1, scs2)
    scs_len_t = scs_length_dp(scs1, scs2)
    scs_len_o = scs_length_optimized(scs1, scs2)
    reconstructed_scs = scs_reconstruct_string(scs1, scs2)
    print(f"Shortest Common Supersequence('{scs1}', '{scs2}'):")
    print(f"  Rec: {scs_len_r} | Memo: {scs_len_m} | Tab: {scs_len_t} | Opt: {scs_len_o}")
    print(f"  Reconstructed String: '{reconstructed_scs}'")
    # "cabac" has length 5; both "abac" and "cab" are subsequences of "cabac"
    assert scs_len_r == scs_len_m == scs_len_t == scs_len_o == 5
    assert len(reconstructed_scs) == 5
    print("  [SUCCESS] All Shortest Common Supersequence variants returned 5.\n")

    # 5. Longest Palindromic Subsequence (LPS)
    pal_str = "bbbab"
    # Longest palindromic subsequence is "bbbb" (length 4)
    lps_r = lps_rec(pal_str)
    lps_m = lps_memo(pal_str)
    lps_t = lps_dp(pal_str)
    lps_o = lps_optimized(pal_str)
    print(f"Longest Palindromic Subsequence('{pal_str}'):")
    print(f"  Rec: {lps_r} | Memo: {lps_m} | Tab: {lps_t} | Opt: {lps_o}")
    assert lps_r == lps_m == lps_t == lps_o == 4
    print("  [SUCCESS] All Longest Palindromic Subsequence variants returned 4.\n")

    print("=" * 75)
    print("[ALL TESTS PASSED] LCS & Two-Sequence Alignment DP module verified successfully!")
    print("=" * 75)
