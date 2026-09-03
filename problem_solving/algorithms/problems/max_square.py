"""
Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example 1:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4

Example 2:
Input: matrix = [["0","1"],["1","0"]]
Output: 1

Example 3:
Input: matrix = [["0"]]
Output: 0


"""

"""
Instead of starting at a cell and trying to grow a square downwards and rightwards (which requires checking a lot of overlapping boundaries), 
we look backwards from the bottom-right corner of a square.

The Dynamic Programming Intuition
Let dp[i][j] represent the side length of the largest square whose bottom-right corner is at matrix[i][j].

1. What determines if matrix[i][j] can form a square?
If matrix[i][j] == '0', then no square can end here. So, dp[i][j] = 0.
If matrix[i][j] == '1', it can potentially be part of a square. But how large can that square be?
2. The Recurrence Relation
To form a square of side length K ending at cell (i,j), you need:

A square of size at least K−1 directly above it: dp[i-1][j]
A square of size at least K−1 directly to its left: dp[i][j-1]
A square of size at least K−1 diagonally top-left: dp[i-1][j-1]
Since the actual square size is limited by the smallest adjacent square, we take the minimum of these three neighbors and add 1 (for the current cell itself):

dp[i][j]=min(dp[i−1][j],dp[i][j−1],dp[i−1][j−1])+1

Visual Example
Consider this subgrid where all adjacent cells are part of 2×2 squares:

    (top-left)    (top)
       [2]        [2]

    (left)     (current)
       [2]        ['1']  --> dp[i][j] = min(2, 2, 2) + 1 = 3
Because all 3 neighbors can support a 2×2 square, adding the current '1' forms a 3×3 square! If even one of those neighbors were a 0 or 1, 
the smallest one would bottleneck the result.

"""

class Solution:

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])

        # dp[i][j] represents length of maximal square ending at matrix[i-1][j-1]
        # Padded with extra row and column of 0s to simplify border conditions
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        max_side = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Note matrix is 0-indexed, so we check matrix[i-1][j-1]
                if matrix[i - 1][j - 1] == "1":
                    dp[i][j] = (
                        min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    )
                    max_side = max(max_side, dp[i][j])

        # Return the area (side * side)
        return max_side * max_side
