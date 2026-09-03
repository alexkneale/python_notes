"""
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
Example 2:

Input: grid = [[1,2,3],[4,5,6]]
Output: 12
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 200
0 <= grid[i][j] <= 200


"""

import numpy as np

class Solution:

    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # edge cases - one column/row
        if m == 1:
            return sum(grid[0])
        elif n == 1:
            flatList = [val[0] for val in grid]
            return sum(flatList)
        # regular case
        else:
            npGrid = np.zeros((m,n))
            npGrid[0,0] = grid[0][0]
            # calc min sum vals for top row/left col
            for i in range(1,m):
                npGrid[i,0] += npGrid[i-1,0]
                npGrid[i,0] += grid[i][0]
            for i in range(1,n):
                npGrid[0,i] += npGrid[0,i-1]
                npGrid[0,i] += grid[0][i]

            # min path logic
            for rowIndex in range(1,m):
                for colIndex in range(1,n):
                    minValueToAdd = min(npGrid[rowIndex,colIndex-1],npGrid[rowIndex-1,colIndex])
                    npGrid[rowIndex,colIndex] = grid[rowIndex][colIndex] + minValueToAdd
            return int(npGrid[m-1,n-1])

            




        