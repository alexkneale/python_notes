"""
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row,
you may move to either index i or index i + 1 on the next row.

 

Example 1:

Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
Example 2:

Input: triangle = [[-10]]
Output: -10
 

Constraints:

1 <= triangle.length <= 200
triangle[0].length == 1
triangle[i].length == triangle[i - 1].length + 1
-104 <= triangle[i][j] <= 104



"""

from typing import List
def minimumTotal(triangle: List[List[int]]) -> int:
    if len(triangle) == 1:
        return triangle[0][0]
    
    # dp contains min number of steps to get to each index
    dp = [0 for _ in range(len(triangle))]
    print(dp)
    dp[0] = triangle[0][0] + triangle[1][0]
    dp[1] = triangle[0][0] + triangle[1][1]

    for i, row in enumerate(triangle[2:]):
        rowIndex = i+2
        lenRow = len(row)
        newRow = [0 for _ in range(len(triangle))]
        newRow[0] = dp[0] + row[0]
        newRow[lenRow-1] = dp[lenRow-2] + row[lenRow-1]
        
        for j in range(1,lenRow-1):
            minParent = min(dp[j], dp[j-1])
            newRow[j] = minParent + row[j]
        dp = newRow
        print(f"Sum row at iteration {rowIndex}: {dp}")

    print(dp)
    
    return min(dp)


if __name__ == "__main__":
    example = [[2],[3,4],[6,5,7],[4,1,8,3]]
    minimumTotal(example)