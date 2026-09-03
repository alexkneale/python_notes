"""
Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.

Return the maximum product you can get.

 

Example 1:

Input: n = 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.
Example 2:

Input: n = 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
 

Constraints:

2 <= n <= 58


"""


class Solution:
    def integerBreak(self, n: int) -> int:
        # contains max product of k positive integers used to break i up (index i indicates integer i+1)
        # starting at n = 2 (as n = 1 is not valid)

        if n == 1:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        
        dp = [1] * (n-1)
        # n = 3
        dp[1] = 2
        # start at 4
        for i in range(2,n-1):
            currentNum = i+2
            # case of currentNum * 1
            baseCase = currentNum
            print(f"Base Case: {baseCase}")

            for j in range(0,i):
                
                
                diff = i - j
                secondBaseCase = (j+2)*diff
                print(f"Second Base Case {secondBaseCase}")
                print(f"Difference: {diff}")
                print(f"Calculation: {dp[j]*diff}")
                dp[i] = max(dp[i], baseCase,secondBaseCase, dp[j]*diff)
        print(dp)
        return dp[-1]

if __name__ == "__main__":
    sol = Solution()
    sol.integerBreak(n=10)
        
