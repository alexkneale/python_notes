"""
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, 
return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
 

Constraints:

1 <= coins.length <= 12
1 <= coins[i] <= 231 - 1
0 <= amount <= 104

"""

from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[i] will store the minimum number of coins needed to make up amount i.
        # We initialize the array with amount + 1, which represents "infinity" in this context
        # (since the maximum coins needed for any amount i is i, using 1-unit coins).
        dp = [amount + 1] * (amount + 1)
        
        # Base case: 0 coins are needed to make up amount 0.
        dp[0] = 0
        
        # Build the dp table from 1 up to amount
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        # If dp[amount] is still amount + 1, it means the amount cannot be formed.
        return dp[amount] if dp[amount] <= amount else -1

if __name__ == "__main__":
    sol = Solution()
    
    # Test case 1
    coins1, amount1 = [1, 2, 5], 11
    print(f"Coins: {coins1}, Amount: {amount1} -> Min coins: {sol.coinChange(coins1, amount1)} (Expected: 3)")
    
    # Test case 2
    coins2, amount2 = [2], 3
    print(f"Coins: {coins2}, Amount: {amount2} -> Min coins: {sol.coinChange(coins2, amount2)} (Expected: -1)")
    
    # Test case 3
    coins3, amount3 = [1], 0
    print(f"Coins: {coins3}, Amount: {amount3} -> Min coins: {sol.coinChange(coins3, amount3)} (Expected: 0)")
    
    # Test case 4 (Greedy fails)
    coins4, amount4 = [1, 3, 4], 6
    print(f"Coins: {coins4}, Amount: {amount4} -> Min coins: {sol.coinChange(coins4, amount4)} (Expected: 2, since 3+3=6 is better than 4+1+1=6)")