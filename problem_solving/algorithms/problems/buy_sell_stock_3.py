"""
You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

Example 1:

Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. 
You must sell before buying again.
Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.

"""

# one solution - very slow - O(n^2)



import math
class Solution:

    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        current_sum = nums[0]

        for num in nums[1:]:
            # Either extend the existing subarray or start a new one at `num`
            current_sum = max(num, current_sum + num)
            # Track the largest sum encountered so far
            max_sum = max(max_sum, current_sum)

        return max_sum
    
    def maxProfit(self, prices: List[int]) -> int:
        priceDiffs = []

        for i in range(1,len(prices)):
            priceDiffs.append(prices[i]-prices[i-1])
        if len(priceDiffs) == 1:
            if priceDiffs[0] >= 0:
                return priceDiffs[0]
            else:
                return 0
        maxSum = 0
        for j in range(1,len(priceDiffs)):
            firstPartSum = self.maxSubArray(priceDiffs[:j])
            secondPartSum = self.maxSubArray(priceDiffs[j:])
            if firstPartSum + secondPartSum > maxSum and firstPartSum >=0 and secondPartSum >=0:
                maxSum = firstPartSum + secondPartSum
            # for negative sum cases
            elif secondPartSum < 0 and firstPartSum > maxSum:
                maxSum = firstPartSum
            elif firstPartSum < 0 and secondPartSum > maxSum:
                maxSum = secondPartSum

        return maxSum

# better solution - O(n)

import math
class Solution:

    
    
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        
        n = len(prices)
        # max profit up to day i
        leftProfit = [0] * n
        # max profit from day i onwards
        rightProfit = [0] * n

        minPrice = prices[0]

        for i in range(1,n):
            minPrice = min(minPrice, prices[i])
            leftProfit[i] = max(leftProfit[i-1], prices[i] - minPrice)
        
        # loop from back to start - as want to find max profit from day i onwards (to end of the prices array)
        maxPrice = prices[-1]
        # start at n-2 as, we fill out maxPrice with last element (at index n-1)
        for j in range(n-2, -1, -1):
            maxPrice = max(maxPrice, prices[j])
            rightProfit[j] = max(rightProfit[j+1], maxPrice - prices[j])
        
        maxProfit = 0
        for i in range(n):
            maxProfit = max(maxProfit, rightProfit[i]+leftProfit[i])

        return maxProfit