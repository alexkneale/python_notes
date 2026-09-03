"""
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
"""

# solution - very memory/time intensive due to storing repeated lists, and calling index() and pop()

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxVal = max(prices)
        minVal = min(prices)
        minValIndex = prices.index(minVal)
        maxValIndex = prices.index(maxVal)
        if len(prices) == 1:
            return 0
        if minValIndex < maxValIndex:
            return maxVal-minVal
        elif len(prices) == 2:
            return 0
        else:
            minValRemovedList = prices.copy()
            maxValRemovedList = prices.copy()

            minValRemovedList.remove(minVal)
            maxValRemovedList.remove(maxVal)

            return max(self.maxProfit(minValRemovedList), self.maxProfit(maxValRemovedList))

# alternative - not great but better

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
        if len(prices) == 1:
            return 0
        diffPrices = [a_i - b_i for a_i, b_i in zip(prices[1:], prices[:-1])]
        
        maxSubSum = self.maxSubArray(diffPrices)
        maxSubSum = 0 if maxSubSum < 0 else maxSubSum

        return maxSubSum

# best solution - O(n)
# notice how max profit at any time i is just (price at i) - (lowest price < i)

import math

def maxProfit(self, prices: List[int]) -> int:
        minPrice = math.inf
        maxProfit = 0

        for price in prices:
            if price < minPrice:
                minPrice = price
            elif price - minPrice > maxProfit:
                maxProfit = price - minPrice
        return maxProfit