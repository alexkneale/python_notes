
"""
Given an integer array nums, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

Note that the product of an array with a single element is the value of that element.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
 

Constraints:

1 <= nums.length <= 2 * 104
-10 <= nums[i] <= 10
The product of any subarray of nums is guaranteed to fit in a 32-bit integer.

"""
from typing import List

import math
class Solution:

    def getNegativeIndices(self, nums: List[int]) -> List[int]:
        return [i for i,val in enumerate(nums) if val < 0]
        

    def maxProduct(self, nums: List[int]) -> int:
        # split nums in multiple lists, split at 0
        listsWithoutZeros = []
        startIndex = 0
        isZero = False
        for i,num in enumerate(nums):
            if num == 0:
                isZero = True
                listsWithoutZeros.append(nums[startIndex:i])
                startIndex = i + 1
        if not isZero:
            listsWithoutZeros = [nums]
        
        maxProd = 0
        print(f"Lists without Zeros: {listsWithoutZeros}")
        
        for listNums in listsWithoutZeros:
            if len(listNums) == 1:
                if listNums[0] > maxProd:
                    maxProd = listNums[0]
                else:
                    continue
            negIndices = self.getNegativeIndices(listNums)
            if len(negIndices)%2 == 0:
                fullProd = math.prod(listNums)
                if fullProd > maxProd:
                    maxProd = fullProd
            else:
                fullProd1 = math.prod(listNums[negIndices[0]:])
                fullProd2 = math.prod(listNums[:negIndices[-1]])
                if max(fullProd1, fullProd2) > maxProd:
                    maxProd = max(fullProd1, fullProd2)
        
        return maxProd

if __name__ == "__main__":
    maxProd = Solution()
    print(maxProd.maxProduct(nums=[-2,0,-1]))            