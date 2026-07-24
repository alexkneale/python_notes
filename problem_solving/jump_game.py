"""
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your 
maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

 

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
 

Constraints:

1 <= nums.length <= 104
0 <= nums[i] <= 105

"""

class Solution:
    def getZeroIndices(self, nums : List[int]) -> List[int]:
        return [i for i,val in enumerate(nums) if val==0]

    def isZeroPassable(self, subNums : List[int]) -> bool:
        maxIndex = len(subNums) - 1
        for i in range(maxIndex,-1,-1):
            if i + subNums[i] > maxIndex:
                return True
        return False
    def isZeroReachable(self, subNums: List[int]) -> bool:
        maxIndex = len(subNums) - 1
        for i in range(maxIndex,-1,-1):
            if i + subNums[i] >= maxIndex:
                return True
        return False


    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        if 0 not in nums:
            return True
        else:
            zeroIndices = self.getZeroIndices(nums)
            zeroCount = len(zeroIndices)
            maxIndex = len(nums) - 1

            for i in range(zeroCount):
                zeroIndex = zeroIndices[i]
                if zeroIndex == maxIndex:
                    subNums = nums[:zeroIndex+1]
                    if self.isZeroReachable(subNums):
                        return True
                    else:
                        return False
                else:
                    subNums = nums[:zeroIndex+1]
                    if self.isZeroPassable(subNums):
                        continue
                    else:
                        return False
            return True
            # for each 0 in array, check if it's passable
            # it's passable if at least one element behind it has value > distance between it and the 0
