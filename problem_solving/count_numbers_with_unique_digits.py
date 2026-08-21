"""
Given an integer n, return the count of all numbers with unique digits, x, where 0 <= x < 10n.

 

Example 1:

Input: n = 2
Output: 91
Explanation: The answer should be the total numbers in the range of 0 ≤ x < 100, excluding 11,22,33,44,55,66,77,88,99
Example 2:

Input: n = 0
Output: 1
 

Constraints:

0 <= n <= 8

"""

# brute force
# very slow

class Solution:
    def testUniqueness(self, x: int) -> bool:
        str_x = str(x)
        unique_char_x = set(str_x)
        if len(str_x) == len(unique_char_x):
            return True
        else:
            return False

    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        
        max_val = 10**n
        unique_digits_count = 0

        for num in range(max_val):
            if self.testUniqueness(num):
                unique_digits_count += 1

        return unique_digits_count
# using combinatorics

class Solution:
    # n = 0 - one unique digits - return 1
    # n = 1 - 1,...,9 - 9 unique numbers
    # n = 2 - for each digit in n = 1, can add 9 unique digits (0,1,...,9) to end to create unique number - 9*9 unique numbers
    # n = 3 - for each digit in n = 2, can add 8 unique digits to end to create unique number - 9*9*8
    # and so forth


    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        if n == 1:
            return 10

        currentCount = 10
        currentNumProduct = 9
        currentNextNum = 9

        for i in range(2,n+1):

            currentCount += currentNumProduct * currentNextNum

            currentNumProduct = currentNumProduct*currentNextNum

            currentNextNum -= 1

        return currentCount