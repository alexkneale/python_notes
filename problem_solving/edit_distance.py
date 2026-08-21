"""
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character
 

Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
 

Constraints:

0 <= word1.length, word2.length <= 500
word1 and word2 consist of lowercase English letters.

"""


# recursive method - using Levenshtein distance

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        if len(word1) == 0:
            return len(word2)
        elif len(word2) == 0:
            return len(word1)
        elif word1[0] == word2[0]:
            return self.minDistance(word1[1:], word2[1:])
        else:
            return 1+min(self.minDistance(word1[1:],word2), self.minDistance(word1,word2[1:]), self.minDistance(word1[1:],word2[1:]))


# but this is very slow

# faster solution - just using indices, and using cacheing

from functools import cache
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def dp(i: int, j: int) -> int:
            if i == len(word1):
                return len(word2) - j
            if j == len(word2):
                return len(word1) - i

            # initial matching case - just move forward
            if word1[i] == word2[j]:
                return dp(i+1,j+1)
            
            delete = dp(i+1,j)
            insert = dp(i,j+1)
            replace = dp(i+1, j+1)

            return 1 + min(delete, insert, replace)

        return dp(0,0)

