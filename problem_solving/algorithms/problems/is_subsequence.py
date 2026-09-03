"""
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters
 without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

 

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true
Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false
 

Constraints:

0 <= s.length <= 100
0 <= t.length <= 104
s and t consist only of lowercase English letters.

"""

class Solution:

    def getAllIndicesOfCharInString(self, s: str, char: str) -> List[int]:
        return [i for i, val in enumerate(s) if val == char]

    # brute force
    def isSubsequence(self, s: str, t: str) -> bool:

        # edge case
        if len(s) == 0 and len(t) >= 0:
            return True
        

        # final conditions
        charToCheck = s[0]
        if len(s) == 1:
            if charToCheck in t:
                return True
            else:
                return False
        
        # looping conditions
        if charToCheck in t:
            charIndices = self.getAllIndicesOfCharInString(t, charToCheck)
            for index in charIndices:
                return self.isSubsequence(s[1:], t[index+1:]) 

        else:
            return False

# faster solution

    def isSubsequence(self, s: str, t: str) -> bool:

        if not s:
            return True

        i = 0

        for char in t:
            if char == s[i]:
                i += 1
                if i == len(s):
                    return True
        return i == len(s)

"""
"Suppose there are lots of incoming s, say s 

s1,s2 ,…,sk
where k≥1,000,000,000, and you want to check one by one to see if t has its subsequence. In this scenario, how would you change your code?"

If t is fixed and very large while s queries are numerous:

Preprocess t by creating a hash map storing the indices for each character:
{'a': [0, 4, 7], 'b': [1, 5], ...}
For each query s, use Binary Search (bisect_right) to quickly find the smallest index in t that comes after the previously matched index.
This reduces the time per query s from O(∣t∣) down to O(∣s∣log∣t∣).


"""

        