"""
Given a string s, find the longest palindromic subsequence's length in s.

A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

 

Example 1:

Input: s = "bbbab"
Output: 4
Explanation: One possible longest palindromic subsequence is "bbbb".
Example 2:

Input: s = "cbbd"
Output: 2
Explanation: One possible longest palindromic subsequence is "bb".
 

Constraints:

1 <= s.length <= 1000
s consists only of lowercase English letters.

"""

class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:

        if len(s) <= 1:
            return len(s)

        def expand_from_center(subsequence: str, left: int, right: int):
            max_subsequence = 1
            while left >= 0 and right < len(subsequence):
                if subsequence[left] == subsequence[right]:
                    left -= 1
                    right += 1
                    max_subsequence = right - left
                else:
                    lenExpandingRight = 0
                    charToFindLeft = subsequence[left]
                    if charToFindLeft in subsequence[right]:
                        max_subsequence_right = expand_from_center(subsequence=)

            return len(max_subsequence)
        
        max_len = 1


if __name__ == "__main__":
    sol = Solution()
    
    # Test case 1
    s1 = "bbbab"
    print(f"string: {s1} -> Max len subsequence: {sol.longestPalindromeSubseq(s1)} (Expected: 4)")