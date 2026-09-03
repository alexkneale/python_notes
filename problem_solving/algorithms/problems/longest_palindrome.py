"""
Given a string s, return the longest palindromic substring in s.

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
 

Constraints:

1 <= s.length <= 1000
s consist of only digits and English letters.
"""

# brute force : time complexity = force O(n^3), space complexity O(1)

class Solution:

    def isPalindrome(self, s: str) -> bool:
        if s == s[::-1]:
            return True
        else:
            return False

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n == 1:
            return s
        maxSize = 0
        maxPalindrome = ''
        # 0,1
        for i in range(n):
            # 1,2
            for j in range(i+1,n+1):
                if self.isPalindrome(s[i:j]) and (j-i+1) > maxSize:
                    maxSize = j-i
                    maxPalindrome = s[i:j]
        return maxPalindrome

"""
Approach 2: Expand Around Center

Intuition :
To enumerate all palindromic substrings of a given string, we first expand a given string at each possible starting position of a palindrome 
and also at each possible ending position of a palindrome and keep track of the length of the longest palindrome we found so far.

Approach :
We observe that a palindrome mirrors around its center. Therefore, a palindrome can be expanded from its center, and there are only 2n - 1 such centers.
You might be asking why there are 2n - 1 but not n centers? The reason is the center of a palindrome can be in between two letters. Such palindromes 
have even number of letters (such as "abba") and its center are between the two 'b's.'
Since expanding a palindrome around its center could take O(n) time, the overall complexity is O(n^2).

Algorithm :
At starting we have maz_str = s[0] and max_len = 1 as every single character is a palindrome.
Now, we will iterate over the string and for every character we will expand around its center.
For odd length palindrome, we will consider the current character as the center and expand around it.
For even length palindrome, we will consider the current character and the next character as the center and expand around it.
We will keep track of the maximum length and the maximum substring.
Print the maximum substring.

Complexity Analysis
- Time complexity : O(n^2). Since expanding a palindrome around its center could take O(n) time, the overall complexity is O(n^2).
- Space complexity : O(1).


"""

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s

        def expand_from_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        max_str = s[0]

        for i in range(len(s) - 1):
            odd = expand_from_center(i, i)
            even = expand_from_center(i, i + 1)

            if len(odd) > len(max_str):
                max_str = odd
            if len(even) > len(max_str):
                max_str = even

        return max_str

"""
Approach 3: Dynamic Programming

To improve over the brute force solution, we first observe how we can avoid unnecessary re-computation while validating palindromes. 
Consider the case "ababa". If we already knew that "bab" is a palindrome, it is obvious that "ababa" must be a palindrome since the two 
left and right end letters are the same.

Algorithm :
We initialize a boolean table dp and mark all the values as false.
We will use a variable max_len to keep track of the maximum length of the palindrome.
We will iterate over the string and mark the diagonal elements as true as every single character is a palindrome.
Now, we will iterate over the string and for every character we will expand around its center.
For odd length palindrome, we will consider the current character as the center and expand around it.
For even length palindrome, we will consider the current character and the next character as the center and expand around it.
We will keep track of the maximum length and the maximum substring.
Print the maximum substring.

Complexity Analysis
- Time complexity : O(n^2). This gives us a runtime complexity of O(n^2).
- Space complexity : O(n^2). It uses O(n^2) space to store the table.
"""

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s
        
        Max_Len=1
        Max_Str=s[0]
        dp = [[False for _ in range(len(s))] for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = True
            for j in range(i):
                if s[j] == s[i] and (i-j <= 2 or dp[j+1][i-1]):
                    dp[j][i] = True
                    if i-j+1 > Max_Len:
                        Max_Len = i-j+1
                        Max_Str = s[j:i+1]
        return Max_Str

"""
Approach 4: Manacher's Algorithm

Intuition :
To avoid the unnecessary validation of palindromes, we can use Manacher's algorithm. The algorithm is explained brilliantly in this article. 
The idea is to use palindrome property to avoid unnecessary validations. We maintain a center and right boundary of a palindrome. 
We use previously calculated values to determine if we can expand around the center or not. If we can expand, we expand and update 
the center and right boundary. Otherwise, we move to the next character and repeat the process. We also maintain a variable max_len 
to keep track of the maximum length of the palindrome. We also maintain a variable max_str to keep track of the maximum substring.

Algorithm :
We initialize a boolean table dp and mark all the values as false.
We will use a variable max_len to keep track of the maximum length of the palindrome.
We will iterate over the string and mark the diagonal elements as true as every single character is a palindrome.
Now, we will iterate over the string and for every character we will expand around its center.
For odd length palindrome, we will consider the current character as the center and expand around it.
For even length palindrome, we will consider the current character and the next character as the center and expand around it.
We will keep track of the maximum length and the maximum substring.
Print the maximum substring.

Complexity Analysis :
- Time complexity : O(n). Since expanding a palindrome around its center could take O(n) time, the overall complexity is O(n).
- Space complexity : O(n). It uses O(n) space to store the table.

"""

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s
        
        Max_Len=1
        Max_Str=s[0]
        s = '#' + '#'.join(s) + '#'
        dp = [0 for _ in range(len(s))]
        center = 0
        right = 0
        for i in range(len(s)):
            if i < right:
                dp[i] = min(right-i, dp[2*center-i])
            while i-dp[i]-1 >= 0 and i+dp[i]+1 < len(s) and s[i-dp[i]-1] == s[i+dp[i]+1]:
                dp[i] += 1
            if i+dp[i] > right:
                center = i
                right = i+dp[i]
            if dp[i] > Max_Len:
                Max_Len = dp[i]
                Max_Str = s[i-dp[i]:i+dp[i]+1].replace('#','')
        return Max_Str