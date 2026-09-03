"""
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

 

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
Example 2:

Input: n = 1
Output: ["()"]
 

Constraints:

1 <= n <= 8


"""

# backtracking
# recursion
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current_str: str, open_count: int, close_count: int):
            # Base case: string reaches maximum length (2 * n)
            if len(current_str) == 2 * n:
                result.append(current_str)
                return
            
            # Add '(' if we haven't used all n open brackets
            if open_count < n:
                backtrack(current_str + "(", open_count + 1, close_count)
            
            # Add ')' if it will balance an existing open bracket
            if close_count < open_count:
                backtrack(current_str + ")", open_count, close_count + 1)

        backtrack("", 0, 0)
        return result

# If you specifically want to use Dynamic Programming, valid parenthesis combinations for n follow the Catalan number structure:

# dp[n]={"("+a+")"+b∣a∈dp[i],b∈dp[n−1−i] for 0≤i<n}

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        dp = [[] for _ in range(n + 1)]
        dp[0] = [""]
        
        for i in range(1, n + 1):
            for left_count in range(i):
                right_count = i - 1 - left_count
                for left in dp[left_count]:
                    for right in dp[right_count]:
                        dp[i].append(f"({left}){right}")
                        
        return dp[n]
