"""

"""
from typing import List, Any

class Solution:
    def isPalindrome(self, s: str) -> bool:
        if s[::-1] == s:
            return True
        else:
            return False

    # def findNextPalindrome(self, s: str, startIndex = 0) -> int:
    #     # start from end and 
    #     for i in range(len(s)-1,startIndex,-1):
    #         if self.isPalindrome(s[startIndex:i+1]):
    #             return i
    #     return startIndex


    # def partition(self, s: str) -> List[List[str]]:
    #      # idea - find all palindromes in s first
    #     # THEN - find all combinations of split str with these palindromes

    #     startIndexPalindrome=0
    #     endIndexPalindrome=0
    #     allPalindromes = []
    #     # get 
    #     while startIndexPalindrome < len(s)-1:
    #         newPotentialEndIndex = self.findNextPalindrome(s, startIndexPalindrome)
    #         if newPotentialEndIndex == startIndexPalindrome:
    #             startIndexPalindrome +=1
    #         else:
    #             allPalindromes.append((startIndexPalindrome,newPotentialEndIndex))
    #             startIndexPalindrome=newPotentialEndIndex+1

    #     allValidLists = []
    #     base_list = list(s)
    #     allValidLists.append(base_list)
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)

        for startIndex in range(n):
            for endIndex in range(1,n):
                if 



        


       

if __name__ == "__main__":
    solution = Solution()
    allPalinfromes = solution.partition("abcaabcddcefaa")
    print(allPalinfromes)