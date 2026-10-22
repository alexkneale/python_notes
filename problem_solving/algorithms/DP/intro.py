"""
================================================================================
DYNAMIC PROGRAMMING: FOUNDATIONS & PROBLEM IDENTIFICATION
Module 1: The Intro & The Intuition Behind DP
================================================================================

TABLE OF CONTENTS:
1. WHAT IS DYNAMIC PROGRAMMING? (THE SIMPLE DEFINITION)
2. THE TWO PILLARS: WHEN DOES A PROBLEM QUALIFY FOR DP?
   - Pillar 1: Optimal Substructure
   - Pillar 2: Overlapping Subproblems
3. HOW TO IDENTIFY DP VS. NORMAL RECURSION
   - The Element of Choice (Branching)
   - The Optimization Keyword (Min, Max, Best, Count)
4. SINGLE-BRANCH VS. MULTI-BRANCH RECURSION (THE ROOT CAUSE)
   - Example 1: Factorial (Single Call -> No Overlap -> No DP Needed)
   - Example 2: Paths / Choices (Multiple Calls -> Massive Overlap -> DP Candidate)
5. CODE WALKTHROUGH: VISUALIZING THE OVERLAPPING DISASTER
   - Instrumented recursion showing duplicate subproblem calculations
6. THE GOLDEN RULE OF DP: THE 3-STEP RECURSIVE BLUEPRINT
7. THE 10 "PARENT PROBLEMS" OF DYNAMIC PROGRAMMING (ROADMAP)
8. RUNNABLE DEMONSTRATION & CALL-COUNT PROOFS
================================================================================
"""

from typing import List


# ==============================================================================
# SECTION 1: WHAT IS DYNAMIC PROGRAMMING?
# ==============================================================================
"""
A simple way to think of Dynamic Programming:
    DP = Recursion + Remembering What You Already Did

Core Mental Model:
1. DP is NOT some magical, completely distinct math formula. DP is just
   *carefully structured recursion* where you avoid doing the same work twice.
2. If you cannot solve the problem with recursion first, you CANNOT solve it with DP.
3. NEVER start by drawing a 2D table or grid. Beginners often make the mistake of
   trying to guess the table dimensions (`dp[i][j]`) before understanding the
   underlying recurrence. 
4. Always discover the recursive relationship first. Dynamic Programming is just
   an optimization layer added on top of a working recursive tree.
"""


# ==============================================================================
# SECTION 2: THE TWO PILLARS OF DYNAMIC PROGRAMMING
# ==============================================================================
"""
For any problem to be solvable by Dynamic Programming, it MUST satisfy TWO
essential mathematical properties:

--------------------------------------------------------------------------------
PILLAR 1: OPTIMAL SUBSTRUCTURE
--------------------------------------------------------------------------------
- Meaning: An optimal solution to the overall problem can be constructed directly
  from optimal solutions to its smaller subproblems.
- Analogy:
    If the shortest route from London to Edinburgh passes through York, then:
    (Shortest route London -> Edinburgh) = 
        (Shortest route London -> York) + (Shortest route York -> Edinburgh).
    If this wasn't true, you couldn't break the trip down into sub-legs.

--------------------------------------------------------------------------------
PILLAR 2: OVERLAPPING SUBPROBLEMS
--------------------------------------------------------------------------------
- Meaning: When breaking down the main problem recursively, the SAME subproblems
  keep getting evaluated again and again with the exact same inputs.
- Non-overlapping (Divide & Conquer):
    In Merge Sort, you divide array [1, 2, 3, 4] into [1, 2] and [3, 4].
    Notice [1, 2] and [3, 4] are completely disjoint (unique). They never repeat.
    Thus, Merge Sort is Divide & Conquer, NOT DP.
- Overlapping (Dynamic Programming):
    In calculating WaysToClimb(5 stairs), you calculate WaysToClimb(3) when
    taking 2 steps from step 1, AND you calculate WaysToClimb(3) again when
    taking 1 step from step 2.
    The exact same question "How many ways to climb 3 stairs?" is asked repeatedly.
"""


# ==============================================================================
# SECTION 3: HOW TO IDENTIFY DP VS. NORMAL RECURSION
# ==============================================================================
"""
Look for these two tell-tale signs in any interview question description:

1. THE ELEMENT OF CHOICE:
   - At each stage, the algorithm has to make a choice between two or more options:
     * "Do I take this item, or do I leave it?" (0/1 Knapsack)
     * "Do I step 1 stair, or do I step 2 stairs?" (Climbing Stairs)
     * "Do I match these two characters, or skip one?" (Longest Common Subsequence)
     * "Do I move down, or do I move right?" (Grid Path)
   - Because you don't know in advance which choice is best, your recursive
     function must branch out to explore multiple choices.

2. ASKING FOR SOMETHING OPTIMAL OR COMBINATORIAL:
   - The question asks for:
     * Minimum (e.g., minimum coins to make change)
     * Maximum (e.g., maximum profit, largest value knapsack)
     * Total number of ways (e.g., total unique paths to the end)
     * Possibility (e.g., can we partition this array into two equal subsets? Yes/No)

If a problem has CHOICES + asks for OPTIMAL / TOTAL WAYS:
-> 95% chance it is a Dynamic Programming problem!
"""


# ==============================================================================
# SECTION 4: SINGLE-BRANCH VS. MULTI-BRANCH RECURSION
# ==============================================================================
"""
Why does recursion sometimes need DP, and other times not?
It all comes down to the number of recursive calls made per frame.
"""

# ------------------------------------------------------------------------------
# CASE A: Single-Branch Recursion (NO DP NEEDED)
# ------------------------------------------------------------------------------
def factorial(n: int) -> int:
    """
    Recurrence: fact(n) = n * fact(n - 1)
    
    Notice: There is only ONE recursive call inside this function.
    The call stack looks like a straight line:
        fact(5) -> fact(4) -> fact(3) -> fact(2) -> fact(1)
        
    Are there any choices? NO. (Deterministic step: just subtract 1).
    Are there overlapping subproblems? NO. (fact(3) is only called once in total).
    Conclusion: You NEVER need DP for single-branch recursion!
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


# ------------------------------------------------------------------------------
# CASE B: Multi-Branch Recursion (OVERLAPPING SUBPROBLEMS EMERGE -> DP APPLIES)
# ------------------------------------------------------------------------------
def count_ways_to_climb(stairs_left: int) -> int:
    """
    Problem: You are climbing a staircase. It takes `stairs_left` steps to reach
    the top. Each time, you have a CHOICE: climb 1 step or 2 steps.
    How many distinct ways can you climb to the top?

    Recurrence: ways(n) = ways(n - 1) + ways(n - 2)

    Notice: There are TWO recursive calls here.
    The call tree splits into two branches at EVERY single node:

                         climb(4)
                       /          \\
               climb(3)            climb(2)   <-- climb(2) evaluated here!
              /        \\          /       \\
          climb(2)     climb(1)  climb(1) climb(0)
          /      \\
      climb(1) climb(0)

    LOOK AT THE REDUNDANCY:
    - climb(2) is calculated 2 separate times.
    - climb(1) is calculated 3 separate times.
    - climb(0) is calculated 2 separate times.
    
    As `stairs_left` grows to 40, climb(2) will be recalculated millions of times.
    This creates an exponential O(2^n) explosion.
    """
    if stairs_left == 0:
        return 1  # Reached the exact top: 1 valid way
    if stairs_left < 0:
        return 0  # Overshot: 0 valid ways

    # The choice: take 1 step OR take 2 steps
    ways_taking_one_step = count_ways_to_climb(stairs_left - 1)
    ways_taking_two_steps = count_ways_to_climb(stairs_left - 2)

    return ways_taking_one_step + ways_taking_two_steps


# ==============================================================================
# SECTION 5: CODE WALKTHROUGH: VISUALIZING THE OVERLAPPING DISASTER
# ==============================================================================

# Global counters to empirically demonstrate how bad plain recursion is
call_log: List[int] = []

def instrumented_climb(stairs_left: int) -> int:
    """
    Identical to `count_ways_to_climb`, but records every time a subproblem
    is evaluated to show the overlapping subproblems in real time.
    """
    global call_log
    call_log.append(stairs_left)

    if stairs_left == 0:
        return 1
    if stairs_left < 0:
        return 0

    return instrumented_climb(stairs_left - 1) + instrumented_climb(stairs_left - 2)


# ==============================================================================
# SECTION 6: THE GOLDEN RULE: HOW TO APPROACH ANY DP PROBLEM FROM SCRATCH
# ==============================================================================
"""
Whenever you encounter an interview problem that feels like DP, follow these
three steps in order. NEVER skip straight to the final code.

STEP 1: DEFINE THE STATE (What variables describe the subproblem?)
   - Look at what changes as you make decisions:
     * Knapsack: (current item index `i`, remaining bag capacity `w`)
     * Climbing Stairs: (stairs remaining `n`)
     * LCS: (index `i` in string1, index `j` in string2)

STEP 2: EXPRESS THE CHOICES & BASE CASES (Draw the Recursion Tree)
   - What are the decisions available at this state?
   - What are the base cases? (e.g., ran out of items, capacity is 0, reached end).
   - Write the pure recursive function:
       def solve(state):
           if base_case: return ...
           ans = explore_choices(...)
           return ans

STEP 3: IDENTIFY THE OVERLAPPING SUBPROBLEMS
   - Look at the arguments passed in step 2.
   - Do different paths in the recursion tree pass the exact same arguments?
   - If YES, you have successfully confirmed that the problem requires DP!
   (At this point, you are ready to add storage/caching to save those answers).
"""


# ==============================================================================
# SECTION 7: THE 10 "PARENT PROBLEMS" OF DYNAMIC PROGRAMMING
# ==============================================================================
"""
Almost every dynamic programming question asked in technical interviews is a
disguised variation of one of these 10 classic parent patterns. Mastering the
base recursive tree of these 10 patterns gives you the blueprint for all DP:

1. 0/1 KNAPSACK (Finite choices)
   - Core concept: You have items with weights and values. For each item, you
     either include it (1) or exclude it (0). Cannot reuse items.
   - Variations: Subset Sum, Equal Sum Partition, Minimum Subset Sum Difference,
     Target Sum.

2. UNBOUNDED KNAPSACK (Infinite choices)
   - Core concept: Similar to 0/1 Knapsack, but each item can be selected
     an unlimited number of times.
   - Variations: Coin Change (Min coins), Coin Change 2 (Number of ways),
     Rod Cutting Problem.

3. FIBONACCI / 1D LINEAR SEQUENCE
   - Core concept: The answer at position `i` depends directly on one or two
     previous positions (`i-1`, `i-2`).
   - Variations: Climbing Stairs, House Robber, Decode Ways, Min Cost Climbing Stairs.

4. LONGEST COMMON SUBSEQUENCE (LCS - 2 Strings / Sequences)
   - Core concept: Given two strings, compare their characters at indices `(i, j)`.
     If they match, move both pointers. If they don't, branch: skip character in
     string 1, or skip character in string 2.
   - Variations: Longest Common Substring, Edit Distance, Shortest Common
     Supersequence, Longest Palindromic Subsequence.

5. LONGEST INCREASING SUBSEQUENCE (LIS)
   - Core concept: Given an array, find the longest subsequence where numbers are
     strictly increasing. At each element, look at all valid predecessors.
   - Variations: Russian Doll Envelopes, Maximum Sum Increasing Subsequence,
     Number of Longest Increasing Subsequences.

6. KADANE'S ALGORITHM (Maximum Subarray DP)
   - Core concept: At each position `i`, decide whether to extend the existing
     running subarray or start a fresh subarray at element `i`.
   - Variations: Maximum Subarray Sum, Maximum Product Subarray, Maximum Circular Subarray.

7. MATRIX CHAIN MULTIPLICATION (Interval / Partition DP)
   - Core concept: Break down a range `[i, j]` by placing a partition `k` at
     every possible split point between `i` and `j`, and find the optimal split.
   - Variations: Minimum Cost to Cut a Stick, Burst Balloons, Palindrome Partitioning.

8. DP ON TREES
   - Core concept: Run a post-order traversal (DFS). Compute optimal answers for
     left and right child subtrees first, then combine them to solve the root.
   - Variations: Binary Tree Maximum Path Sum, House Robber III, Diameter of Binary Tree.

9. DP ON 2D GRIDS (Matrix Traversal)
   - Core concept: Move through an `m x n` grid where movement is restricted
     (e.g., only right or down). State is represented by grid coordinates `(r, c)`.
   - Variations: Unique Paths, Minimum Path Sum, Cherry Pickup, Dungeon Game.

10. STATE COMPRESSION / BITMASK DP
    - Core concept: When subproblem state requires keeping track of a small set
      of visited items (typically n <= 16), use an integer's binary bits as a set.
    - Variations: Traveling Salesman Problem (TSP), Assigning Tasks to Workers.
"""


# ==============================================================================
# SECTION 8: RUNNABLE DEMONSTRATION & PROOF
# ==============================================================================
if __name__ == "__main__":
    from collections import Counter

    print("=" * 70)
    print("MODULE 1: WHY RECURSION ALONE IS NOT ENOUGH")
    print("=" * 70)

    # 1. Single-Branch: Factorial
    print("\n1. SINGLE-BRANCH RECURSION (No overlap, no DP needed):")
    print(f"   Factorial(5) = {factorial(5)}")
    print("   -> Every subproblem (5, 4, 3, 2, 1) was solved exactly ONCE.")

    # 2. Multi-Branch: Climbing Stairs
    print("\n2. MULTI-BRANCH RECURSION (Choices lead to overlapping subproblems):")
    stairs_to_test = 6
    total_ways = instrumented_climb(stairs_to_test)
    print(f"   Ways to climb {stairs_to_test} stairs: {total_ways}")
    print(f"   Total recursive function calls triggered: {len(call_log)}")

    # 3. Analyze the Redundant Work
    counts = Counter(call_log)
    print("\n   Breakdown of how many times each subproblem was recalculated:")
    print("   " + "-" * 55)
    print("   Subproblem (Stairs Left)  |  Number of Times Recomputed")
    print("   " + "-" * 55)
    for subproblem, freq in sorted(counts.items(), key=lambda x: x[0], reverse=True):
        status = "<- DUPLICATE WORK!" if freq > 1 else "<- Base trigger"
        print(f"   climb({subproblem:<2})                   |  {freq:<2} time(s)  {status}")
    print("   " + "-" * 55)

    print("\nKEY TAKEAWAY:")
    print("Notice how climb(2), climb(1), and climb(0) were re-calculated over and over.")
    print("In the next module, we will explore how simply REMEMBERING these results")
    print("turns this slow exponential tree into an ultra-fast linear algorithm!")
