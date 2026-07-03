# ============================================================================
# COMPREHENSIVE PYTHON BASICS & UNDER THE HOOD MECHANICS TUTORIAL
# ============================================================================
# Target Audience: Experienced developers refreshing foundational elements.
# Topic: Core Types, Memory Layouts, Collections, Loops, Strings, and Scopes.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import sys

# ============================================================================
# 1. CORE DATA TYPES & MUTABILITY vs IMMUTABILITY
# ============================================================================
"""
Python variables are dynamic references to heap-allocated objects.
The fundamental types are split into two categories:

1. Immutable (Value cannot change in-place; modifications return a new object):
   - int, float, bool, str, tuple, NoneType.
2. Mutable (Value can be changed in-place; references are modified together):
   - list, dict, set, bytearray.
"""

def demonstrate_mutability_basics():
    print("\n--- 1. Core Types & Mutability ---")
    
    # Immutable string modification (forces a new object allocation)
    s1 = "hello"
    old_id = id(s1)
    s1 += " world"
    new_id = id(s1)
    print(f"String changed? Yes. Old ID: {hex(old_id)}, New ID: {hex(new_id)}")
    
    # NoneType is a singleton (only one 'None' object exists in memory)
    n1 = None
    n2 = None
    print(f"Is None a singleton? {n1 is n2} (Address: {hex(id(None))})")
    
    # The 'is' operator checks if identity (addresses) are identical.
    # The '==' operator checks if values are equal.


# ============================================================================
# 2. PYTHON COLLECTIONS: UNDER THE HOOD
# ============================================================================
"""
Understanding the computational complexity and internal structures of Python's
built-in collections is key to writing high-performance code.

A. LISTS (Dynamic Arrays)
   - Store references to other objects (heterogeneous).
   - Appending is O(1) amortized because Python overallocates capacity.
   - Inserting/Deleting at index 0 is O(N) because all elements must shift.

B. TUPLES (Fixed Arrays)
   - Smaller memory footprint than lists.
   - Immutable, making them hashable if all their elements are hashable.

C. DICTIONARIES (Hash Tables)
   - Average case O(1) lookup, insertion, and deletion.
   - Dict keys MUST be 'hashable' (must implement __hash__ and __eq__).
   - In Python 3.7+, dicts maintain insertion order by default!

D. SETS (Hash Tables without values)
   - Average case O(1) membership testing (`in` operator).
   - Perfect for removing duplicates and set algebra (unions, intersections).
"""

def demonstrate_collections_internals():
    print("\n--- 2. Collections and Memory Footprints ---")
    
    # Compare memory size of List vs Tuple for the same data
    lst = [1, 2, 3, 4, 5]
    tup = (1, 2, 3, 4, 5)
    print(f"Size of List:  {sys.getsizeof(lst)} bytes")
    print(f"Size of Tuple: {sys.getsizeof(tup)} bytes (Tuples are more memory-efficient!)")
    
    # Set Operations (Algebra)
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    print(f"Union (A | B):        {set_a | set_b}")
    print(f"Intersection (A & B): {set_a & set_b}")
    print(f"Difference (A - B):   {set_a - set_b}")
    
    # Dict key requirements
    # Any hashable object can be a key. Lists cannot be keys!
    try:
        invalid_dict = { [1, 2]: "failed" }
    except TypeError as e:
        print(f"Using a List as a Dict key failed as expected: {e}")


# ============================================================================
# 3. CONTROL FLOW, ADVANCED LOOPING, & COMPREHENSIONS
# ============================================================================
"""
A. Ternary Operator (Conditional expression)
   Syntax: `value_if_true if condition else value_if_false`

B. Loop-Else Clause (A unique Python feature)
   Both `for` and `while` loops can have an `else` block!
   - The `else` block runs ONLY if the loop completed naturally without hitting a `break`.
   - If a `break` is hit, the `else` block is bypassed. Useful for search loops.

C. Comprehensions
   Syntactic sugar for creating lists, dicts, and sets inline.
"""

def demonstrate_control_flow():
    print("\n--- 3. Control Flow & Loop-Else ---")
    
    # Ternary
    age = 20
    status = "Adult" if age >= 18 else "Minor"
    print(f"Ternary Status: {status}")
    
    # For-Else search pattern
    target_num = 7
    numbers = [1, 3, 5, 9]
    
    print("Searching for 7 in list:")
    for num in numbers:
        if num == target_num:
            print("  Found it!")
            break
    else:
        # This executes because the loop finished without encountering a 'break'
        print("  Finished searching: 7 was NOT in the list.")
        
    # Comprehensions
    # List comprehension with filtering
    evens_squared = [x**2 for x in range(10) if x % 2 == 0]
    print(f"Evens squared: {evens_squared}")
    
    # Dict comprehension
    char_map = {char: idx for idx, char in enumerate("abc")}
    print(f"Char map: {char_map}")


# ============================================================================
# 4. STRING MANIPULATION & THE CONCATENATION PITFALL
# ============================================================================
"""
Because strings are immutable, adding two strings (`s = s + "x"`) creates 
an entirely new string object in memory, copying all characters from the original.

Doing this in a loop results in quadratic complexity: O(N^2) runtime!
The correct, Pythonic way to combine multiple strings is to collect them 
in a list and join them with `.join()`, which runs in linear O(N) time.
"""

def demonstrate_string_performance():
    print("\n--- 4. String Manipulation & Concatenation Performance ---")
    
    words = ["word"] * 1000
    
    # Bad approach (quadratic O(N^2) memory re-allocations)
    start_time = time_nanos()
    result_bad = ""
    for w in words:
        result_bad += w
    bad_duration = time_nanos() - start_time
    
    # Good approach (linear O(N) pre-computed allocation)
    start_time = time_nanos()
    result_good = "".join(words)
    good_duration = time_nanos() - start_time
    
    print(f"Join speed is generally much faster and uses far fewer memory re-allocations!")
    print(f"Strings match? {result_bad == result_good}")


def time_nanos():
    import time
    return time.perf_counter_ns()


# ============================================================================
# 5. FUNCTIONS, PARAMETERS, & THE SCOPE (LEGB) RULE
# ============================================================================
"""
A. Argument Unpacking
   - `*args` captures arbitrary positional arguments as a Tuple.
   - `**kwargs` captures arbitrary keyword arguments as a Dict.

B. The Mutable Default Argument Gotcha
   Default arguments are evaluated ONCE at function definition time, NOT at execution.
   If you use a mutable default parameter (like a list `[]`), that same object 
   is shared across all function calls! Always default to `None` instead.

C. The LEGB Scope Resolution Rule
   Python resolves variable names in a strict order:
   1. Local (L): Inside the current function.
   2. Enclosing (E): Inside nested outer functions (nonlocal).
   3. Global (G): Top-level module namespace.
   4. Built-in (B): Python's pre-loaded objects (len, range, ValueError).
"""

def mutable_default_gotcha(item, collection=[]):
    collection.append(item)
    return collection

def good_default_alternative(item, collection=None):
    if collection is None:
        collection = []
    collection.append(item)
    return collection

def demonstrate_functions_scopes():
    print("\n--- 5. Functions & Default Argument Gotchas ---")
    
    # Calling bad function multiple times
    res1 = mutable_default_gotcha(1)
    res2 = mutable_default_gotcha(2)
    print(f"Mutable Default Gotcha - Call 1: {res1}")
    print(f"Mutable Default Gotcha - Call 2: {res2}")
    print(f"  Are they the exact same object in memory? {res1 is res2}")
    print("  (Both printed [1, 2] because they point to the exact same class-level default list!)")
    
    # Calling good function multiple times
    good1 = good_default_alternative(1)
    good2 = good_default_alternative(2)
    print(f"Correct Pattern - Call 1: {good1}")
    print(f"Correct Pattern - Call 2: {good2} (Perfect! Fresh list created each call.)")


# ============================================================================
# 6. BASIC ERROR HANDLING (TRY-EXCEPT-ELSE-FINALLY)
# ============================================================================
"""
Understanding how Python handles exceptions is key to writing robust software.

- `try`: Run code that might raise an exception.
- `except`: Capture and handle specified exceptions (avoid generic 'except Exception:').
- `else`: Run code ONLY if no exceptions were raised in the try block.
- `finally`: Run code always, whether an exception occurred or not (used for cleanup).
"""

def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("  [Except] Attempted to divide by zero!")
        result = None
    else:
        print("  [Else] Division completed successfully!")
    finally:
        print("  [Finally] Division attempt finished!")
    return result

def demonstrate_exception_basics():
    print("\n--- 6. Basic Exception Handling Flow ---")
    
    print("Call with valid inputs (10, 2):")
    divide(10, 2)
    
    print("\nCall with invalid inputs (10, 0):")
    divide(10, 0)


# ============================================================================
# 7. FILE I/O AND BASIC CONTEXT MANAGERS
# ============================================================================
"""
When reading and writing files, you should always use the `with` statement 
(the context manager pattern).

Doing:
```python
f = open('file.txt')
data = f.read()
f.close()
```
is prone to resource leaks. If an exception occurs during `read()`, the file 
is never closed.

Using `with open(...)` guarantees that the file stream is closed automatically 
upon leaving the block, even if exceptions are raised.
"""

def demonstrate_file_io():
    print("\n--- 7. File I/O with Context Managers ---")
    
    filename = "scratch_temp.txt"
    
    # Writing to a file
    with open(filename, "w") as f:
        f.write("Hello, interview prep basics!\nLine two.")
    
    # Reading from a file
    with open(filename, "r") as f:
        content = f.read()
        print("File Content Read:")
        for line in content.splitlines():
            print(f"  > {line}")
            
    # Clean up the temp file
    import os
    if os.path.exists(filename):
        os.remove(filename)


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    demonstrate_mutability_basics()
    demonstrate_collections_internals()
    demonstrate_control_flow()
    demonstrate_string_performance()
    demonstrate_functions_scopes()
    demonstrate_exception_basics()
    demonstrate_file_io()
    
    print("\n--- Basics Tutorial Complete ---")
    print("Key takeaway: Python basics are simple on the surface, but understanding")
    print("mutability, allocations, collections complexity, and the LEGB scope")
    print("resolution rules makes your code robust and fast.")
