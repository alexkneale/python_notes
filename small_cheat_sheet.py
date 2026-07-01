"""
ADVANCED PYTHON CHEAT SHEET FOR INTERVIEWS
Purpose: To demonstrate deep understanding beyond basic syntax.
"""

# ==============================================================================
# 1. VARIABLE ASSIGNMENT & MEMORY MANAGEMENT
# ==============================================================================
# Python handles variables as references, not memory buckets.
# Everything is an object (even classes!).
# https://www.pythonmorsels.com/everything-is-an-object/#:~:text=Summary-,In%20Python%2C%20everything%20is%20an%20object.,variable%20to%20is%20an%20object.
x = [1, 2, 3]
y = x  # y is a reference to the same object in memory as x
y.append(4)
print(x)  # Output: [1, 2, 3, 4] -> Mutable objects are passed by reference

# Showing off: Understanding 'is' vs '=='
# '==' checks value equality. 'is' checks identity (memory address).
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True
print(a is b)  # False (different objects in memory)

# ==============================================================================
# 2. DEFINING FUNCTIONS (Advanced Concepts)
# ==============================================================================
def advanced_function(*args, **kwargs):
    """
    *args: Arbitrary positional arguments (tuple)
    **kwargs: Arbitrary keyword arguments (dict)
    """
    # Closures: Functions that remember the state of their enclosing scope
    def inner(factor):
        return [arg * factor for arg in args]
    return inner

# Decorators: The "Show Off" feature.
# A decorator is a function that takes a function and extends its behavior.
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    return a + b

# A. Scope: LEGB Rule (Local, Enclosing, Global, Built-in)
# Understanding scope is critical for debugging closures.
x = "Global" 
def outer():
    x = "Enclosing"
    def inner():
        # 'nonlocal' allows modifying variables in the enclosing scope
        nonlocal x 
        x = "Modified Enclosing"
    inner()
    return x

# B. Functional Programming: Lambda, Map, Filter, Reduce
# Using functions as first-class objects
numbers = [1, 2, 3, 4, 5]
# Lambda: Anonymous, one-line functions
squared = list(map(lambda x: x**2, numbers)) 

# C. Default Argument Gotcha
# NEVER use mutable defaults (lists/dicts) as arguments. 
# They are evaluated only once at definition time, not runtime.
def bad_function(item, collection=[]): # <--- AVOID THIS
    collection.append(item)
    return collection

def good_function(item, collection=None):
    if collection is None:
        collection = []
    collection.append(item)
    return collection



# ==============================================================================
# 3. IF/ELSE STATEMENTS & TERNARY OPERATORS
# ==============================================================================
# Python's ternary operator is a one-liner: value_if_true if condition else value_if_false
status = "Admin"
access = "Granted" if status == "Admin" else "Denied"

# The 'match/case' statement (Python 3.10+): Pattern matching
# This is much more powerful than if/elif chains.
command = "quit"
match command:
    case "start":
        print("Starting...")
    case "quit" | "exit":
        print("Quitting...")
    case _:
        print("Unknown command")

# ==============================================================================
# 4. LIST/STRING INDEXING & SLICING
# ==============================================================================
# Python slicing syntax: [start:stop:step]
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Reversing a list in-place:
reversed_data = data[::-1]

# Negative indexing:
last_element = data[-1]

# List Comprehensions: The standard for Pythonic iteration
# Syntax: [expression for item in iterable if condition]
squares = [x**2 for x in range(10) if x % 2 == 0]

# ==============================================================================
# 5. RANGE FUNCTION & GENERATORS
# ==============================================================================
# range(start, stop, step) is an immutable sequence type, not a list.
# Memory efficient for huge loops.

# Generators: Use 'yield' to create a lazy iterator (saves memory)
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# To iterate through:
# for num in fibonacci(10): print(num)

# ==============================================================================
# 6. RECURSION & TAIL CALL OPTIMIZATION (THE "GOTCHA")
# ==============================================================================
# Note: Python does NOT support Tail Call Optimization (TCO) natively.
# Be careful with deep recursion (RecursionError).

def factorial(n):
    """
    Standard recursive approach.
    Bonus: Use 'functools.lru_cache' to memoize results for performance!
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Advanced Interviewer Tip: 
# "While recursion is elegant, Python's recursion limit is usually 1000.
# For production-grade recursion, I'd consider iterative solutions 
# or using @functools.lru_cache to cache previous calculations."

# ==============================================================================
# 7. CLASS BASICS (BONUS SHOW-OFF)
# ==============================================================================
class DataProcessor:
    """
    Using @property to create getter/setter behavior 
    without breaking the public API.
    """
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        if new_val < 0:
            raise ValueError("Value cannot be negative")
        self._value = new_val


# ==============================================================================
# 8. OOP: CLASSES, INHERITANCE & META-PROGRAMMING
# ==============================================================================

# A. Inheritance and Polymorphism
# Python supports Multiple Inheritance (Method Resolution Order - MRO).
class Base:
    def greet(self):
        return "Hello from Base"

class Mixin:
    def log(self):
        print("Logging activity...")

class Derived(Base, Mixin):
    def greet(self):
        # Using super() is the "Pythonic" way to call parent methods.
        # It handles MRO (Method Resolution Order) correctly.
        return super().greet() + " and Derived!"

# B. Encapsulation: "We are all consenting adults here"
# Python doesn't have true private variables, but it has naming conventions:
# single _ (internal use), double __ (name mangling to prevent accidental override).
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Name mangled to _BankAccount__balance

    @property
    def balance(self):
        return self.__balance

# C. Dunder (Magic) Methods
# These allow your classes to act like built-in Python types.
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        # Allows usage of '+' operator: v1 + v2
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        # Defines string representation for print()
        return f"Vector({self.x}, {self.y})"

    def __len__(self):
        # Defines behavior for len()
        return 2

# D. Class Decorators & Static Methods
# @staticmethod: Logic related to the class but doesn't need access to instance/class state.
# @classmethod: Receives 'cls' instead of 'self'. Often used for Factory methods.
class DataFactory:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_json(cls, json_str):
        # Factory method: creates an instance from a different format
        return cls(data=json_str.split(","))

    @staticmethod
    def is_valid(data):
        return len(data) > 0

# E. Abstract Base Classes (ABCs)
# Enforcement of an Interface. Useful for architecting large systems.
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * (self.radius ** 2)

# F. Method Decorators (Advanced)
# A decorator can be used to validate inputs or track performance across methods.
def validate_input(func):
    def wrapper(self, value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        return func(self, value)
    return wrapper

class Calculator:
    @validate_input
    def set_value(self, value):
        self.val = value

# Final Advice:
# Always mention PEP 8 (style guide) and Type Hinting (e.g., def func(a: int) -> int:).
# It shows you write code for teams, not just for compilers.

# type coercion -> https://www.pythonmorsels.com/type-coercion/?watch

# Truthiness -> https://www.pythonmorsels.com/truthiness/?watch
# Truthiness is about asking the question: if we converted this object to a boolean, what would we get?

# None type
# https://www.pythonmorsels.com/none/?watch

# Pass: no-op

# So pass is Python's no-op (meaning no operation should occur). That's what many programming languages call a statement that does nothing.
# You won't see pass very often. When you do see pass, it's because there's a block of code and there's nothing useful to put in it. So someone wrote pass instead.

from random import randint

answer = randint(1, 5)
n = 0
while n != answer:
    try:
        n = int(input("What number (1-5) am I thinking of? "))
    except ValueError:
        pass  # When given a non-number, just keep looping
    else:
        if n == answer:
            print(f"{n} is right!")
        else:
            print(f"Nope. Not {n}.")

class BillingError(Exception):
    pass

# ==============================================================================
# 2. LOOP CONTROL: PASS VS CONTINUE
# ==============================================================================

# CONTINUE: Skips the rest of the current iteration and jumps to the next.
# PASS: A null operation; nothing happens. Used for structural placeholders.

for i in range(5):
    if i == 1:
        continue  # Skips print(1), moves to i=2
    if i == 3:
        pass      # Does nothing, keeps executing the loop body
    print(f"Looping: {i}")

# ==============================================================================
# 3. ITERABLES VS ITERATORS
# ==============================================================================
# Iterable: Any object that can return an iterator (e.g., list, tuple, string).
# Iterator: An object that maintains state and produces values via __next__().

my_list = [1, 2, 3]
iterator = iter(my_list) # Get iterator from iterable
print(next(iterator))    # Manual control over iteration

# This is what 'for' loops do under the hood:
# 1. Call iter() on the object.
# 2. Call next() repeatedly until StopIteration is raised.

# ==============================================================================
# 4. DATA STRUCTURES (TUPLES, SETS, DICTS)
# ==============================================================================

# TUPLES: Immutable. Efficient for fixed data.
# Tip: Use 'namedtuple' for readability over plain tuples.
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)

# SETS: Unordered, unique elements. O(1) average lookup time.
# Used for fast membership testing and removing duplicates.
s = {1, 2, 3, 3, 3} # Result: {1, 2, 3}
if 2 in s: # Extremely fast check
    print("Found it")

# DICTIONARIES: O(1) lookup. Use .get() to avoid KeyError.
data = {"a": 1, "b": 2}
value = data.get("c", 0) # Returns 0 instead of crashing if key doesn't exist

# Tip: Use collections.defaultdict for cleaner code when aggregating data
from collections import defaultdict
counts = defaultdict(int)
for char in "banana":
    counts[char] += 1

# ==============================================================================
# 5. ARRAYS IN NUMPY (HIGH PERFORMANCE)
# ==============================================================================
# Standard Python lists are heterogeneous and slow for math. 
# NumPy arrays are homogeneous (one data type) and optimized for C-speed math.

import numpy as np

# Vectorization: Avoid loops when using NumPy!
arr = np.array([1, 2, 3, 4])
arr_squared = arr ** 2 # Instead of a list comprehension, this operates on the whole array

# Slicing in NumPy (Multi-dimensional)
matrix = np.array([[1, 2], [3, 4]])
# Get everything in the first column:
first_column = matrix[:, 0] 

# Why mention this?
# Because in data-heavy roles, using loops for numerical processing 
# instead of NumPy vectorization is a major "red flag" for senior roles.


# RESERVED WORDS

# Python only has a few dozen reserved words: 
# False, None, True, and, as, assert, async, await, break, class, continue, def, del, elif, else, except, finally, for, from, global, if, 
# import, in, is, lambda, nonlocal, not, or, pass, raise, return, try, while, with, yield.

