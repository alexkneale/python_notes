# ============================================================================
# COMPREHENSIVE PYTHON ITERABLES, ITERATORS & THE ITERATOR PROTOCOL
# ============================================================================
# Target Audience: Experienced developers refreshing foundational elements.
# Topic: Iterables vs. Iterators, the Iterator Protocol, Custom Collections,
#        Under-the-Hood mechanics, gotchas, and interview-essential itertools.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import sys
import collections.abc
from itertools import count, cycle, repeat, chain, islice, groupby, permutations, combinations, product

# ============================================================================
# 1. CORE CONCEPTS: ITERABLES vs. ITERATORS
# ============================================================================
"""
At a high level, the difference is between a "container of data" and a "stream of data".

A. ITERABLE:
   - An object capable of returning its members one at a time.
   - Must implement:
     1. `__iter__()` which returns an Iterator object.
     2. OR `__getitem__()` with sequential integer indexes starting at 0 (Sequence fallback).
   - Examples: list, tuple, str, dict, set, range.
   - Typically stateless (reusable multiple times).

B. ITERATOR:
   - An object representing a stream of data.
   - Must implement the ITERATOR PROTOCOL, which consists of two methods:
     1. `__next__()` -> returns the next item. If no more items remain, raises `StopIteration`.
     2. `__iter__()` -> returns the iterator itself (`return self`).
   - Stateful (tracks its position in the stream). Once consumed, it is exhausted!

C. THE BUILT-IN FUNCTIONS:
   - `iter(obj)` calls `obj.__iter__()`.
   - `next(it)` calls `it.__next__()`.
"""

def demonstrate_concepts():
    print("\n" + "="*50)
    print("1. CORE CONCEPTS: ITERABLES vs. ITERATORS")
    print("="*50)

    my_list = [10, 20, 30]  # This is an Iterable
    
    # 1. Verification via Collections Abstract Base Classes (ABCs)
    print(f"Is list an Iterable? {isinstance(my_list, collections.abc.Iterable)}")
    print(f"Is list an Iterator? {isinstance(my_list, collections.abc.Iterator)}")

    # 2. Get an iterator from the list using iter()
    my_iterator = iter(my_list)
    print(f"Is the result of iter(list) an Iterable? {isinstance(my_iterator, collections.abc.Iterable)}")
    print(f"Is the result of iter(list) an Iterator? {isinstance(my_iterator, collections.abc.Iterator)}")

    # 3. Retrieve elements one by one
    print(f"First next():  {next(my_iterator)}")
    print(f"Second next(): {next(my_iterator)}")
    print(f"Third next():  {next(my_iterator)}")

    # 4. What happens when it's empty?
    try:
        next(my_iterator)
    except StopIteration:
        print("Fourth next(): Caught StopIteration! The iterator is exhausted.")


# ============================================================================
# 2. UNDER THE HOOD OF A FOR-LOOP
# ============================================================================
"""
When you write:
    for item in iterable:
        # process item

Python actually executes the equivalent of:
    _iterator = iter(iterable)
    while True:
        try:
            item = next(_iterator)
        except StopIteration:
            break
        # process item
"""

def demonstrate_for_loop_under_the_hood():
    print("\n" + "="*50)
    print("2. UNDER THE HOOD OF A FOR-LOOP")
    print("="*50)

    colors = ["red", "green", "blue"]
    print("Normal for-loop output:")
    for color in colors:
        print(f"  {color}")

    print("\nManual while-loop emulation:")
    # 1. Get the iterator
    color_iter = iter(colors)
    # 2. Infinite loop to retrieve values
    while True:
        try:
            color = next(color_iter)
            print(f"  {color}")
        except StopIteration:
            # 3. Clean exit when StopIteration is raised
            print("  [StopIteration caught, exiting loop gracefully]")
            break


# ============================================================================
# 3. CUSTOM ITERATORS & ITERABLES (THE INTERVIEW GOLDMINE)
# ============================================================================
"""
In interviews, you might be asked to implement a custom data structure (like a 
Binary Search Tree, Linked List, or Custom Range) that can be iterated over.

There are two primary design patterns:
1. The Consolidated Pattern (Self-Iterator):
   - The class implements BOTH `__iter__` and `__next__`.
   - `__iter__` returns `self`.
   - Problem: It can only be iterated over concurrently ONCE. Two nested loops will conflict 
     because they share the same state.

2. The Separated Pattern (Iterable + Iterator classes):
   - The main class is the Iterable; its `__iter__` returns a NEW instance of a separate Iterator class.
   - This allows multiple independent, concurrent, and nested iterations.
"""

# --- Pattern 1: Consolidated Self-Iterator (e.g., a simple countdown) ---
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        # By protocol, an iterator's __iter__ must return itself!
        # This allows it to be used in loops or functions that expect iterables.
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# --- Pattern 2: Separated Iterable and Iterator (e.g., custom playlist) ---
class Playlist:
    """The Iterable: Holds the data container."""
    def __init__(self, songs):
        self.songs = songs

    def __iter__(self):
        # Returns a completely new Iterator instance, resetting state per loop
        return PlaylistIterator(self.songs)

class PlaylistIterator:
    """The Iterator: Maintains the cursor/state for one traversal."""
    def __init__(self, songs):
        self.songs = songs
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.songs):
            raise StopIteration
        song = self.songs[self.index]
        self.index += 1
        return song


def demonstrate_custom_iteration():
    print("\n" + "="*50)
    print("3. CUSTOM ITERATORS & ITERABLES")
    print("="*50)

    # 1. Testing Consolidated Self-Iterator
    print("Consolidated Countdown iterator:")
    counter = Countdown(3)
    for num in counter:
        print(f"  {num}")

    # 2. Testing Separated Pattern (Nested Loops)
    print("\nSeparated Playlist Iterable (Nested iteration test):")
    my_playlist = Playlist(["Song A", "Song B"])
    
    # Nested loops work perfectly because each loop calls iter(playlist), 
    # creating a separate, independent PlaylistIterator state.
    for song1 in my_playlist:
        for song2 in my_playlist:
            print(f"  Outer: {song1} | Inner: {song2}")


# ============================================================================
# 4. INTERVIEW GOTCHAS & PITFALLS
# ============================================================================
"""
A. GOTCHA 1: Iterator Exhaustion (One-Shot Consumption)
   - Iterators are active streams, not static containers. Once you consume them 
     (e.g., using a list comprehension, sum(), or putting them in a list), they 
     are completely exhausted.
   - If you try to iterate again, they will instantly raise StopIteration.

B. GOTCHA 2: Mutating a Collection while Iterating
   - DICTIONARIES/SETS: Modifying keys (inserting or deleting) during iteration 
     will raise: `RuntimeError: dictionary changed size during iteration`.
   - LISTS: Will NOT raise an error, but will silently skip elements or produce
     duplicate operations because Python tracks iteration by index pointers!
     - Solution: Iterate over a copy of the list/dict, or collect items to modify 
       and apply them after iteration, or use list comprehensions to build new ones.

C. GOTCHA 3: Lazy Evaluation Memory Efficiency
   - Generators and iterators compute values on-demand (lazy evaluation).
   - Lists pre-compute and store all values in memory (eager evaluation).
   - This makes Iterators O(1) space, whereas Lists are O(N) space.
"""

def demonstrate_gotchas():
    print("\n" + "="*50)
    print("4. COMMON INTERVIEW GOTCHAS & PITFALLS")
    print("="*50)

    # --- GOTCHA 1: Iterator Exhaustion ---
    numbers = [1, 2, 3, 4]
    num_iter = iter(numbers)
    
    # First consumption
    list_one = list(num_iter)
    print(f"First list conversion:  {list_one}")
    
    # Second consumption
    list_two = list(num_iter)
    print(f"Second list conversion: {list_two} (Notice it is empty!)")

    # --- GOTCHA 2: Mutating Collections ---
    # 2a. Dictionary Mutation
    my_dict = {"a": 1, "b": 2, "c": 3}
    print("\nAttempting to delete from dict during iteration:")
    try:
        for key in my_dict:
            if key == "b":
                del my_dict[key]
    except RuntimeError as e:
        print(f"  Caught expected error: {e}")

    # 2b. List Mutation (The silent killer!)
    # Goal: Remove all even numbers. Let's see why raw iteration fails.
    lst = [2, 4, 6, 8, 10]
    print(f"\nInitial list: {lst}")
    for item in lst:
        if item % 2 == 0:
            lst.remove(item)
    print(f"List after in-place deletion attempt: {lst}")
    print("  ^ Why [4, 8] remained: When 2 is removed at index 0, index 1 becomes 4. ")
    print("    But Python's iterator advances to index 1, skipping 4 entirely!")

    # Correct Pattern: Iterate over a copy or use comprehension
    lst_correct = [2, 4, 6, 8, 10]
    lst_correct = [x for x in lst_correct if x % 2 != 0]
    print(f"Correct list modification using list comprehension: {lst_correct}")

    # --- GOTCHA 3: Memory footprint comparison ---
    # Generator expression (iterator) vs List Comprehension
    mega_range = 10_000_000
    
    # Iterator: computed lazily on-demand
    lazy_gen = (x * 2 for x in range(mega_range))
    # List: fully allocated in memory
    eager_lst = [x * 2 for x in range(mega_range)]

    print(f"\nMemory Footprint (10 million items):")
    print(f"  Generator Iterator Size: {sys.getsizeof(lazy_gen)} bytes (O(1) Memory)")
    print(f"  List Comprehension Size: {sys.getsizeof(eager_lst)} bytes (O(N) Memory)")


# ============================================================================
# 5. THE ULTIMATE ITERATION COMPANION: ITERTOOLS
# ============================================================================
"""
In coding interviews, implementing complex loops manually can be slow and bug-prone.
The standard library's `itertools` provides fast, memory-efficient C-implementations
for common iteration patterns.

Below are the absolute must-know tools for Python technical interviews:

1. Infinite Iterators:
   - `count(start, step)` -> infinite stream of incrementing numbers.
   - `cycle(iterable)` -> cycles through an iterable indefinitely.
   - `repeat(elem, [n])` -> repeats an element indefinitely or N times.

2. Terminating/Filtering Iterators:
   - `chain(*iterables)` -> treats multiple sequences as a single contiguous sequence.
   - `islice(iterable, start, stop, step)` -> slices an iterator/generator without converting it to a list.
   - `groupby(iterable, key=None)` -> groups consecutive duplicate/matching keys. (Requires sorted input!)

3. Combinatoric Iterators:
   - `product(*iterables, repeat=1)` -> Cartesian product. Prevents deep nested loops!
   - `permutations(iterable, r)` -> All unique orderings of length R.
   - `combinations(iterable, r)` -> All unique subsets of length R (order doesn't matter).
"""

def demonstrate_itertools():
    print("\n" + "="*50)
    print("5. STANDARD LIBRARY MAGIC: ITERTOOLS")
    print("="*50)

    # 1. itertools.islice (slice an infinite or lazy stream)
    print("islice(count(10, 5), 3) -> slice of infinite incrementing generator:")
    infinite_stream = count(start=10, step=5)
    sliced_stream = islice(infinite_stream, 3) # stops after 3 items
    print(f"  {list(sliced_stream)}")

    # 2. itertools.chain (combine multiple iterables cleanly)
    print("\nchain() -> flatter, cleaner concatenation:")
    list_a = [1, 2]
    list_b = [3, 4]
    combined = chain(list_a, list_b)
    print(f"  {list(combined)}")

    # 3. itertools.groupby (group consecutive items)
    print("\ngroupby() -> perfect for run-length encoding (RLE):")
    data = "AAABBCADDD"
    # Group consecutive matching characters
    grouped = [(char, len(list(group))) for char, group in groupby(data)]
    print(f"  RLE of '{data}': {grouped}")

    # 4. itertools.product (Cartesian product - replaces nested loops!)
    print("\nproduct() -> clean replacement for nested loops:")
    ranks = ['A', 'K']
    suits = ['♠', '♥']
    # Instead of two nested for-loops:
    deck = list(product(ranks, suits))
    print(f"  Cards: {deck}")

    # 5. combinations and permutations
    print("\npermutations() vs combinations():")
    items = [1, 2, 3]
    print(f"  Permutations of {items} (len 2): {list(permutations(items, 2))}")
    print(f"  Combinations of {items} (len 2): {list(combinations(items, 2))}")


# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (ITERATORS)
# ============================================================================
"""
Q1: What is the fundamental difference between an Iterable and an Iterator?
A: 
- An Iterable is a reusable container of items (like a list, string, or tuple) that implements `__iter__()` 
  to return an Iterator. It is typically stateless.
- An Iterator is a stateful stream of data that implements the Iterator Protocol: `__next__()` (to return 
  the next element or raise `StopIteration`) and `__iter__()` (returning `self`). It can only be consumed once.

Q2: Why must an Iterator's `__iter__()` method return `self`?
A: By having `__iter__()` return `self`, an Iterator is itself also an Iterable.
   This allows an iterator to be used in contexts that expect any iterable, such as `for` loops, 
   list comprehensions, or passing it to built-in functions like `sum()`, `max()`, or `any()`.

Q3: What are the risks of mutating a collection (like a list or dict) while iterating over it?
A: 
- For dicts/sets: Python will raise a `RuntimeError: dictionary changed size during iteration` to protect 
  hash-table integrity.
- For lists: Python will not raise an error, but because it tracks iteration by an internal index pointer, 
  removing or inserting elements will shift indices, leading to items being skipped or processed multiple times.
"""


# ============================================================================
# MAIN EXECUTION ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("============================================================================")
    print("RUNNING PYTHON ITERATOR REFERENCE GUIDE DEMO")
    print("============================================================================")
    
    demonstrate_concepts()
    demonstrate_for_loop_under_the_hood()
    demonstrate_custom_iteration()
    demonstrate_gotchas()
    demonstrate_itertools()
    
    print("\n" + "="*50)
    print("DEMO RUN COMPLETE. Review this file for code patterns & interview study!")
    print("="*50)
