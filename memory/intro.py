# ============================================================================
# COMPREHENSIVE PYTHON MEMORY MANAGEMENT TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Topic: Pointers, Mutability, Garbage Collection, and CPython Memory Internals.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import gc
import sys
import copy
import weakref

# ============================================================================
# 1. THE CPYTHON MEMORY ARCHITECTURE (HEAP vs STACK)
# ============================================================================
"""
In CPython, memory is divided into two main areas:
1. The Stack: Stores function call frames, local variable NAMES, and references.
2. The Private Heap: Stores all actual Python OBJECTS and data structures.

Python variables are NOT "buckets" that hold data (like in C or C++).
Instead, Python variables are just string NAMES (labels) pointing to memory 
addresses on the Heap. EVERYTHING is a reference.
"""

def demonstrate_references():
    print("\n--- 1. Variables as References ---")
    
    a = [1, 2, 3] # A list object is created on the Heap. 'a' points to it.
    b = a         # 'b' points to the EXACT SAME object. No copying happens!
    
    # 'is' checks if memory addresses match (identity).
    # '==' checks if values match (equality).
    print(f"a is b? {a is b} (Memory Address: {hex(id(a))})")
    
    b.append(4)
    print(f"We modified 'b'. What is 'a'? {a}") # 'a' is also modified!


# ============================================================================
# 2. MUTABILITY & MEMORY ALLOCATION
# ============================================================================
"""
- Immutable types: int, float, bool, str, tuple, frozenset.
  Once created, their memory block CANNOT be altered. Operations create NEW objects.
- Mutable types: list, dict, set, bytearray, user-defined classes.
  Their memory block can be altered in place.
"""

def demonstrate_mutability():
    print("\n--- 2. Mutability & Memory ---")
    
    # Immutable Example
    x = 10
    original_id = id(x)
    x += 1 # Python cannot change '10' to '11'. It creates '11' and re-points 'x'.
    print(f"Immutable Int - Old ID: {hex(original_id)}, New ID: {hex(id(x))}")
    
    # Mutable Example
    lst = [10]
    original_lst_id = id(lst)
    lst.append(11) # Modifies the existing Heap object in-place.
    print(f"Mutable List  - Old ID: {hex(original_lst_id)}, New ID: {hex(id(lst))}")

    # sys.getsizeof() shows the bytes consumed by the OBJECT itself, 
    # not the objects it contains!
    empty_list_size = sys.getsizeof([])
    print(f"Size of empty list object: {empty_list_size} bytes")


# ============================================================================
# 3. SHALLOW vs DEEP COPY
# ============================================================================
"""
When you actually want to duplicate memory, you use the 'copy' module.
"""

def demonstrate_copying():
    print("\n--- 3. Shallow vs Deep Copy ---")
    
    original = [[1, 2], [3, 4]]
    
    # Shallow Copy: Creates a new outer list, but points to the SAME inner lists.
    shallow = copy.copy(original)
    
    # Deep Copy: Recursively clones the outer AND all inner mutable objects.
    deep = copy.deepcopy(original)
    
    # Modify an inner list
    original[0].append(99)
    
    print(f"Original: {original}")
    print(f"Shallow:  {shallow} <- Inner list was modified because it shares references!")
    print(f"Deep:     {deep} <- Completely independent memory tree.")


# ============================================================================
# 4. OBJECT INTERNING (MEMORY OPTIMIZATIONS)
# ============================================================================
"""
To save memory and speed up execution, CPython pre-allocates certain objects.
1. Small Integer Caching: Integers from -5 to 256 are pre-allocated when Python starts.
2. String Interning: Strings that look like valid identifiers (alphanumeric + underscores)
   are often stored in a singleton dictionary to reuse the same memory address.
"""

def demonstrate_interning():
    print("\n--- 4. Object Interning ---")
    
    # Small ints (-5 to 256) are singletons
    a = 256
    b = 256
    print(f"256 is 256? {a is b}") # True
    
    # To prevent the compiler from optimizing 'x = 257; y = 257' in a script file,
    # we compute it dynamically to show how PVM handles memory allocation.
    x = 256 + 1
    y = 256 + 1
    print(f"257 is 257? {x is y}") # False! Outside the cached array, new objects created.
    
    # String interning
    s1 = "hello_world"
    s2 = "hello_world"
    print(f"'hello_world' is 'hello_world'? {s1 is s2}") # True


# ============================================================================
# 5. GARBAGE COLLECTION PART 1: REFERENCE COUNTING
# ============================================================================
"""
CPython's primary memory management system is Reference Counting.
Every object has a hidden `ob_refcnt` field. 
- When a variable points to it, ref count +1.
- When a variable goes out of scope, or is reassigned, or `del` is called, ref count -1.
- When ref count reaches 0, the memory is IMMEDIATELY freed.
"""

def demonstrate_ref_counting():
    print("\n--- 5. Reference Counting ---")
    
    # Create an object (List) and bind it to name 'data'. Ref count = 1.
    data = [1, 2, 3, 4, 5]
    
    # Why does sys.getrefcount return 2? 
    # Because passing 'data' into the sys.getrefcount() function creates a 
    # temporary reference inside the function's local scope!
    print(f"Ref count of data: {sys.getrefcount(data)} (Expect 2)")
    
    alias = data
    print(f"Ref count after alias: {sys.getrefcount(data)} (Expect 3)")
    
    del alias # Deletes the NAME 'alias', not the list object!
    print(f"Ref count after 'del alias': {sys.getrefcount(data)} (Expect 2)")
    
    # If we exit this function, 'data' goes out of scope, ref count hits 0, memory freed.


# ============================================================================
# 6. GARBAGE COLLECTION PART 2: THE GENERATIONAL GC (CYCLES)
# ============================================================================
"""
Reference counting has one fatal flaw: Circular References.
If Object A points to Object B, and Object B points to Object A, their ref counts
will never reach 0, even if no external variables point to them (Memory Leak).

To fix this, CPython has a secondary "Generational Garbage Collector" (`gc` module)
that periodically pauses execution to scan for unreachable cycles.
"""

class Node:
    def __init__(self, name: str):
        self.name = name
        self.related_node = None
        
    def __del__(self):
        # Dunder method called when the object is actually destroyed in memory.
        print(f"    [Memory Freed] Node '{self.name}' destroyed.")

def create_circular_reference():
    print("\n--- 6. Generational GC (Circular References) ---")
    
    node1 = Node("A")
    node2 = Node("B")
    
    # Create a cycle: A -> B and B -> A
    node1.related_node = node2
    node2.related_node = node1
    
    print("Cycle created. Deleting local variables 'node1' and 'node2'...")
    # These names are removed from the stack, but the objects keep each other alive
    # on the Heap. Their ref count drops to 1, not 0.
    del node1
    del node2
    
    print("Local variables deleted. Objects are still in memory! (No '__del__' printed yet)")
    
    # Manually trigger the Generational GC to find the isolated cycle and nuke it.
    print("Running gc.collect()...")
    collected = gc.collect()
    print(f"GC swept up {collected} unreachable objects.")


# ============================================================================
# 7. WEAK REFERENCES (PREVENTING CYCLES)
# ============================================================================
"""
Sometimes you WANT an object to know about another object without keeping it alive.
Example: Caching systems, or Parent-Child tree relationships.
We use the `weakref` module. A weak reference does NOT increment the ref count.
"""

class HeavyCache:
    def __init__(self, data):
        self.data = data

def demonstrate_weakrefs():
    print("\n--- 7. Weak References ---")
    
    obj = HeavyCache("10GB of data")
    
    # Create a weak reference to 'obj'. Ref count of 'obj' remains 1.
    w_ref = weakref.ref(obj)
    
    print(f"Accessing via weakref: {w_ref().data}")
    
    # If we delete the only strong reference...
    del obj
    
    # The weak reference now returns None, because the object was garbage collected!
    print(f"Accessing via weakref after deletion: {w_ref()}")


# ============================================================================
# 8. MEMORY PROFILING / PITFALLS
# ============================================================================
"""
Common ways to create Memory Leaks in Python:
1. Appending to global lists/dicts and never clearing them.
2. Unclosed files or network connections.
3. Catching Exceptions and storing them globally (Tracebacks hold references to 
   the entire stack frame, keeping local variables alive indefinitely).

Optimizations:
1. Use Generators (`yield`) instead of Lists for massive datasets (lazy eval).
2. Use `__slots__` in classes to prevent dynamic `__dict__` allocation (covered in OOP).
3. Use specialized libraries (NumPy) for huge arrays of numbers; they store data in
   contiguous C-arrays rather than fragmented Python objects.
"""


# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (MEMORY MANAGEMENT)
# ============================================================================
"""
Q1: What are circular references, and how does CPython handle them?
A: A circular reference occurs when two or more objects hold strong references to each other (e.g., A references B, and B references A).
   Because of this, their reference counts can never drop to 0, even if they become completely unreachable from the root program execution.
   CPython solves this with a cyclical, generational garbage collector (GC) that periodically scans objects in memory,
   identifies isolated circular groups of references, and cleans them up.

Q2: What is the purpose of the `weakref` module?
A: `weakref` allows you to create references to objects without increasing their reference count (`ob_refcnt`).
   If only weak references to an object remain, Python will safely deallocate it, and the weak reference will return `None`.
   This is extremely useful for building memory-efficient cache systems, lookup tables, and parent-child tree relations.

Q3: Mention 3 common ways to create a memory leak in Python.
A: 
1. Storing data in global lists or dictionaries that are never pruned or cleared.
2. Holding strong circular references inside custom objects that have custom `__del__` methods in Python versions prior to 3.4.
3. Keeping a reference to a traceback object from a caught exception (as tracebacks keep the entire execution stack frame alive).
"""


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Disable the automatic GC temporarily so we can demonstrate manual collection
    gc.disable()
    
    demonstrate_references()
    demonstrate_mutability()
    demonstrate_copying()
    demonstrate_interning()
    demonstrate_ref_counting()
    
    create_circular_reference()
    
    demonstrate_weakrefs()
    
    # Re-enable GC
    gc.enable()
    print("\n--- Tutorial Complete ---")
    print("Key takeaway: Python manages memory via Reference Counting (instant cleanup)")
    print("and a Generational Garbage Collector (cycle cleanup).")
