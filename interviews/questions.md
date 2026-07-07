# Comprehensive Python Technical Interview Q&A Guide

## Table of Contents
1. [Core Foundations & Basics](#1-core-foundations--basics)
2. [Memory Management & Internals](#2-memory-management--internals)
3. [Object-Oriented Programming & Metaprogramming](#3-object-oriented-programming--metaprogramming)
4. [Iterators, Generators & Coroutines](#4-iterators-generators--coroutines)
5. [Concurrency, GIL & Parallelism](#5-concurrency-gil--parallelism)
6. [Compilation, Execution & PVM](#6-compilation-execution--pvm)

---

## 1. Core Foundations & Basics

### Q: Explain the difference between `is` and `==` in Python under the hood.
*   **`==` (Value Equality)**: Calls the `__eq__()` method of the left-hand operand to check if the values of the two objects are logically equal.
*   **`is` (Identity Equality)**: Compares the actual memory addresses of the two objects (equivalent to comparing `id(a) == id(b)`). It is extremely fast because it does not require calling any methods.
```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True (same values)
print(a is b)  # False (different locations in heap memory)
```

### Q: What is integer interning and string interning?
*   **Integer Interning**: CPython pre-allocates and caches a global array of integers from `-5` to `256`. Any reference to an integer in this range points to the exact same shared object in memory.
*   **String Interning**: String literals that look like identifiers (alphanumeric and underscores) are automatically interned by the compiler to optimize lookup speeds in dictionaries.
```python
x = 256
y = 256
print(x is y)  # True (Cached singleton range)

x = 257
y = 257
print(x is y)  # False (Out of cached range, separate heap allocations)
```

### Q: Why is using mutable default arguments in Python considered an anti-pattern?
When a function is defined, its default arguments are evaluated **once** at definition time, not every time the function is called. If the default is a mutable object (like a `list` or `dict`), all subsequent function calls without an explicit argument will share the exact same object.
```python
# BAD
def append_to(element, target=[]):
    target.append(element)
    return target

# GOOD
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target
```

---

## 2. Memory Management & Internals

### Q: How does Python manage memory? Describe reference counting and generational garbage collection.
Python uses two primary mechanisms to manage heap-allocated objects:
1.  **Reference Counting (Immediate)**: Every Python object contains an internal integer field (`ob_refcnt`) tracking how many references point to it. When `ob_refcnt` drops to `0`, Python immediately deallocates the memory.
2.  **Generational Garbage Collection (Cycle Detector)**: To solve "circular references" (e.g., Object A references Object B, and Object B references Object A, preventing their counts from ever reaching 0), Python runs a background cyclical GC. It groups objects into 3 generations (Gen 0, 1, and 2) based on survival age. Older generations are scanned less frequently, detecting and freeing unreachable cycles.

### Q: What is a memory leak in Python, and how can it occur if we have automatic garbage collection?
A memory leak occurs when objects are no longer needed but remain reachable by the program's root references, preventing the garbage collector from freeing them.
**Common Causes:**
*   **Global Variables**: Appending items to a global list or dictionary and never clearing them.
*   **Unclosed Resources**: Files, sockets, and DB connections staying open.
*   **Tracebacks in Exceptions**: Storing exceptions or tracebacks globally keeps the local variables of that entire call stack frame alive.
*   **Circular References with Custom `__del__` (Pre-Python 3.4)**: Before Python 3.4, the GC could not safely collect cycles containing objects with custom destructors.

---

## 3. Object-Oriented Programming & Metaprogramming

### Q: How does Python resolve multiple inheritance conflicts? Explain Method Resolution Order (MRO).
Python uses the **C3 Linearization** algorithm to compute a deterministic Method Resolution Order (MRO) for any class hierarchy. MRO maintains two key properties:
1.  **Subclass First**: Parents cannot be checked before their subclasses.
2.  **Local Precedence**: The order of parent classes in the class definition is preserved from left to right.
You can view any class's MRO using the `__mro__` attribute or the `.mro()` method.
```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.__mro__)
# Output: (D, B, C, A, object)
```

### Q: What is the purpose of `__slots__`?
By default, every Python instance stores its attributes in a dynamic dictionary (`__dict__`). This allows adding attributes at runtime but incurs significant memory overhead.
By defining `__slots__` as a sequence of attribute names, Python bypasses `__dict__` and allocates a fixed-size array in memory for these attributes.
```python
class HeavyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class OptimizedClass:
    __slots__ = ('x', 'y')  # Saves considerable memory per instance
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### Q: What is a Metaclass?
A metaclass is a "class of a class." While standard classes define the behavior of *instances*, metaclasses define the behavior and structure of *classes* themselves. 
Every class in Python is an instance of a metaclass (by default, `type`). Metaclasses are typically used to enforce API patterns, register subclasses automatically, or validate class structures at definition time.

---

## 4. Iterators, Generators & Coroutines

### Q: What is the difference between an Iterable and an Iterator?
*   **Iterable**: Any object that implements `__iter__()` (returning an iterator) or `__getitem__()` (sequence lookup). It represents a reusable container of elements.
*   **Iterator**: An active stream of data that implements `__next__()` (returning the next value or raising `StopIteration`) and `__iter__()` (returning `self`). It is stateful and consumed in a single pass.

### Q: What is a generator, and what are its advantages?
A generator is a specialized, compact function that uses the `yield` keyword instead of `return`. 
*   **Lazy Evaluation**: It computes values on-the-fly, pausing execution until the caller requests the next item.
*   **O(1) Space Complexity**: Instead of materializing a list of a million items in RAM, a generator calculates one value at a time, utilizing minimal memory regardless of dataset size.

---

## 5. Concurrency, GIL & Parallelism

### Q: What is the GIL (Global Interpreter Lock), and why does it exist?
The **Global Interpreter Lock** is a mutex used by CPython to ensure that only **one native thread** executes Python bytecode at any given time.
*   **Why it exists**: It prevents race conditions on CPython's internal memory management (which is not thread-safe by default due to reference counting).
*   **Impact**: It prevents pure multithreaded Python code from taking advantage of multiple CPU cores for CPU-bound tasks.
*   **Workaround**: Use multiprocessing (which spawns separate processes, each with its own GIL), write C extensions, or leverage specialized libraries (NumPy, Cython) that release the GIL during execution.

### Q: When should you use Threading vs. Multiprocessing vs. Asyncio?
*   **Multithreading (`threading`)**: Best for **I/O-bound** tasks where the program spends most of its time waiting (e.g., waiting for network APIs, files, or database queries). While waiting, Python releases the GIL, allowing other threads to run.
*   **Multiprocessing (`multiprocessing`)**: Best for **CPU-bound** tasks requiring heavy computations (e.g., image processing, training ML models, or parsing large datasets). Spawns separate processes with independent memory and GILs.
*   **Asyncio (`asyncio`)**: Best for **high-concurrency, single-threaded I/O-bound** tasks (e.g., high-throughput web servers or web scrapers). Uses an event loop to run cooperative tasks (coroutines) concurrently, avoiding thread-switching overhead.

---

## 6. Compilation, Execution & PVM

### Q: Is Python compiled or interpreted?
**Both**. Python is compiled to bytecode first, then interpreted by the Python Virtual Machine (PVM).
1.  **Compilation**: When you run a script, the interpreter compiles the source code (`.py`) into an intermediate representation called **Bytecode** (`.pyc` files). This stage is fast and mainly validates syntax and structure.
2.  **Interpretation**: The PVM (a stack-based virtual machine) reads and executes the bytecode instructions step-by-step.

### Q: How does Name Lookup work in Python?
Python resolves names using the **LEGB rule**:
1.  **L (Local)**: Inside the current function block.
2.  **E (Enclosing)**: Inside any enclosing or nested functions (closures).
3.  **G (Global)**: At the module level of the current file.
4.  **B (Built-in)**: Python's globally built-ins (e.g., `print()`, `len()`, `ValueError`).
If a name is not found in any of these namespaces, Python raises a `NameError`.
