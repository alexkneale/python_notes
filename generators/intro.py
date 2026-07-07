# ============================================================================
# COMPREHENSIVE PYTHON GENERATORS TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Topic: yield, state suspension, iterators, coroutines (send/throw), yield from.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import sys
from typing import Generator, Iterator, Iterable, Any

# ============================================================================
# 1. CORE MECHANICS: FUNCTIONS vs GENERATORS
# ============================================================================
"""
A standard function executes line-by-line, computes everything, and hits `return`.
At `return`, the local state (variables) is destroyed.

A generator uses the `yield` keyword. When python sees `yield`, it treats the
function entirely differently. Calling the function doesn't execute its body; 
it returns a "Generator Object". 
When iterated over, the generator runs up to the `yield`, returns the value, 
and SUSPENDS its state in memory. The next iteration resumes EXACTLY where it left off.
"""

def stateful_generator() -> Iterator[str]:
    print("  -> [Generator Internal] Initializing...")
    local_variable = 10
    
    yield f"First yield (local_variable = {local_variable})"
    
    # State is preserved between yields!
    print("  -> [Generator Internal] Resuming after first yield...")
    local_variable += 5
    
    yield f"Second yield (local_variable = {local_variable})"
    
    print("  -> [Generator Internal] Finishing execution...")
    # Falling off the end (or explicitly returning) raises StopIteration.


# ============================================================================
# 2. GENERATOR EXPRESSIONS & MEMORY EFFICIENCY
# ============================================================================
"""
You can create generators using comprehension-like syntax with parentheses `()`.
This is incredibly powerful for memory management (Lazy Evaluation).
"""

def memory_profiling():
    print("\n--- 2. Memory: List Comprehensions vs Generator Expressions ---")
    
    # List Comprehension: Allocates the entire sequence in RAM immediately.
    list_comp = [x ** 2 for x in range(1_000_000)]
    
    # Generator Expression: Allocates almost nothing. Computes ONE item at a time.
    gen_expr = (x ** 2 for x in range(1_000_000))
    
    print(f"List size in memory: {sys.getsizeof(list_comp):,} bytes")
    print(f"Generator size in memory: {sys.getsizeof(gen_expr):,} bytes")
    
    # Generators are SINGLE-USE. Once exhausted, they are empty.
    small_gen = (x for x in range(3))
    print(f"First pass: {list(small_gen)}")
    print(f"Second pass: {list(small_gen)} (Empty! It has been exhausted)")


# ============================================================================
# 3. UNDER THE HOOD: THE ITERATOR PROTOCOL
# ============================================================================
"""
To be iterable (usable in a for-loop), an object must implement `__iter__()`.
To be an iterator, it must also implement `__next__()`.
Generators automatically implement both. A `for` loop is just syntactic sugar 
that catches the `StopIteration` exception gracefully.
"""

def manual_iteration():
    print("\n--- 3. Manual Iteration (How `for` loops work) ---")
    
    gen = (x for x in ["A", "B"])
    
    print(next(gen)) # Outputs "A"
    print(next(gen)) # Outputs "B"
    
    try:
        next(gen)
    except StopIteration:
        print("Caught StopIteration! The generator has no more values.")


# ============================================================================
# 4. ADVANCED: COROUTINES (.send, .throw, .close)
# ============================================================================
"""
Generators are not just data PRODUCERS; they can also CONSUME data.
If you put `yield` on the right side of an assignment (`value = yield`), 
you can push data INTO the generator using the `.send()` method.

Type Hinting a full generator: Generator[YieldType, SendType, ReturnType]
"""

def running_average_coroutine() -> Generator[float, float, str]:
    """A generator that accepts numbers via .send() and yields the current average."""
    total = 0.0
    count = 0
    average = 0.0
    
    try:
        while True:
            # 1. Yields 'average' to the caller.
            # 2. Suspends.
            # 3. When caller uses .send(val), the generator resumes,
            #    and 'new_val' receives the sent value.
            new_val = yield average 
            
            total += new_val
            count += 1
            average = total / count
            
    except ValueError:
        print("  -> [Coroutine Internal] Caught ValueError thrown by caller!")
    finally:
        print("  -> [Coroutine Internal] Cleaning up coroutine resources...")
        
    return "Coroutine Finished" # This string is attached to the StopIteration error

def demonstrate_coroutines():
    print("\n--- 4. Coroutines and .send() ---")
    
    coro = running_average_coroutine()
    
    # PRIMING THE COROUTINE:
    # Before you can send() data in, the generator must advance to the first `yield`.
    # You MUST call next(coro) or coro.send(None) first!
    initial_avg = next(coro)
    print(f"Initial avg: {initial_avg}")
    
    print(f"Sending 10, New Avg: {coro.send(10.0)}")
    print(f"Sending 20, New Avg: {coro.send(20.0)}")
    
    try:
        # .throw() injects an exception into the generator at the point of the yield.
        coro.throw(ValueError("Bad data!"))
    except StopIteration as e:
        print(f"Caught expected StopIteration from coroutine return: {e.value}")
    
    # .close() forcefully terminates the generator (raises GeneratorExit inside it).
    coro.close()


# ============================================================================
# 5. DELEGATING GENERATORS (`yield from`)
# ============================================================================
"""
`yield from` was introduced in Python 3.3. It does two things:
1. Syntactic sugar for: `for item in iterable: yield item`
2. **Crucially**: It establishes a transparent two-way channel between the caller 
   and the sub-generator. Calls to `.send()` and `.throw()` are passed 
   directly to the sub-generator!
"""

def sub_generator() -> Iterator[int]:
    yield 1
    yield 2

def parent_generator_old_way() -> Iterator[int]:
    """The manual, verbose way."""
    for item in sub_generator():
        yield item

def parent_generator_new_way() -> Iterator[int]:
    """The clean way using yield from."""
    yield from sub_generator()
    yield from [3, 4] # Works with any iterable!

# Advanced recursion with `yield from`
def flatten(nested: Iterable) -> Iterator[Any]:
    """Recursively flattens an infinitely nested structure."""
    for item in nested:
        # Check if the item is an iterable (but not a string, to avoid infinite char recursion)
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            # Delegate yielding completely to the recursive call
            yield from flatten(item)
        else:
            yield item

def demonstrate_yield_from():
    print("\n--- 5. Yield From & Delegation ---")
    
    print(f"Parent output: {list(parent_generator_new_way())}")
    
    nested_list = [1, [2, [3, "A"], 4], [5, 6]]
    print(f"Flattened nested list: {list(flatten(nested_list))}")


# ============================================================================
# 6. PRACTICAL USAGE: DATA PIPELINES (Chaining Generators)
# ============================================================================
"""
Because generators evaluate lazily, you can chain them together to process
massive amounts of data without blowing up RAM. Data passes through the 
pipeline one element at a time.
"""

def demonstrate_pipelines():
    print("\n--- 6. Generator Pipelines ---")
    
    # Mocking a huge server log file
    mock_log_lines = [
        "INFO: Server started",
        "ERROR: Database connection timeout",
        "INFO: User logged in",
        "ERROR: Disk space low",
        "DEBUG: Cache cleared"
    ]
    
    # Pipeline Step 1: Read lines (In reality, this would be `for line in file: yield line`)
    def stream_logs() -> Iterator[str]:
        for line in mock_log_lines:
            yield line
            
    # Pipeline Step 2: Filter for errors
    def filter_errors(lines: Iterator[str]) -> Iterator[str]:
        for line in lines:
            if "ERROR" in line:
                yield line
                
    # Pipeline Step 3: Extract the specific message
    def extract_message(lines: Iterator[str]) -> Iterator[str]:
        for line in lines:
            yield line.split(": ")[1]

    # Wire them together! (Notice no data is actually processed yet)
    log_stream = stream_logs()
    error_stream = filter_errors(log_stream)
    message_stream = extract_message(error_stream)
    
    # Consume the pipeline. Data flows through Step 1 -> 2 -> 3 one item at a time.
    print("Extracted Error Messages:")
    for msg in message_stream:
        print(f"  - {msg}")


# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (GENERATORS)
# ============================================================================
"""
Q1: What is a generator, and what are its memory/performance advantages?
A: A generator is a function containing the `yield` keyword. Unlike a standard function that runs to completion 
   and returns a single collection, a generator pauses execution and yields values one at a time.
   - Memory Advantage: It evaluates values lazily, requiring O(1) space complexity regardless of stream size.
   - Speed Advantage: It yields items immediately without waiting to build a complete list.

Q2: What is the difference between `yield` and `yield from`?
A: 
- `yield value` suspends the generator and returns a single value to the caller.
- `yield from iterable` acts as a clean shorthand delegation. It yields all elements from the target sub-generator 
  or iterable directly to the caller, bypassing the need for a manual `for item in iterable: yield item` loop.

Q3: What do `.send()`, `.throw()`, and `.close()` do on a generator object?
A: These methods allow bidirectional communication between the caller and the generator:
   - `gen.send(value)`: Resumes the generator and sends `value` into it, which becomes the result of the active `yield` expression inside the generator.
   - `gen.throw(type, value)`: Raises an exception inside the generator at the point where it was suspended.
   - `gen.close()`: Raises a `GeneratorExit` exception inside the generator, forcing it to clean up resources and terminate.
"""


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("--- 1. Stateful Generators ---")
    gen = stateful_generator()
    print("Caller: Requested first value...")
    print(next(gen))
    print("Caller: Doing some other work...")
    print("Caller: Requested second value...")
    print(next(gen))
    
    memory_profiling()
    manual_iteration()
    demonstrate_coroutines()
    demonstrate_yield_from()
    demonstrate_pipelines()
    
    print("\n--- Tutorial Complete ---")
    print("Key takeaway: Generators map logic to time instead of memory. They allow lazy")
    print("evaluation, massive memory savings, and advanced two-way communication.")
