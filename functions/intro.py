# ============================================================================
# COMPREHENSIVE PYTHON FUNCTIONS TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Instructions: You can run this entire file. It is syntactically valid Python.
# Read through the comments and functions to understand the deep mechanics.
# ============================================================================

import time
import inspect
from functools import wraps, partial, lru_cache
from typing import Callable, Any, TypeVar, ParamSpec, Generator, Iterator

# ============================================================================
# 1. ADVANCED PARAMETERS AND ARGUMENTS
# ============================================================================

# PEP 570 introduced Positional-Only parameters (`/`).
# PEP 3102 introduced Keyword-Only parameters (`*`).
def complex_signature(pos_only1, pos_only2, /, standard1, standard2, *, kw_only1, kw_only2=None):
    """
    Args before `/` MUST be positional. They cannot be passed by keyword.
    Args between `/` and `*` can be positional OR keyword.
    Args after `*` MUST be passed by keyword.
    
    Why use this? 
    - `/` prevents callers from relying on parameter names that might change.
    - `*` forces explicit naming for boolean flags or optional configs, improving readability.
    """
    return f"{pos_only1}, {standard1}, {kw_only1}"

# *args and **kwargs (Packing and Unpacking)
def argument_packer(*args: int, **kwargs: str):
    """
    *args packs excess positional arguments into a tuple.
    **kwargs packs excess keyword arguments into a dictionary.
    """
    # args is a tuple: e.g., (1, 2, 3)
    # kwargs is a dict: e.g., {'name': 'Alice', 'role': 'Admin'}
    pass

def argument_unpacker():
    my_list = [1, 2, 3]
    my_dict = {"kw_only1": "hello", "kw_only2": "world"}
    
    # We can use * and ** to UNPACK iterables/mappings into function calls.
    # We are calling the `complex_signature` function defined above.
    return complex_signature(*my_list[:2], standard1="foo", standard2="bar", **my_dict)


# ============================================================================
# 2. SCOPE, THE LEGB RULE, AND CLOSURES
# ============================================================================
# Python resolves scope using LEGB: Local, Enclosing, Global, Built-in.

GLOBAL_VAR = "I am global"

def scope_demonstrator():
    enclosing_var = "I am enclosed"
    
    def inner_function():
        # Using nonlocal allows us to mutate a variable in the ENCLOSING scope.
        # Without `nonlocal`, `enclosing_var = ...` would just create a new local variable.
        nonlocal enclosing_var
        enclosing_var = "I am modified enclosed"
        
        # Using global allows us to mutate a variable in the GLOBAL scope.
        global GLOBAL_VAR
        GLOBAL_VAR = "I am modified global"
        
        local_var = "I am local" # Lives and dies in this function
        return local_var
        
    inner_function()
    return enclosing_var

# Closures
# A closure occurs when a nested function captures and remembers the state 
# of its enclosing scope, EVEN AFTER the outer function has finished executing.
def make_multiplier(factor: int) -> Callable[[int], int]:
    """Returns a function that multiplies by 'factor'."""
    def multiplier(number: int) -> int:
        # 'factor' is captured from the enclosing scope.
        # It lives on in the closure's memory (__closure__ attribute).
        return number * factor
    
    return multiplier


# ============================================================================
# 3. FIRST-CLASS & HIGHER-ORDER FUNCTIONS
# ============================================================================
# In Python, functions are first-class citizens. They are objects of type `function`.
# They can be assigned to variables, passed as arguments, and returned.

# Lambdas (Anonymous Functions)
# Syntax: lambda arguments: expression
# Limitations: Can only contain a single expression (no statements, no assignments, no `return` keyword).
square = lambda x: x ** 2

def higher_order_example(data: list, transform_func: Callable[[Any], Any]) -> list:
    """A Higher-Order Function takes a function as an arg, or returns one."""
    return [transform_func(item) for item in data]


# ============================================================================
# 4. DECORATORS (DEEP DIVE)
# ============================================================================
# Decorators are simply higher-order functions that take a function and extend 
# its behavior without modifying it explicitly.

# Setting up advanced Type Hinting for Decorators (Python 3.10+)
# ParamSpec allows us to capture the exact parameter types of the decorated function.
# TypeVar allows us to capture the exact return type.
P = ParamSpec('P')
R = TypeVar('R')

def timer_decorator(func: Callable[P, R]) -> Callable[P, R]:
    """A standard decorator."""
    
    @wraps(func) 
    # @wraps is CRUCIAL. It copies the original function's metadata 
    # (__name__, __doc__, etc.) to the wrapper. Without it, debugging is a nightmare.
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        
        # Execute the actual function
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        print(f"[Timer] {func.__name__} executed in {end_time - start_time:.4f}s")
        return result
        
    return wrapper

# Decorators with Arguments
# Requires 3 levels of nested functions: 
# 1. The decorator factory (takes the arguments).
# 2. The actual decorator (takes the function).
# 3. The wrapper (takes the function arguments).
def retry(max_retries: int = 3, delay: float = 0.5):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts}/{max_retries} failed: {e}")
                    time.sleep(delay)
            raise RuntimeError(f"Function {func.__name__} failed after {max_retries} retries.")
        return wrapper
    return decorator

# Class-based Decorators
# Useful when decorators require complex state management.
class RateLimiter:
    def __init__(self, calls_per_second: int):
        self.interval = 1.0 / calls_per_second
        self.last_called = 0.0

    def __call__(self, func: Callable) -> Callable:
        # The __call__ dunder makes the instance callable like a function
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - self.last_called
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_called = time.time()
            return func(*args, **kwargs)
        return wrapper


# Applying decorators (Evaluated bottom-up / closest to the function first)
@timer_decorator
@retry(max_retries=2, delay=0.1)
def unstable_network_request():
    """Simulates a network request that occasionally fails."""
    if time.time() % 2 > 1:
        raise ConnectionError("Network jitter")
    return "Success Data"


# ============================================================================
# 5. GENERATORS AND LAZY EVALUATION (`yield`)
# ============================================================================
# Generators are functions that return an iterator. 
# Instead of returning all items at once (like a list), they yield one item at a time, 
# suspending their state (local variables, instruction pointer) in memory until asked for the next item.

def fibonacci_generator(limit: int) -> Iterator[int]:
    """Generates Fibonacci numbers up to a limit without holding them all in memory."""
    a, b = 0, 1
    while a < limit:
        yield a  # Pauses execution here, returns 'a' to caller
        a, b = b, a + b
        # Upon calling next(), execution resumes exactly here

# `yield from` (Delegating Generators)
# Allows a generator to yield values from another iterable/generator transparently.
def flatten_nested_lists(nested_list):
    """Recursively flattens an arbitrarily nested list."""
    for item in nested_list:
        if isinstance(item, list):
            # Instead of a nested loop: for sub_item in flatten(...): yield sub_item
            # We use `yield from` to delegate yielding to the recursive call.
            yield from flatten_nested_lists(item)
        else:
            yield item

# Generator Expressions (Similar to list comprehensions, but with parentheses)
# Memory efficient: generates items on the fly.
gen_expr = (x ** 2 for x in range(1000000)) # Takes almost 0 bytes of memory


# ============================================================================
# 6. FUNCTOOLS & ADVANCED FUNCTION UTILITIES
# ============================================================================

# functools.partial
# "Freezes" some portion of a function's arguments and/or keywords, 
# resulting in a new object with a simplified signature.
def power(base, exponent):
    return base ** exponent

# Create a new function that always squares its input
square_partial = partial(power, exponent=2)

# functools.lru_cache (or just @cache in Python 3.9+)
# Memoization: Caches the return values of a function based on its arguments.
# Extremely useful for recursive functions or expensive I/O.
@lru_cache(maxsize=128)
def expensive_computation(x: int, y: int) -> int:
    print(f"Computing {x} + {y}...") # Will only print once per unique (x,y)
    return x + y


# ============================================================================
# 7. FUNCTION INTROSPECTION (UNDER THE HOOD)
# ============================================================================
# Functions have dunder attributes that store their internal workings.

def inspect_me(a: int, b: int = 10) -> int:
    """This is a docstring."""
    c = a + b
    return c


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("--- 1. Parameters ---")
    # complex_signature(1, 2, kw_only1="test") # TypeError: standard1 missing
    # complex_signature(1, 2, 3, 4, 5) # TypeError: takes 4 positional, 5 given
    print(complex_signature(1, 2, 3, standard2=4, kw_only1="forced_kw"))
    print(argument_unpacker())
    
    print("\n--- 2. Scope & Closures ---")
    print(f"Before scope demo: GLOBAL_VAR = '{GLOBAL_VAR}'")
    enclosed = scope_demonstrator()
    print(f"After scope demo: GLOBAL_VAR = '{GLOBAL_VAR}'")
    print(f"Enclosed var became: '{enclosed}'")
    
    times_five = make_multiplier(5)
    print(f"Closure multiplier (10 * 5): {times_five(10)}")
    # Inspecting the closure data (highly advanced concept)
    closure_cell = times_five.__closure__[0].cell_contents
    print(f"Value trapped inside the closure object in memory: {closure_cell}")

    print("\n--- 3. First-Class Functions ---")
    data = [1, 2, 3, 4, 5]
    print(f"Mapped via higher-order function: {higher_order_example(data, lambda x: x*10)}")

    print("\n--- 4. Decorators ---")
    try:
        # Note: Because of @wraps, the __name__ is preserved.
        print(f"Calling: {unstable_network_request.__name__}")
        print(unstable_network_request())
    except RuntimeError as e:
        print(e)

    print("\n--- 5. Generators ---")
    fib = fibonacci_generator(50)
    print("Fibonacci generator:", list(fib))
    
    nested = [1, [2, [3, 4], 5], 6]
    print(f"Flattened with 'yield from': {list(flatten_nested_lists(nested))}")

    print("\n--- 6. Functools ---")
    print(f"Partial func (power of 2) on 5: {square_partial(5)}")
    
    print("LRU Cache demonstration:")
    expensive_computation(5, 5) # Prints "Computing..."
    expensive_computation(5, 5) # Does NOT print, pulls from cache
    print("Cache info:", expensive_computation.cache_info())

    print("\n--- 7. Introspection ---")
    print("Docstring:", inspect_me.__doc__)
    print("Default args:", inspect_me.__defaults__) # (10,)
    print("Local variables (varnames):", inspect_me.__code__.co_varnames) # ('a', 'b', 'c')
    
    # The 'inspect' module is safer and cleaner for this than raw dunders
    sig = inspect.signature(complex_signature)
    print("Parsed Signature of complex_signature:")
    for name, param in sig.parameters.items():
        print(f"  {name}: {param.kind}")