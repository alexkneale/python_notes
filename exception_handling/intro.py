# ============================================================================
# COMPREHENSIVE PYTHON EXCEPTION HANDLING TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Instructions: You can run this entire file. It is syntactically valid Python.
# Read through the comments and functions to understand the deep mechanics.
# ============================================================================

import sys
import time
import logging
import traceback
import contextlib
from typing import Any

# Configure basic logging for demonstrations
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ============================================================================
# 1. THE COMPLETE TRY-EXCEPT-ELSE-FINALLY BLOCK
# ============================================================================

def full_exception_lifecycle(numerator: Any, denominator: Any) -> float:
    """
    Demonstrates the full anatomy of exception handling block.
    """
    print(f"\n--- Attempting {numerator} / {denominator} ---")
    result = 0.0
    try:
        # 1. TRY: Put code here that might raise an exception.
        # Keep this block as small as possible to avoid catching unrelated errors.
        result = numerator / denominator
        
    except ZeroDivisionError as e:
        # 2. EXCEPT: Catches specific exceptions. 
        # 'as e' binds the exception instance to the variable 'e'.
        print(f"Failed: Cannot divide by zero. (Details: {e})")
        
    except TypeError as e:
        # You can have multiple except blocks. They are evaluated top-to-bottom.
        print(f"Failed: Invalid types provided. (Details: {e})")
        
    except (ValueError, KeyError) as e:
        # You can catch multiple exceptions in a single block using a tuple.
        print(f"Failed: Value or Key error. (Details: {e})")
        
    else:
        # 3. ELSE: Executes ONLY if the 'try' block succeeds (no exceptions raised).
        # Best practice: Put code here that should only run on success, 
        # but shouldn't be caught by the except blocks above.
        print(f"Success! The result is {result}")
        return result
        
    finally:
        # 4. FINALLY: Executes NO MATTER WHAT (success, handled exception, 
        # or unhandled exception). Even if you 'return' in the try/else block!
        # Primarily used for resource cleanup (closing files, DB connections).
        print("Cleanup: This runs regardless of the outcome.")
        
    return result


# ============================================================================
# 2. EXCEPTION HIERARCHY & ANTI-PATTERNS
# ============================================================================
# Python exceptions are organized in a class hierarchy.
# BaseException
#  ├── SystemExit (Raised by sys.exit())
#  ├── KeyboardInterrupt (Raised by Ctrl+C)
#  └── Exception (Base class for all standard errors)
#       ├── ArithmeticError -> ZeroDivisionError
#       ├── LookupError -> KeyError, IndexError
#       └── ...

def dangerous_bare_except():
    """
    ANTI-PATTERN: Using a bare 'except:' or 'except BaseException:'.
    This catches EVERYTHING, including KeyboardInterrupt (Ctrl+C) and SystemExit,
    making it impossible to shut down your program gracefully.
    """
    try:
        time.sleep(0.1)
    except: # BAD! Never do this.
        pass 

def safer_broad_except():
    """
    BEST PRACTICE (if you must catch everything): Catch 'Exception'.
    This catches all application-level errors, but allows SystemExit/KeyboardInterrupt 
    to bubble up and close the program.
    """
    try:
        1 / 0
    except Exception as e: # Acceptable for top-level loops or logging boundaries
        logging.error(f"An unexpected application error occurred: {e}")


# ============================================================================
# 3. RAISING AND CHAINING EXCEPTIONS (raise from)
# ============================================================================

def fetch_data_from_db(query: str):
    # Simulating a low-level error
    raise ConnectionResetError("TCP socket closed unexpectedly.")

def process_user_request():
    """
    Exception Chaining: When catching a low-level error and raising a high-level 
    custom/domain error, use 'raise ... from ...' to preserve the traceback.
    """
    try:
        fetch_data_from_db("SELECT * FROM users")
    except ConnectionResetError as e:
        # 1. 'raise e' -> Re-raises the exact same exception.
        # 2. 'raise RuntimeError("DB failed")' -> Implicit chaining (__context__).
        #    Python will show "During handling of the above exception, another exception occurred."
        # 3. 'raise ... from e' -> EXPLICIT chaining (__cause__).
        #    Clearly indicates that the new exception was a direct result of the old one.
        
        raise RuntimeError("Failed to process user request due to database failure.") from e


# ============================================================================
# 4. CUSTOM EXCEPTIONS
# ============================================================================
# Creating your own exception classes makes your API much cleaner and easier 
# for other developers to handle programmatically.

class APIClientError(Exception):
    """Base exception for all API client-related errors."""
    pass

class RateLimitExceededError(APIClientError):
    """Raised when the API rate limit is hit."""
    
    def __init__(self, message: str, retry_after: int):
        # Always call the superclass __init__ first
        super().__init__(message)
        # Store custom data on the exception object for the handler to use
        self.retry_after = retry_after
        self.error_code = 429

def make_api_call():
    # Simulating an API rate limit
    raise RateLimitExceededError("You have exceeded your API quota.", retry_after=60)

def handle_custom_exception():
    try:
        make_api_call()
    except RateLimitExceededError as e:
        print(f"API Error {e.error_code}: {e}")
        print(f"-> Please wait {e.retry_after} seconds before retrying.")


# ============================================================================
# 5. ADVANCED TRACEBACK MANIPULATION
# ============================================================================
# Sometimes you want to log the full stack trace WITHOUT crashing the program.

def crashy_function():
    return 1 / 0

def capture_traceback():
    try:
        crashy_function()
    except ZeroDivisionError:
        # sys.exc_info() returns a tuple: (type, value, traceback)
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # traceback.format_exc() returns the standard crash text as a string
        tb_string = traceback.format_exc()
        
        print("We caught an error! Here is the internal traceback string:")
        print("-" * 40)
        print(tb_string.strip())
        print("-" * 40)


# ============================================================================
# 6. PYTHONIC PARADIGM: EAFP vs LBYL
# ============================================================================

def dict_lookup_lbyl(my_dict: dict, key: str):
    """
    LBYL: Look Before You Leap (Common in C/Java).
    Requires multiple lookups, slower, and prone to race conditions (e.g., file existence).
    """
    if key in my_dict:
        return my_dict[key]
    return None

def dict_lookup_eafp(my_dict: dict, key: str):
    """
    EAFP: Easier to Ask for Forgiveness than Permission (Pythonic).
    Faster (happy path has no overhead), avoids race conditions.
    Python relies heavily on Exceptions for standard control flow (e.g., StopIteration in loops).
    """
    try:
        return my_dict[key]
    except KeyError:
        return None


# ============================================================================
# 7. CONTEXTLIB: SUPPRESS
# ============================================================================

def delete_file_if_exists(filepath: str):
    import os
    
    # Old way:
    # try: os.remove(filepath)
    # except FileNotFoundError: pass
    
    # Modern, elegant way to explicitly ignore specific exceptions:
    with contextlib.suppress(FileNotFoundError):
        os.remove(filepath)
        print(f"File {filepath} removed.")


# ============================================================================
# 8. PYTHON 3.11+ FEATURE: EXCEPTION GROUPS & except*
# ============================================================================
# ExceptionGroup allows you to raise and handle MULTIPLE exceptions simultaneously.
# Especially useful in concurrent programming (e.g., asyncio.gather).

def validate_user_data(data: dict):
    errors = []
    
    if not isinstance(data.get("age"), int):
        errors.append(TypeError("'age' must be an integer"))
    if not isinstance(data.get("email"), str):
        errors.append(TypeError("'email' must be a string"))
    if "password" not in data:
        errors.append(ValueError("Missing 'password'"))
        
    if errors:
        # Raise multiple exceptions wrapped in one group
        raise ExceptionGroup("User Data Validation Failed", errors)

def handle_exception_groups():
    try:
        validate_user_data({"age": "twenty", "email": None})
    except* TypeError as e:
        # 'except*' triggers if ONE OR MORE exceptions in the group match the type.
        # 'e' here is an ExceptionGroup containing ONLY the TypeErrors.
        print(f"Caught {len(e.exceptions)} TypeErrors:")
        for err in e.exceptions:
            print(f"  - {err}")
    except* ValueError as e:
        print(f"Caught {len(e.exceptions)} ValueErrors:")
        for err in e.exceptions:
            print(f"  - {err}")


# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (EXCEPTION HANDLING)
# ============================================================================
"""
Q1: When does the `else` block in a `try/except/else/finally` statement execute?
A: The `else` block executes only if the code in the `try` block completes successfully without raising any exceptions.
   This is useful for isolating the code that you want to guard (in the `try` block) from code that runs only if the guarded step succeeded,
   preventing the accidental catching of exceptions raised by subsequent logic.

Q2: What is "Exception Chaining" using `raise ... from ...`, and what is the difference between that and a bare `raise`?
A: 
- `raise NewException from original_exception` sets the `__cause__` of the new exception to the original one.
  This preserves the full traceback history (explicit exception chaining), showing exactly how one error led to another.
- `raise NewException` (implicit chaining) shows both but links them with "During handling of the above exception...".
- Using `raise NewException from None` explicitly suppresses the previous context traceback, making the output cleaner.

Q3: What are Exception Groups (`ExceptionGroup`) and the `except*` syntax introduced in Python 3.11?
A: Exception Groups allow a program to raise and handle multiple unrelated exceptions concurrently.
   This is common in concurrent frameworks (like asyncio or TaskGroups) where multiple tasks can fail at the same time.
   The `except*` syntax allows you to match and handle specific sub-types of exceptions out of an ExceptionGroup individually, 
   unwrapping only the matched errors while leaving the rest to propagate.
"""


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n--- 1. Full Try/Except Lifecycle ---")
    full_exception_lifecycle(10, 2)   # Success path
    full_exception_lifecycle(10, 0)   # ZeroDivisionError path
    full_exception_lifecycle(10, "a") # TypeError path

    print("\n--- 2. Broad Exceptions ---")
    safer_broad_except()

    print("\n--- 3. Exception Chaining ---")
    try:
        process_user_request()
    except RuntimeError as e:
        print(f"Caught High-Level Error: {e}")
        print(f"Original cause (__cause__): {e.__cause__}")

    print("\n--- 4. Custom Exceptions ---")
    handle_custom_exception()

    print("\n--- 5. Traceback Extraction ---")
    capture_traceback()

    print("\n--- 6. EAFP vs LBYL ---")
    data_dict = {"name": "Alice"}
    print(f"EAFP Lookup: {dict_lookup_eafp(data_dict, 'age')} (No crash!)")

    print("\n--- 7. Contextlib Suppress ---")
    print("Attempting to delete a fake file...")
    delete_file_if_exists("fake_ghost_file.txt") # Doesn't crash!

    # Note: ExceptionGroups require Python 3.11 or higher. 
    # If you run this on Python 3.10-, it will raise a SyntaxError due to `except*`.
    if sys.version_info >= (3, 11):
        print("\n--- 8. Exception Groups (Python 3.11+) ---")
        handle_exception_groups()
    else:
        print("\n--- 8. Exception Groups ---")
        print(f"Skipping ExceptionGroup demo. Python 3.11+ required, you are on {sys.version_info.major}.{sys.version_info.minor}")
