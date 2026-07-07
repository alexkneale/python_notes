# ============================================================================
# COMPREHENSIVE PYTHON TESTING & MOCKING TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers preparing for Mid-to-Senior level roles.
# Topic: Mock vs MagicMock, The "Where to Patch" Gotcha, Mocking Side Effects,
#        Mocking Context Managers, AsyncMock, and Python's unittest.mock.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import sys
import asyncio

# ============================================================================
# 1. MOCK vs MAGICMOCK
# ============================================================================
"""
`unittest.mock` provides two main classes: `Mock` and `MagicMock`.

- `Mock`: A basic mock object. It creates attributes and methods dynamically on first access.
  However, it does NOT implement magic/dunder methods by default (e.g., trying to iterate 
  over a Mock, or calling len() on it, will raise a TypeError).

- `MagicMock`: A subclass of Mock that implements default implementations of almost 
  every magic (dunder) method (like __len__, __iter__, __str__, __getitem__, __enter__, etc.).
  This is the default mock type used by `@patch`.
"""

def demonstrate_mock_vs_magicmock():
    print("\n--- 1. Mock vs MagicMock ---")
    
    mock_obj = Mock()
    magic_mock_obj = MagicMock()
    
    # Both can mock standard attributes & methods
    mock_obj.some_method.return_value = 42
    print(f"Mock method return: {mock_obj.some_method()}")
    
    # --- The Magic/Dunder Difference ---
    # 1. Calling len()
    try:
        len(mock_obj)
    except TypeError as e:
        print(f"Calling len() on Mock failed as expected: {e}")
        
    print(f"Calling len() on MagicMock succeeded: {len(magic_mock_obj)} (Default: 0)")
    
    # 2. Iteration
    try:
        for x in mock_obj:
            pass
    except TypeError as e:
        print(f"Iterating over Mock failed as expected: {e}")
        
    magic_mock_obj.__iter__.return_value = [1, 2, 3]
    print(f"Iterating over MagicMock: {list(magic_mock_obj)}")


# ============================================================================
# 2. THE CRITICAL "WHERE TO PATCH" GOTCHA
# ============================================================================
"""
This is the single most common testing error in Python.
The Rule: PATCH THE NAME WHERE IT IS LOOKED UP (IMPORTED), NOT WHERE IT IS DEFINED!

Scenario:
We have a module `services.py` that imports `requests` to make a call.
```python
# services.py
import requests

def fetch_data():
    return requests.get("https://api.com").json()
```

If we want to mock `requests.get`:
- DO NOT patch `'requests.get'`.
- DO patch `'services.requests.get'`.

Why? Because `services.py` has loaded its own reference to `requests` inside its namespace.
If you patch `'requests.get'`, you modify the global `requests` module, but `services.py` 
keeps using its already-imported reference!
"""


# ============================================================================
# 3. SIDE EFFECTS AND EXCEPTIONS
# ============================================================================
"""
Sometimes you don't want a mock to return a static value. You want it to:
1. Raise an exception (to test error handling).
2. Return different values on consecutive calls.
3. Delegate to a dynamic helper function.

We achieve this using the `side_effect` attribute of a Mock.
"""

def demonstrate_side_effects():
    print("\n--- 3. Side Effects and Exceptions ---")
    
    mock_api = Mock()
    
    # --- 1. Raising an Exception ---
    mock_api.get_user.side_effect = ValueError("User not found in database")
    
    try:
        mock_api.get_user(id=99)
    except ValueError as e:
        print(f"Successfully mocked exception raising: {e}")
        
    # --- 2. Multiple Consecutive Return Values ---
    # Pass an iterable (like a list) to return those values in order on subsequent calls
    mock_api.get_status.side_effect = ["Pending", "Processing", "Completed"]
    
    print(f"Call 1 status: {mock_api.get_status()}")
    print(f"Call 2 status: {mock_api.get_status()}")
    print(f"Call 3 status: {mock_api.get_status()}")
    
    # --- 3. Dynamic Handler Function ---
    def dynamic_response(x):
        return x * 10
        
    mock_api.calculate.side_effect = dynamic_response
    print(f"Call with 5: {mock_api.calculate(5)}")
    print(f"Call with 12: {mock_api.calculate(12)}")


# ============================================================================
# 4. MOCKING CONTEXT MANAGERS
# ============================================================================
"""
Mocking code that uses `with` statements (like open() or database transactions)
can be tricky because you must mock the __enter__ and __exit__ magic methods.

Since MagicMock implements context manager protocols automatically:
- Mocking a context manager is as simple as configuring `mock.__enter__.return_value`.
"""

class DBTransactor:
    def __enter__(self):
        return "Real DB Connection"
    def __exit__(self, exc_type, exc_val, tb):
        pass

def code_under_test(transactor):
    with transactor as conn:
        print(f"  [Code Under Test] Executing query on: {conn}")
        return "Query Done!"

def demonstrate_mock_context_manager():
    print("\n--- 4. Mocking Context Managers ---")
    
    mock_transactor = MagicMock()
    # Configure the value returned inside the 'with' block
    mock_transactor.__enter__.return_value = "Mocked Test Connection"
    
    result = code_under_test(mock_transactor)
    print(f"Code returned: {result}")
    
    # Verify the context manager was entered and exited
    mock_transactor.__enter__.assert_called_once()
    mock_transactor.__exit__.assert_called_once()


# ============================================================================
# 5. MOCKING ASYNC FUNCTIONS (ASYNCMOCK)
# ============================================================================
"""
In modern Python (3.8+), you cannot use a regular `Mock` or `MagicMock` to mock 
an async function directly because awaiting a standard mock doesn't return a coroutine.

We must use `AsyncMock`, which behaves like a standard mock but returns an 
awaitable coroutine when called.
"""

async def fetch_async_data():
    await asyncio.sleep(0.1)
    return "Real Web Value"

async def async_code_under_test():
    # Imagine this function calls 'fetch_async_data'
    data = await fetch_async_data()
    return f"Processed: {data}"

async def demonstrate_async_mocking():
    print("\n--- 5. Mocking Async Functions ---")
    
    # We patch the function inside this module (__main__)
    with patch('__main__.fetch_async_data', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = "Mocked Async Web Value"
        
        # Call the code under test
        result = await async_code_under_test()
        print(f"Async test returned: {result}")
        
        # Verify the mock was called
        mock_fetch.assert_awaited_once()


# ============================================================================
# RUNNABLE UNIT TESTS DEMO (USING UNITTEST FRAMEWORK)
# ============================================================================

# Code representing what we want to test:
class InternalService:
    def execute_api_call(self):
        # Imagine this makes an actual HTTP call
        return "Real Network Data"

class ApplicationFlow:
    def __init__(self, service):
        self.service = service
        
    def run_flow(self):
        data = self.service.execute_api_call()
        if data == "Real Network Data":
            return "Production Success"
        return f"Mock Success: {data}"

# The Test Case:
class TestApplicationFlow(unittest.TestCase):
    
    def test_flow_with_mocked_service(self):
        # 1. Create the mock
        mock_service = Mock()
        mock_service.execute_api_call.return_value = "Stubbed API Data"
        
        # 2. Inject mock dependency
        flow = ApplicationFlow(mock_service)
        
        # 3. Call method and assert results
        result = flow.run_flow()
        self.assertEqual(result, "Mock Success: Stubbed API Data")
        
        # 4. Verify call expectations
        mock_service.execute_api_call.assert_called_once()


def run_unittest_suite():
    print("\n--- Running Unittest Suite Demonstration ---")
    # Load and run the specific test case defined above
    suite = unittest.TestLoader().loadTestsFromTestCase(TestApplicationFlow)
    unittest.TextTestRunner(verbosity=1).run(suite)
# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (TESTING & MOCKING)
# ============================================================================
"""
Q1: What is the difference between `Mock` and `MagicMock` in Python's `unittest.mock`?
A: 
- `Mock` is a basic object that allows you to mock attributes and method calls, returning a new Mock for any undefined attributes.
- `MagicMock` is a subclass of `Mock` that comes pre-configured with support for Python's magic (dunder) methods 
  (e.g., `__len__`, `__iter__`, `__str__`, `__enter__`, `__exit__`).
  If you need to mock an object that will be used in a `with` statement, a `for` loop, or has its length checked, use `MagicMock`.

Q2: What is the difference between `return_value` and `side_effect` on a Mock?
A: 
- `return_value` defines a static value that is returned every single time the mocked method is called.
- `side_effect` allows you to define a dynamic behavior. It can accept:
  1. An exception class or instance (which will be raised when the mock is called).
  2. An iterable (where each call to the mock returns the next item in the iterable).
  3. A custom callable/function (which receives the arguments passed to the mock and executes custom logic).

Q3: What does the `@patch` decorator do, and why is target string resolution important?
A: `@patch` intercepts imports of a specific module or class and replaces it with a Mock during the test run.
   A key gotcha is "mocking where the class is looked up, not where it is defined."
   If `module_a` imports `from module_b import Client`, and you patch `module_b.Client`, 
   `module_a`'s reference is unchanged. You must instead patch the target location: `module_a.Client`.
"""


if __name__ == "__main__":
    demonstrate_mock_vs_magicmock()
    demonstrate_side_effects()
    demonstrate_mock_context_manager()
    
    # Run async mocking
    asyncio.run(demonstrate_async_mocking())
    
    # Run the standard unittest suite
    run_unittest_suite()
    
    print("\n--- Tutorial Complete ---")
    print("Key takeaway: Mock for simple targets, MagicMock for protocol support.")
    print("Remember to patch where the module is imported, and use AsyncMock for async functions.")
