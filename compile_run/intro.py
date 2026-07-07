# ============================================================================
# COMPREHENSIVE PYTHON EXECUTION & ERRORS TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Topic: Compile-time vs Runtime errors, Bytecode, and the Python Virtual Machine.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import dis
import ast
import sys
import traceback
from typing import Callable


# ============================================================================
# 1. THE PYTHON EXECUTION PIPELINE (CPYTHON ARCHITECTURE)
# ============================================================================
"""
When you run `python script.py`, Python doesn't execute the source code directly.
Instead, CPython (the standard Python implementation) goes through a specific pipeline:

1. Lexing/Tokenizing: Breaks the raw text into distinct tokens (keywords, symbols).
2. Parsing: Builds an Abstract Syntax Tree (AST) representing the logical structure.
3. Compiling: Converts the AST into a lower-level intermediate representation called Bytecode.
   -> *IF THIS FAILS, YOU GET A COMPILE-TIME ERROR.*
4. Execution: The Python Virtual Machine (PVM) reads the bytecode and executes it step-by-step.
   -> *IF THIS FAILS, YOU GET A RUNTIME ERROR.*

Note: The PVM is a "Stack-Based Virtual Machine". It operates by pushing values onto
a stack, popping them to perform operations, and pushing the result back.
"""


# ============================================================================
# 2. "COMPILE-TIME" ERRORS (Syntax & Indentation)
# ============================================================================
# Python's compiler is relatively simple compared to Java or C++.
# It primarily checks that the code follows structural rules (Syntax) and spacing (Indentation).
# If a file contains a SyntaxError at the very bottom, NO CODE in that file will execute.

def demonstrate_compile_time_errors():
    print("Attempting to compile bad syntax dynamically...")
    
    # We use `compile()` here because if I wrote actual invalid syntax in this 
    # file, the Python interpreter would refuse to run this entire script!
    bad_code_string = """
def bad_function()
    print("I forgot the colon!")
    """
    
    try:
        # compile(source, filename, mode)
        compile(bad_code_string, "<string>", "exec")
    except SyntaxError as e:
        print(f"Caught Compile-Time Error: {type(e).__name__} -> {e}")


# ============================================================================
# 3. RUNTIME ERRORS (Late Binding & Dynamic Execution)
# ============================================================================
# Because Python is dynamically typed, the compiler DOES NOT verify if variables exist, 
# if types are correct, or if methods belong to an object. It just generates bytecode 
# instructions (e.g., "Look up this name when we get here").
# Therefore, almost all logical and type-related errors happen at RUNTIME.

def runtime_error_bypasses_compiler():
    """
    This function compiles perfectly fine into bytecode, even though 'undefined_variable'
    does not exist, and '5 + "hello"' is an illegal operation.
    """
    if False:
        # This branch is never executed by the PVM. 
        # Therefore, the runtime error is never triggered!
        print(undefined_variable)
        x = 5 + "hello"
    return "Compiled and ran successfully because the bad code was skipped!"

def trigger_runtime_error():
    """
    If we actually execute the bad instruction, the PVM throws a Runtime Error.
    """
    try:
        print(this_variable_does_not_exist)
    except NameError as e:
        print(f"Caught Runtime Error: {type(e).__name__} -> {e}")


# ============================================================================
# 4. EXAMINING THE AST (ABSTRACT SYNTAX TREE)
# ============================================================================
# Before bytecode is generated, Python builds an AST. You can manipulate or 
# inspect this tree programmatically (used heavily by linters like Flake8).

def demonstrate_ast():
    source_code = "x = 5 + 3"
    tree = ast.parse(source_code)
    
    print("\nAbstract Syntax Tree for 'x = 5 + 3':")
    # dump() gives us a text representation of the nodes
    print(ast.dump(tree, indent=2))
    """
    Output looks roughly like:
    Module(
      body=[
        Assign(
          targets=[Name(id='x', ctx=Store())],
          value=BinOp(
            left=Constant(value=5),
            op=Add(),
            right=Constant(value=3)))])
    """


# ============================================================================
# 5. DEEP DIVE: DISASSEMBLING BYTECODE WITH `dis`
# ============================================================================
# We can use the standard library `dis` module to look at the exact PVM 
# instructions Python generated for our functions.

def math_operation(a, b):
    result = a + (b * 10)
    return result

def examine_bytecode():
    print("\nBytecode Disassembly of 'math_operation':")
    print("-" * 50)
    dis.dis(math_operation)
    print("-" * 50)
    
    """
    Explanation of the disassembly you will see in the console:
    
    LOAD_FAST (a): Pushes local variable 'a' onto the stack.
    LOAD_FAST (b): Pushes local variable 'b' onto the stack.
    LOAD_CONST (10): Pushes the constant integer 10 onto the stack.
    BINARY_MULTIPLY: Pops 10 and 'b', multiplies them, pushes result to stack.
    BINARY_ADD: Pops the mult-result and 'a', adds them, pushes result to stack.
    STORE_FAST (result): Pops the add-result and stores it in local var 'result'.
    LOAD_FAST (result): Pushes 'result' back onto the stack.
    RETURN_VALUE: Pops the top of the stack and returns it to the caller.
    
    Notice how the compiler doesn't care if 'a' and 'b' are integers or strings.
    It just generates BINARY_ADD. The actual `__add__` dunder method is dynamically 
    resolved by the PVM AT RUNTIME.
    """


# ============================================================================
# 6. CONSTANT FOLDING (PEEPHOLE/AST OPTIMIZATION)
# ============================================================================
# While Python doesn't do heavy ahead-of-time (AOT) compilation like C, 
# it DOES perform minor optimizations during the compile step.

def constant_folding_demo():
    # Python calculates this AT COMPILE TIME. 
    # It will not perform the multiplication at runtime.
    seconds_in_day = 60 * 60 * 24 
    return seconds_in_day

def examine_constant_folding():
    print("\nNotice how 60 * 60 * 24 is reduced to a single LOAD_CONST (86400):")
    dis.dis(constant_folding_demo)


# ============================================================================
# 7. THE LATE BINDING GOTCHA (Closures & Loops)
# ============================================================================
# A classic bug resulting from runtime evaluation vs compile-time evaluation.

def late_binding_issue():
    """
    Goal: Create 3 functions that return 0, 1, and 2 respectively.
    """
    funcs = []
    for i in range(3):
        # The lambda captures the VARIABLE 'i', not the VALUE of 'i' at this exact moment.
        funcs.append(lambda: i)
        
    # At runtime, when we call f(), it looks up the current value of 'i' in the 
    # enclosing scope. Since the loop finished, 'i' is now 2 for ALL functions!
    results = [f() for f in funcs]
    return results

def late_binding_fix():
    funcs = []
    for i in range(3):
        # Fix: We force early binding (at definition/compile time) by passing 
        # 'i' as a default argument. Default arguments are evaluated exactly ONCE, 
        # when the function definition is executed/compiled.
        funcs.append(lambda i=i: i)
        
    results = [f() for f in funcs]
    return results


# ============================================================================
# 8. SHIFTING RUNTIME ERRORS TO "COMPILE TIME" (MyPy / Type Checking)
# ============================================================================
"""
Because Python compilers (like CPython) do not enforce types, modern Python 
development relies on STATIC TYPE CHECKING (e.g., MyPy, Pyright).

These tools run an entirely separate pass over your code BEFORE execution 
(acting like a pseudo-compile-time check).

Example:
```python
def uppercase(text: str) -> str:
    return text.upper()

uppercase(5) # Python Compiler: "Looks good to me!" -> Generates Bytecode
             # MyPy (Static Analysis): "ERROR: Argument 1 has incompatible type int; expected str"
             # Python PVM (Runtime): "AttributeError: 'int' object has no attribute 'upper'"
"""




# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (COMPILE & RUN)
# ============================================================================
"""
Q1: Is Python an interpreted language or a compiled language?
A: Both. CPython first parses and compiles Python source code (.py) into an intermediate,
   lower-level representation called Bytecode (stored as .pyc files). 
   Then, the Python Virtual Machine (PVM)—a stack-based interpreter—reads and executes those bytecode instructions.

Q2: What is "Late Binding" in Python closures, and why does it catch developers off guard?
A: Python's closures bind variables by reference/name, not by value. The lookup of enclosing-scope variables 
   happens when the function is *called*, not when it is *defined*.
   In a loop creating lambda functions, all lambdas will reference the same final value of the loop variable.
   To fix this, we can force early binding using a default parameter value (e.g., `lambda x=x: x`).

Q3: Why doesn't a runtime error (like referencing an undefined variable) stop a Python file from compiling?
A: CPython's compiler primarily checks structural syntax and indentation. It does not perform static analysis 
   to check if variables exist, if types match, or if methods belong to objects. Since these names are resolved 
   at runtime via dynamic lookups, any invalid reference will compile perfectly but raise a NameError/AttributeError at runtime.
"""


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================
"""
The if __name__ == "__main__": block ensures specific code only runs when you execute a script directly, not when you import
it as a module into another file. This allows a file to serve as a reusable library while still acting as a standalone executable.

When you run a Python file directly, Python sets a special internal variable called __name__ to the string "__main__". If you import 
that file into a different Python script, __name__ is set to the file's actual module/file name instead
"""
if __name__ == "__main__":
    print("--- 1 & 2. Compile-Time Errors ---")
    demonstrate_compile_time_errors()

    print("\n--- 3. Runtime Errors (Dynamic Typing) ---")
    print(runtime_error_bypasses_compiler())
    trigger_runtime_error()

    print("\n--- 4. Examining the AST ---")
    demonstrate_ast()

    print("\n--- 5. Disassembling Bytecode (PVM Stack) ---")
    examine_bytecode()

    print("\n--- 6. Constant Folding (Compiler Optimization) ---")
    examine_constant_folding()

    print("\n--- 7. Late Binding (Runtime Lookup Gotcha) ---")
    print(f"Broken Late Binding (Expected [0, 1, 2]): {late_binding_issue()}")
    print(f"Fixed Default Binding (Expected [0, 1, 2]): {late_binding_fix()}")

    print("\n--- Tutorial Complete ---")
    print("Key takeaway: Python trusts you at compile time, and verifies at runtime.")
