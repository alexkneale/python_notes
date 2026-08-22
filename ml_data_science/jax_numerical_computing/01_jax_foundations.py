# ============================================================================
# JAX FOUNDATIONS: FUNCTIONAL PROGRAMMING, ASYNC DISPATCH, AND PRNG MECHANICS
# ============================================================================
# Target Audience: Senior ML Engineers, Quantitative Developers, Researchers.
# Topic: Asynchronous Dispatch, Pure Functions, Immutability, and Stateless PRNG.
# Compatibility: Python 3.10+, JAX 0.4+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is JAX?
   - JAX is a library for high-performance numerical computing. It has an API that is 
     almost identical to NumPy, but it can run on GPUs and TPUs, and features automatic 
     differentiation, compilation, and vectorization built-in.
2. What is a Pure Function?
   - JAX requires functions to be mathematically "pure". A pure function:
     - Always returns the same output for the same input arguments.
     - Has no "side effects" (it does not modify global variables, write to files, 
       print to the screen, or change external states).
3. Why is State Immutability important?
   - In JAX, arrays are immutable (cannot be changed in-place). Instead of modifying an 
     array, JAX functions return a new array with the modified values. This allows the 
     underlying compiler to safely optimize execution trees and eliminate redundant copies.
"""



import jax
import jax.numpy as jnp
import numpy as np
import time

# ============================================================================
# 1. JAX VS NUMPY & ASYNCHRONOUS DISPATCH
# ============================================================================
"""
JAX matches NumPy's API closely, but its execution model is fundamentally different:
1. DeviceArray / Array: JAX arrays are backed by memory buffers on an accelerator 
   (GPU, TPU, or CPU).
2. Asynchronous Dispatch: JAX operations are pushed to a device execution queue and 
   returned *before* the hardware actually completes the computation. JAX returns a 
   Future-like placeholder.
   - This prevents Python from bottlenecking the accelerator.
   - If you need to benchmark or force execution completion, you must call `.block_until_ready()`.
"""

def demonstrate_async_dispatch():
    print("\n" + "="*80)
    print("1. JAX VS NUMPY: ASYNCHRONOUS DISPATCH BENCHMARK")
    print("="*80)

    size = 4000
    
    # NumPy (Synchronous on CPU)
    t0 = time.perf_counter()
    x_np = np.random.randn(size, size).astype(np.float32)
    y_np = np.dot(x_np, x_np)
    t_np = time.perf_counter() - t0
    print(f"NumPy dot product time:  {t_np:.5f} seconds")

    # JAX (Asynchronous)
    key = jax.random.PRNGKey(42)
    x_jax = jax.random.normal(key, (size, size))
    
    # 1. Without blocking (Asynchronous dispatch - returns immediately!)
    t0 = time.perf_counter()
    y_jax_async = jnp.dot(x_jax, x_jax)
    t_jax_async = time.perf_counter() - t0
    print(f"JAX dot (Asynchronous):  {t_jax_async:.5f} seconds <-- Insanely fast, but misleading!")

    # 2. Correct JAX Benchmark: Blocking until device completes computation
    t0 = time.perf_counter()
    y_jax_sync = jnp.dot(x_jax, x_jax).block_until_ready()
    t_jax_sync = time.perf_counter() - t0
    print(f"JAX dot (Synchronized):  {t_jax_sync:.5f} seconds <-- Real execution time on device.")


# ============================================================================
# 2. THE PURITY REQUIREMENT: PURE FUNCTIONS ONLY
# ============================================================================
"""
JAX relies on compiling and transforming Python code. To trace code safely, 
JAX requires that compiled functions are PURE:
A function is PURE if:
1. It is deterministic: Given the same inputs, it always returns the same outputs.
2. It has no side effects: It does not modify global state, write to files, 
   print, or modify mutable arguments in-place.

If you compile an impure function with JAX (e.g. using `jax.jit`), the compiler will 
behave unpredictably because side-effects only run during *compilation tracing*, 
not during actual execution of the compiled binary!
"""

# GLOBAL STATE VIOLATION
global_variable = 10.0

def impure_function(x):
    # This is impure because it depends on an external mutable global variable
    return x + global_variable

@jax.jit
def pure_version(x, constant):
    # Pass all dependencies explicitly as arguments to maintain purity!
    return x + constant

def demonstrate_purity_issues():
    print("\n" + "="*80)
    print("2. JAX PURITY REQUIREMENT & COMPILER TRACING")
    print("="*80)

    global global_variable
    global_variable = 10.0

    # Compiling an impure function
    jit_impure = jax.jit(impure_function)

    # First call: Tracing happens here. global_variable=10.0 is burned into the compiled code!
    out1 = jit_impure(5.0)
    print(f"First call (global=10.0, input=5.0): {out1}")

    # Modify the global variable
    global_variable = 100.0

    # Second call: The compiled binary runs! It does NOT re-run Python code, 
    # so it still uses the burned-in value of 10.0!
    out2 = jit_impure(5.0)
    print(f"Second call (global=100.0, input=5.0): {out2} <-- ERROR: State change ignored!")

    # Correct pure design
    jit_pure = jax.jit(pure_version)
    print(f"Pure function call (constant=100.0): {jit_pure(5.0, 100.0)}")


# ============================================================================
# 3. IMMUTABILITY AND FUNCTIONAL UPDATE SYNTAX
# ============================================================================
"""
Because of pure-functional constraints, JAX Arrays are strictly IMMUTABLE.
You cannot write: `x[0] = 9.0` (raises TypeError).
Instead, JAX provides the `.at` syntax to express in-place modifications functionally:
    `y = x.at[index].set(value)`
This returns a *new* array representing the updated array, while internally (using XLA) 
optimizing it to do an in-place modification on the device buffer if no other reference exists!
"""

def demonstrate_immutability():
    print("\n" + "="*80)
    print("3. IMMUTABILITY & THE FUNCTIONAL UPDATE (.at) SYNTAX")
    print("="*80)

    x = jnp.array([1.0, 2.0, 3.0])
    print(f"Original array x: {x}")

    try:
        x[0] = 99.0
    except TypeError as e:
        print(f"Caught expected error: Arrays are immutable in JAX!\n  Error: {e}")

    # Functional update: Create a copy containing the update
    y = x.at[0].set(99.0)
    print(f"Modified copy y:  {y}")
    print(f"Original x remains unchanged: {x}")

    # Multi-dimensional indexing updates are also supported
    matrix = jnp.zeros((3, 3))
    updated_matrix = matrix.at[0, 1:3].add(5.0)
    print(f"Matrix updated at [0, 1:3] with .add():\n{updated_matrix}")


# ============================================================================
# 4. STATELESS PSUDO-RANDOM NUMBER GENERATION (PRNG)
# ============================================================================
"""
NumPy tracks the state of its random number generator globally (stateful PRNG).
If you run random functions in different threads, they modify the same global state, 
violating purity and reproducibility under concurrent execution.

JAX implements a STATELESS PRNG system (Threefry counter-based PRNG):
1. You must explicitly create a starting state: a `PRNGKey`.
2. Every random operation requires a `key` and is completely deterministic given that key.
3. If you need multiple random numbers, you must explicitly SPLIT keys using `jax.random.split`.
   - Never reuse the same key for multiple random functions; doing so will result in 
     identical or correlated outputs, introducing severe data leakage / bugs!
"""

def demonstrate_stateless_prng():
    print("\n" + "="*80)
    print("4. JAX STATELESS PRNG SYSTEM")
    print("="*80)

    # 1. Create seed key
    key = jax.random.PRNGKey(1337)
    print(f"Initial key: {key}")

    # 2. Draw random normal vectors
    # Doing this twice with the same key yields identical outputs (Deterministic!)
    r1 = jax.random.normal(key, (3,))
    r2 = jax.random.normal(key, (3,))
    print(f"r1 using key: {r1}")
    print(f"r2 using key: {r2} <-- Identical! This is dangerous if unexpected.")

    # 3. Correct way: Split key!
    # jax.random.split(key, num_keys) returns an array of keys
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    r_sub1 = jax.random.normal(subkey1, (3,))
    r_sub2 = jax.random.normal(subkey2, (3,))
    print(f"r_sub1 (subkey1): {r_sub1}")
    print(f"r_sub2 (subkey2): {r_sub2} <-- Properly distinct and uncorrelated!")
    
    # Print updated parent key for future splits
    print(f"Updated parent key: {key}")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Why does JAX reject stateful global random states (like numpy's np.random.seed)?
   - Answer: Stateful random generators modify a hidden global state with each call, 
     making them "impure" functions. This violates JAX's pure-functional paradigm, 
     rendering tracing/compilation non-deterministic and thread-unsafe across 
     distributed devices. JAX uses stateless, explicit Threefry PRNG keys.

2. MEDIUM: What is JAX's Asynchronous Dispatch model, and why is it beneficial?
   - Answer: JAX does not wait for computations to finish on the GPU/TPU before 
     returning control to Python; instead, it returns an Array placeholder immediately 
     while queueing the math on the accelerator. This prevents Python's interpreter overhead 
     from throttling fast hardware devices. To get actual synchronous timings, 
     you must use `.block_until_ready()`.

3. HARD: Explain how JAX "traces" a function to build a Jaxpr.
   - Answer: When `jax.jit` or other transformations wrap a function, JAX runs 
     the function once on symbolic placeholders called "Tracers" (representing 
     the abstract shapes and datatypes, e.g., `ShapedArray(float32[10])`) instead of 
     concrete numeric values. JAX intercepts all JAX-numpy calculations performed on these 
     tracers, building a flattened, language-agnostic intermediate representation called 
     a JAX Expression (Jaxpr). XLA then compiles this Jaxpr into machine code.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a functional threshold-clipping operator `functional_clip(arr, low, high)` 
   in JAX that clips values outside `[low, high]` without using `jnp.clip` or in-place assignment.
   - Code Template to execute:
"""

def functional_clip(arr: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
    # We implement clipping purely functionally using element-wise logical select
    # Or simply: jnp.where(arr < low, low, jnp.where(arr > high, high, arr))
    clipped_low = jnp.where(arr < low, low, arr)
    return jnp.where(clipped_low > high, high, clipped_low)


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_async_dispatch()
    demonstrate_purity_issues()
    demonstrate_immutability()
    demonstrate_stateless_prng()
    
    # Testing the functional clip
    print("\nTesting Custom Functional Clip:")
    arr = jnp.array([-5.0, 0.0, 5.0, 15.0])
    clipped = functional_clip(arr, 0.0, 10.0)
    print(f"  Input:   {arr.tolist()}")
    print(f"  Clipped: {clipped.tolist()} (Expected: [0.0, 0.0, 5.0, 10.0])")