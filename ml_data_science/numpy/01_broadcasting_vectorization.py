# ============================================================================
# NUMPY NUMERICAL INTERNALS: BROADCASTING, MEMORY LAYOUTS, AND VIEWS
# ============================================================================
# Target Audience: Applied Mathematicians, Quantitative Developers, AI Engineers.
# Topic: Broadcasting Axioms, C-contiguous vs. Fortran Memory layouts, 
#        CPU Cache Locality, Slicing Views vs. Advanced Copies, and ufuncs.
# Compatibility: Python 3.10+, NumPy 1.22+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is NumPy?
   - NumPy is the foundational library for scientific computing in Python. It provides 
     extremely fast n-dimensional array objects (`ndarray`) and mathematical utilities 
     implemented in optimized C.
2. What is Vectorization?
   - Standard Python lists require looping through each element (e.g. using a `for` loop). 
     This is slow. NumPy operates on whole arrays at once (vectorized operations), executing 
     calculations in parallel compiled C loops, which is orders of magnitude faster.
3. What is Broadcasting?
   - Broadcasting is NumPy's ability to perform arithmetic operations on arrays of different 
     shapes. For example, adding a scalar number (value 1) to a 1D array of shape (5,) 
     "broadcasts" the addition across all five elements automatically without copying data.
"""



import numpy as np
import time

# ============================================================================
# 1. THE MATHEMATICAL LAWS OF BROADCASTING
# ============================================================================
"""
NumPy's broadcasting allows arithmetic operations on arrays of different shapes.
The Broadcasting Rules are strictly evaluated from Right-to-Left (trailing dimensions):

1. Dimension Padding: If two arrays differ in their number of dimensions, the shape 
   of the array with fewer dimensions is padded with ones on its left (leading) side.
   - Example: Array A of shape (3, 4) and Array B of shape (4,). 
     Array B is padded on the left to shape (1, 4).
2. Shape Compatibility: Two dimensions are compatible if:
   - They are equal in size.
   - OR one of the dimensions is exactly 1.
3. Shape Stretching: If a dimension is 1, that dimension is stretched (replicated) 
   conceptually to match the other array's dimension size.
4. Error Case: If dimensions do not match and neither is 1, a ValueError is raised.
"""

def demonstrate_broadcasting():
    print("\n" + "="*80)
    print("1. THE MATHEMATICAL LAWS OF BROADCASTING")
    print("="*80)

    # Array A: Shape (3, 4)
    A = np.array([[10, 20, 30, 40],
                  [50, 60, 70, 80],
                  [90, 100, 110, 120]])
    
    # Array B: Shape (4,) -> Right-to-Left matches, padded to (1, 4) and stretched
    B = np.array([1, 2, 3, 4])
    
    C = A + B
    print(f"Array A shape: {A.shape}")
    print(f"Array B shape: {B.shape}")
    print(f"Result (A + B) shape: {C.shape}")
    print(f"Resulting array:\n{C}")

    # Example of incompatible broadcasting
    # D: Shape (3,)
    D = np.array([1, 2, 3])
    try:
        _ = A + D
    except ValueError as e:
        print(f"\nExpected Broadcasting Error: shape (3, 4) and shape (3,) do not match!\n  Error: {e}")


# ============================================================================
# 2. MEMORY LAYOUTS & CPU CACHE LOCALITY (C vs FORTRAN ORDER)
# ============================================================================
"""
NumPy arrays are stored as a flat block of sequential memory.
How we map multi-dimensional index coordinates (row, col) to flat memory matters:

1. C-Order (Row-Major): Row elements are adjacent in flat memory (Default in NumPy).
   - Moving from index (0, 0) to (0, 1) jumps 1 element.
2. Fortran-Order (Column-Major): Column elements are adjacent in flat memory.
   - Moving from (0, 0) to (1, 0) jumps 1 element.

CPU Cache Locality:
CPUs read memory in 'cache lines' (usually 64 bytes). When you access an element, 
the CPU fetches the entire cache line containing adjacent elements.
- Summing a C-order array along rows (axis=1) accesses adjacent memory. This is 
  cache-friendly, resulting in rapid cache hits!
- Summing a C-order array along columns (axis=0) jumps across rows. If the jump size 
  is larger than the cache line, it causes cache line-misses, throttling speed!
"""

def benchmark_cache_locality():
    print("\n" + "="*80)
    print("2. MEMORY LAYOUTS & CPU CACHE LOCALITY BENCHMARK")
    print("="*80)

    size = 10000
    
    # Create C-contiguous (Row-Major) and Fortran-contiguous (Column-Major) matrices
    arr_c = np.ones((size, size), order='C') # Row-major
    arr_f = np.ones((size, size), order='F') # Column-major

    print(f"Matrix Size: {size}x{size} floats")
    print(f"Is arr_c C-contiguous? {arr_c.flags['C_CONTIGUOUS']}")
    print(f"Is arr_f F-contiguous? {arr_f.flags['F_CONTIGUOUS']}")

    # 1. Sum along columns (axis=0)
    # For C-order: this jumps rows (bad cache locality).
    # For F-order: columns are sequential (great cache locality!).
    t0 = time.perf_counter()
    _ = arr_c.sum(axis=0)
    t_c_axis0 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = arr_f.sum(axis=0)
    t_f_axis0 = time.perf_counter() - t0

    print(f"\nSumming along columns (axis=0):")
    print(f"  C-Contiguous time: {t_c_axis0:.5f} seconds")
    print(f"  F-Contiguous time: {t_f_axis0:.5f} seconds (Cache Hit Optimal!)")
    print(f"  Performance Ratio: {t_c_axis0 / t_f_axis0:.2f}x faster on F-order!")

    # 2. Sum along rows (axis=1)
    # For C-order: rows are sequential (great cache locality!).
    # For F-order: columns are sequential (bad cache locality).
    t0 = time.perf_counter()
    _ = arr_c.sum(axis=1)
    t_c_axis1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = arr_f.sum(axis=1)
    t_f_axis1 = time.perf_counter() - t0

    print(f"\nSumming along rows (axis=1):")
    print(f"  C-Contiguous time: {t_c_axis1:.5f} seconds (Cache Hit Optimal!)")
    print(f"  F-Contiguous time: {t_f_axis1:.5f} seconds")
    print(f"  Performance Ratio: {t_f_axis1 / t_c_axis1:.2f}x faster on C-order!")


# ============================================================================
# 3. SLICING VIEWS VS ADVANCED INDEXING COPIES
# ============================================================================
"""
Memory allocation is expensive. NumPy avoids allocations whenever possible:

1. Basic Slicing (e.g. `y = x[0:5]`): Returns a VIEW. 
   - It shares the exact same underlying raw memory block as `x`.
   - Modifying `y` will directly modify `x`! This is O(1) in time and memory.
2. Advanced Indexing (e.g. `y = x[[0, 1, 3]]` or `y = x[x > 0]`): Returns a COPY.
   - It allocates a completely new memory segment. 
   - Modifying `y` does NOT modify `x`.
   - You can verify if two arrays share memory using `np.shares_memory(a, b)`.
"""

def demonstrate_views_vs_copies():
    print("\n" + "="*80)
    print("3. SLICING VIEWS VS. ADVANCED INDEXING COPIES")
    print("="*80)

    x = np.array([10, 20, 30, 40, 50])
    print(f"Original x: {x}")

    # 1. Basic Slicing View
    view_slice = x[1:4] # [20, 30, 40]
    print(f"  Does view_slice share memory with x? {np.shares_memory(x, view_slice)}")
    
    # Mutate view
    view_slice[0] = 999
    print(f"  After modifying slice, original x changed: {x}")

    # Reset x
    x[1] = 20

    # 2. Advanced Indexing Copy
    copy_indexing = x[[0, 2, 4]] # [10, 30, 50]
    print(f"  Does copy_indexing share memory with x? {np.shares_memory(x, copy_indexing)}")
    
    # Mutate copy
    copy_indexing[0] = 999
    print(f"  After modifying copy, original x remains unchanged: {x}")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What are the broadcasting rules in NumPy?
   - Answer: Dimension evaluation starts from the trailing (rightmost) dimension. 
     (1) If sizes are unequal and one is 1, it stretches to match. (2) If sizes are 
     unequal and neither is 1, broadcasting fails. (3) If array dimensions differ, 
     the smaller dimension array is prepended with 1s on the left to align.

2. MEDIUM: What is the performance pitfall of `np.vectorize`?
   - Answer: `np.vectorize` is a convenience function to support element-wise operations 
     on numpy inputs, but it does NOT optimize execution. It is a simple wrapper 
     implementing a standard Python `for` loop behind the scenes. It performs 
     standard Python interpreter evaluations and object conversions, lacking any of the 
     speed and compilation optimizations of compiled C universal functions (`ufuncs`).

3. HARD: Explain how memory strides change when you transpose a 2D C-order matrix.
   - Answer: Strides dictate how many bytes to skip in flat memory to move 1 step 
     along an axis. For a float64 (8 bytes) C-ordered array of shape (M, N), the strides 
     are (N*8, 8). Transposing swaps the strides to (8, N*8) and swaps the dimensions. 
     No actual memory copy or data relocation takes place; only the metadata representation 
     is updated in O(1) time.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `manual_z_score_normalization(matrix)` that performs 
   column-wise Z-score normalization on a 2D NumPy array:
       Z = (X - mu) / sigma
   Your implementation must leverage broadcasting to avoid Python loops.
   - Code Template to execute:
"""

def manual_z_score_normalization(matrix: np.ndarray) -> np.ndarray:
    # 1. Calculate column-wise mean and std (axis=0)
    # To support proper broadcasting division, keep dimensions via keepdims=True!
    # This keeps mean as shape (1, cols) instead of (cols,)
    mean = np.mean(matrix, axis=0, keepdims=True)
    std = np.std(matrix, axis=0, keepdims=True)
    
    # Avoid Division by Zero by replacing 0.0 standard deviations with 1.0
    std = np.where(std == 0.0, 1.0, std)
    
    # 2. Perform element-wise broadcasted subtraction and division
    return (matrix - mean) / std


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_broadcasting()
    benchmark_cache_locality()
    demonstrate_views_vs_copies()
    
    # Testing the Z-score normalization
    print("\nTesting Custom Z-score Normalization:")
    test_matrix = np.array([[10.0, 100.0],
                            [20.0, 200.0],
                            [30.0, 300.0]])
    normed = manual_z_score_normalization(test_matrix)
    print("  Input Matrix:\n", test_matrix)
    print("  Z-score Normed:\n", normed)
    # Standard properties: column mean should be exactly 0, standard dev should be exactly 1!
    print("  Column Means: ", np.mean(normed, axis=0))
    print("  Column Stds:  ", np.std(normed, axis=0))