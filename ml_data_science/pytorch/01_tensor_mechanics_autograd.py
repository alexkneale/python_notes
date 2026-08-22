# ============================================================================
# PYTORCH DEEP DIVE: TENSOR INTERNALS & AUTOGRAD MECHANICS
# ============================================================================
# Target Audience: Senior ML Engineers / Researchers (MSc/PhD level).
# Topic: Storage, Strides, Contiguity, Pinned Memory, and the Autograd DAG.
# Compatibility: Python 3.10+, PyTorch 2.0+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is a Tensor?
   - A tensor is the fundamental data structure in deep learning. It is simply a 
     multi-dimensional array. A 0-D tensor is a scalar (a single number), a 1-D tensor 
     is a vector (a list), a 2-D tensor is a matrix (a table), and higher dimensions 
     represent nested grids of numbers.
2. What is a Gradient?
   - In calculus, the derivative measures the rate of change of a function. A gradient 
     is a vector of these derivatives for multi-variable functions. It points in the direction 
     of steepest increase of the function (e.g., loss). By moving in the opposite direction, 
     we can minimize errors.
3. What is Autograd?
   - Autograd (Automatic Differentiation) is PyTorch's system for tracking all mathematical 
     operations performed on tensors. As operations are executed, PyTorch builds a 
     Directed Acyclic Graph (DAG) representing the computation. Calling `.backward()` 
     on a loss value traverses this graph in reverse, computing and storing derivatives 
     for all input parameters automatically.
"""



import torch
import numpy as np
import time

# ============================================================================
# 1. TENSOR MEMORY MODEL: STORAGE, STRIDES, AND OFFSETS
# ============================================================================
"""
Under the hood, a PyTorch Tensor is decoupled into two parts:
1. A 1D 'Storage' block: The raw, contiguous, flat array of bytes in memory.
2. A 'Tensor Metadata' wrapper: Defines the shape, strides, data type, storage 
   offset, and device. This enables O(1) metadata modifications (views) without 
   copying the underlying numerical data.

A. STRIDES:
   For a tensor of shape (d_1, d_2, ..., d_k), the stride s_i defines how many 
   elements in the flat 1D storage must be skipped to move 1 step along the i-th 
   dimension.
   - For a 2D tensor of shape (M, N) in row-major order:
     - Stride is (N, 1). To move down 1 row, skip N elements. To move right 1 column, skip 1 element.
   - Memory Address Formula:
     Address(i, j) = Storage_Offset + i * stride[0] + j * stride[1]

B. CONTIGUITY:
   A tensor is 'contiguous' if its layout in the flat storage block matches 
   the default reading order (row-major/C-contiguous in PyTorch, i.e., the last 
   dimension varies fastest).
   - contiguous() returns a new tensor with contiguous storage, copying if needed.
   - view() only works on contiguous tensors (or specific stride-compatible ones),
     whereas reshape() will return a view if possible, otherwise it copies.
"""

def demonstrate_tensor_memory_layout():
    print("\n" + "="*80)
    print("1. TENSOR MEMORY LAYOUT: STORAGE, STRIDES, AND OFFSETS")
    print("="*80)

    # Create a 2D tensor of shape (3, 4)
    x = torch.tensor([[10, 20, 30, 40],
                      [50, 60, 70, 80],
                      [90, 100, 110, 120]], dtype=torch.float32)

    print(f"Original x:\n{x}")
    print(f"Shape:          {x.shape}")
    print(f"Strides:        {x.stride()}")
    print(f"Storage Offset: {x.storage_offset()}")
    print(f"Is contiguous?  {x.is_contiguous()}")
    
    # Let's inspect the actual flat 1D storage
    storage = x.untyped_storage()
    print(f"Underlying raw storage bytes count: {len(storage)}")
    # Storage view via CPU storage list
    print(f"Storage view: {x.storage().tolist()}")

    print("\n--- Transposing the Tensor (O(1) Metadata View Change) ---")
    # Transposing swaps strides but does NOT touch the raw storage!
    x_t = x.t()
    print(f"x_t shape:         {x_t.shape}")
    print(f"x_t strides:       {x_t.stride()}  <-- Notice strides are swapped from (4, 1) to (1, 4)!")
    print(f"x_t storage:       {x_t.storage().tolist()} <-- Storage remains identical!")
    print(f"Is x_t contiguous? {x_t.is_contiguous()} <-- False! Strides don't match standard C-ordering.")

    # Show why view() fails on non-contiguous tensors
    try:
        x_t.view(12)
    except RuntimeError as e:
        print(f"\nCaught Expected Error: view() failed on non-contiguous tensor!\n  Error message: {e}")

    # Correct way: use .reshape() (forces copy/contiguity under the hood if non-contiguous) or call .contiguous().view()
    x_t_reshaped = x_t.reshape(12)
    print(f"\nReshaping with .reshape(12): {x_t_reshaped}")
    print(f"Is reshaped tensor contiguous? {x_t_reshaped.is_contiguous()}")
    print(f"Are raw storage pointers different? {x.data_ptr() == x_t_reshaped.data_ptr()}")


# ============================================================================
# 2. PINNED MEMORY AND CPU-TO-GPU TRANSFERS
# ============================================================================
"""
PINNED (PAGE-LOCKED) MEMORY:
By default, CPU memory allocations are pageable. When copying pageable memory to the GPU, 
the CUDA driver must first allocate page-locked (pinned) CPU memory, copy the pageable data 
into it, and then perform the DMA transfer.
- Allocating CPU tensors directly in pinned memory avoids this extra copy.
- It also enables asynchronous (non_blocking=True) transfers to the GPU, overlapping
  CPU computations with GPU communication.
"""

def demonstrate_pinned_memory():
    print("\n" + "="*80)
    print("2. PINNED MEMORY & ASYNCHRONOUS GPU TRANSFERS")
    print("="*80)

    # Simulating pinning API calls
    x_cpu = torch.randn(1000, 1000)
    print(f"Is pageable tensor pinned? {x_cpu.is_pinned()}")
    x_pinned = x_cpu.pin_memory()
    print(f"Is pinned tensor pinned?   {x_pinned.is_pinned()}")

    if not torch.cuda.is_available():
        print("\nCUDA is not available on this machine. Skipping real CUDA speed benchmark.")
        return

    # Real CUDA benchmarks
    device = torch.device("cuda:0")
    size = (5000, 5000)

    # 1. Pageable memory transfer
    t0 = time.perf_counter()
    cpu_pageable = torch.randn(size)
    gpu_tensor_1 = cpu_pageable.to(device)
    torch.cuda.synchronize()
    t_pageable = time.perf_counter() - t0

    # 2. Pinned memory transfer
    t0 = time.perf_counter()
    cpu_pinned = torch.randn(size).pin_memory()
    gpu_tensor_2 = cpu_pinned.to(device, non_blocking=True)
    # We can perform CPU-bound tasks here while GPU copy is happening in parallel
    for _ in range(100000):
        _ = 2 * 2
    torch.cuda.synchronize()
    t_pinned = time.perf_counter() - t0

    print(f"Pageable Transfer Time: {t_pageable:.6f} seconds")
    print(f"Pinned Transfer Time:   {t_pinned:.6f} seconds")
    print(f"Performance speedup:    {t_pageable/t_pinned:.2f}x")


# ============================================================================
# 3. AUTOGRAD MECHANICS AND THE GRADIENT COMPUTATION GRAPH (DAG)
# ============================================================================
"""
Every time you perform operations on tensors with `requires_grad=True`, PyTorch 
builds a Directed Acyclic Graph (DAG) dynamically.
- Leaves (Variables you created) have `grad_fn = None`.
- Intermediate tensors have a `grad_fn` which is a `torch.autograd.Function` object 
  representing the operator.
- Edges represent inputs and outputs.
- In the backward pass, PyTorch executes the backward method of each node, starting 
  at the loss and recursively propagating vector-Jacobian products (VJPs).

Mathematical Representation of VJP:
Let y = f(x). In autograd, we are given the incoming gradient dL/dy (gradient of the 
loss L with respect to output y). We want to compute:
    dL/dx = (dy/dx)^T * dL/dy   <-- The Vector-Jacobian Product!
"""

def inspect_computation_graph():
    print("\n" + "="*80)
    print("3. UNDER THE HOOD OF THE COMPUTATION GRAPH (DAG)")
    print("="*80)

    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    z = y * 3
    loss = z.sum()

    print(f"Leaf tensor (x) grad_fn:       {x.grad_fn}")
    print(f"Intermediate tensor (y) grad_fn: {y.grad_fn} (<PowBackward0>)")
    print(f"Intermediate tensor (z) grad_fn: {z.grad_fn} (<MulBackward0>)")
    print(f"Loss tensor (loss) grad_fn:    {loss.grad_fn}")

    # Let's traverse the graph programmatically!
    print("\nWalking backwards through the next_functions list:")
    fn = loss.grad_fn
    print(f"1. Start function: {fn}")
    # next_functions is a tuple of tuples: ((Function, index), ...)
    print(f"2. Parent of loss: {fn.next_functions}")
    parent_fn = fn.next_functions[0][0]
    print(f"3. Parent of parent (y): {parent_fn.next_functions}")
    leaf_fn = parent_fn.next_functions[0][0]
    print(f"4. Reaching the leaf: {leaf_fn.next_functions}")


# ============================================================================
# 4. CUSTOM AUTOGRAD FUNCTIONS
# ============================================================================
"""
For performance, numerical stability, or custom logic, you can define your own 
autograd functions.
Let's build a numerically stable Custom Log-Sum-Exp (LSE) activation function:
    LSE(x) = log(sum_{i} e^{x_i})
To prevent overflow/underflow, we compute it via:
    LSE(x) = m + log(sum_{i} e^{x_i - m})  where m = max(x)

Mathematical Derivative:
    d(LSE(x))/dx_j = e^{x_j} / (sum_{i} e^{x_i}) = Softmax(x)_j
    Therefore, the VJP for incoming gradient dL/dy is:
    dL/dx_j = (dL/dy) * Softmax(x)_j
"""

class NumericallyStableLSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass.
        ctx is a context object to store information for the backward pass.
        """
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        # Subtract max_val for numerical stability
        shifted_exp = torch.exp(x - max_val)
        sum_exp = torch.sum(shifted_exp, dim=-1, keepdim=True)
        lse = max_val + torch.log(sum_exp)
        
        # Save tensors for backward pass (Softmax is needed for derivative!)
        softmax = shifted_exp / sum_exp
        ctx.save_for_backward(softmax)
        
        # Return exact shape
        return lse.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.
        grad_output is the incoming gradient dL/dy.
        Must return a tuple containing the gradients with respect to each input of forward().
        """
        # Retrieve saved tensors
        softmax, = ctx.saved_tensors
        
        # Softmax shape is (Batch, Features) or (Features,)
        # grad_output shape is (Batch,) or (scalar)
        # We align dimensions to perform element-wise VJP
        grad_input = grad_output.unsqueeze(-1) * softmax
        
        # Must match inputs to forward() (only 1 input tensor x)
        return grad_input


def demonstrate_custom_autograd():
    print("\n" + "="*80)
    print("4. CUSTOM AUTOGRAD FUNCTION: NUMERICALLY STABLE LOG-SUM-EXP")
    print("="*80)

    # Let's feed extreme values to demonstrate stability
    # Standard PyTorch log(sum(exp)) might overflow if float32 reaches ~88
    x_unstable = torch.tensor([100.0, 101.0, 99.0], requires_grad=True)
    
    # Standard computation (naive)
    try:
        naive_lse = torch.log(torch.sum(torch.exp(x_unstable)))
        naive_lse.backward()
        print(f"Naive forward output:  {naive_lse.item()}")
        print(f"Naive gradients:       {x_unstable.grad}")
    except Exception as e:
        print(f"Naive computation errored or returned NaN: {e}")

    # Reset gradients and test stable standard PyTorch implementation
    x_torch = torch.tensor([100.0, 101.0, 99.0], requires_grad=True)
    torch_lse = torch.logsumexp(x_torch, dim=-1)
    torch_lse.backward()
    print(f"Torch stable LSE output:  {torch_lse.item()}")
    print(f"Torch stable gradients:   {x_torch.grad}")

    # Now, let's test our CUSTOM Autograd Function!
    x_custom = torch.tensor([100.0, 101.0, 99.0], requires_grad=True)
    custom_lse_fn = NumericallyStableLSE.apply
    custom_lse = custom_lse_fn(x_custom)
    custom_lse.backward()

    print(f"Custom stable LSE output: {custom_lse.item()}")
    print(f"Custom stable gradients:  {x_custom.grad}")

    # Check analytical equivalence
    assert torch.allclose(torch_lse, custom_lse)
    assert torch.allclose(x_torch.grad, x_custom.grad)
    print("SUCCESS: Custom autograd forward and backward matches PyTorch's native C++ implementation!")


# ============================================================================
# 5. NO_GRAD, DETACH, AND INFERENCE MODE
# ============================================================================
"""
To disable gradient tracking, PyTorch offers:
1. `detach()`: Creates a new view sharing the same storage, but `requires_grad=False`.
   It is still part of the in-place checks.
2. `with torch.no_grad()`: Context manager. Operations inside do not build the DAG.
   However, returned tensors still have tracking metadata, allowing inplace modifications 
   to cause hard-to-detect errors if they were referenced elsewhere.
3. `with torch.inference_mode()`: Faster context manager introduced in PyTorch 1.9+.
   - Disables all tracking.
   - Bypasses version-counter increments (inplace changes to outputs of inference mode 
     throw a clean error if they are later used in autograd).
   - Significant performance improvements by completely skipping DAG overhead.
"""

def benchmark_inference_modes():
    print("\n" + "="*80)
    print("5. PERFORMANCE COMPARISON: NO_GRAD VS INFERENCE_MODE")
    print("="*80)

    # A simple neural network forward pass simulation
    W = torch.randn(2048, 2048, requires_grad=True)
    X = torch.randn(128, 2048)

    iterations = 500

    # 1. Warm-up
    for _ in range(50):
        _ = torch.matmul(X, W)

    # Standard tracking (grad enabled)
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = torch.matmul(X, W)
        _ = torch.relu(out)
    t_grad = time.perf_counter() - t0

    # no_grad context
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            out = torch.matmul(X, W)
            _ = torch.relu(out)
    t_nograd = time.perf_counter() - t0

    # inference_mode context
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            out = torch.matmul(X, W)
            _ = torch.relu(out)
    t_inf = time.perf_counter() - t0

    print(f"Default (Grad Tracking Enabled): {t_grad:.5f} seconds")
    print(f"Using torch.no_grad():           {t_nograd:.5f} seconds")
    print(f"Using torch.inference_mode():    {t_inf:.5f} seconds")
    print(f"Inference Mode Speedup vs Grad:  {t_grad/t_inf:.2f}x")
    print(f"Inference Mode Speedup vs NoGrad:{t_nograd/t_inf:.2f}x")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the difference between a Tensor's shape and its strides?
   - Answer: Shape defines the size of each dimension (e.g., (3, 4)). Strides define 
     the step size (in terms of number of elements) to move 1 step along each dimension 
     in the raw 1D storage array. For a row-major 2D tensor of shape (M, N), 
     the strides are (N, 1).

2. MEDIUM: Explain what happens when you transpose a tensor and then call .view().
   - Answer: Transposing swaps metadata strides in O(1) time without copying memory, 
     rendering the stride order non-contiguous. Calling `.view()` on a non-contiguous 
     tensor raises a RuntimeError because `.view()` requires elements to be laid out 
     sequentially in physical memory. You must call `.contiguous()` first (which copies 
     the data to a new contiguous memory segment) or use `.reshape()`.

3. HARD: How does PyTorch autograd detect illegal in-place operations?
   - Answer: Every tensor has an internal `_version` counter that increments by 1 
     on every in-place modification. When an intermediate tensor is saved for backward 
     propagation (via `ctx.save_for_backward`), its version counter is recorded. 
     During backpropagation, autograd checks if the current version matches the saved 
     version. If they don't match, a RuntimeError is thrown, preventing incorrect 
     gradient calculations due to modified historical states.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom PyTorch Autograd Function for the Sigmoid function 
   from scratch. Do not use PyTorch's native `torch.sigmoid` or `F.sigmoid` inside.
   - Mathematical Derivative: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
   - Code Template to execute:
"""

class CustomSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Forward: s = 1 / (1 + exp(-x))
        # Save output for backward since sigmoid's derivative is formulated in terms of its output!
        s = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved output
        s, = ctx.saved_tensors
        # Backward: dL/dx = dL/ds * ds/dx = grad_output * s * (1 - s)
        grad_input = grad_output * s * (1.0 - s)
        return grad_input

# To test your implementation in an interview setting:
def run_sigmoid_gradcheck():
    x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    # torch.autograd.gradcheck runs numerical finite-difference checks against your analytical gradients
    test_passed = torch.autograd.gradcheck(CustomSigmoid.apply, x, eps=1e-6, atol=1e-4)
    print(f"Sigmoid Autograd Gradcheck Passed: {test_passed}")


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_tensor_memory_layout()
    demonstrate_pinned_memory()
    inspect_computation_graph()
    demonstrate_custom_autograd()
    run_sigmoid_gradcheck()
    benchmark_inference_modes()