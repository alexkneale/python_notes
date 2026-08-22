# ============================================================================
# TENSORFLOW DEEP DIVE: GRAPH EXECUTION, TRACING, AND GRADIENT TAPE
# ============================================================================
# Target Audience: Senior AI Engineers, ML Researchers.
# Topic: tf.function Tracing, ConcreteFunctions, Eager vs. Graph mode, 
#        and tf.GradientTape Auto-Diff mechanics.
# Compatibility: Python 3.10+, TensorFlow 2.10+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is TensorFlow?
   - TensorFlow is an end-to-end open-source platform for machine learning. It provides 
     powerful tensor operations and hardware acceleration to build and train neural networks.
2. What is a Computation Graph?
   - Instead of running Python code line-by-line, TensorFlow can compile your mathematical 
     equations into a static computation graph. This graph maps out every calculation, 
     allowing TensorFlow to optimize memory reuse and execute nodes in parallel on the GPU.
3. What is tf.GradientTape?
   - TensorFlow uses `tf.GradientTape` to compute derivatives. Think of it like a video tape 
     recorder: inside the tape context, TensorFlow records all mathematical steps as they happen. 
     It then plays the tape backwards to calculate gradients of a loss with respect to your 
     trainable variables.
"""



import tensorflow as tf
import numpy as np
import time

# ============================================================================
# 1. GRAPH EXECUTION: tf.function, TRACING, AND SIGNATURES
# ============================================================================
"""
TF 2.x executes in Eager Mode by default (imperative, line-by-line, like PyTorch).
To compile a function into a highly optimized static computation graph, we decorate 
it with `@tf.function`.

Under the Hood (Tracing & ConcreteFunctions):
1. First Call (Tracing Phase): TensorFlow executes the Python function using symbolic 
   placeholders called TensorSpecs. It traces the operations to build a computation graph.
2. JIT Compilation: This graph is optimized (node pruning, constant folding, operation fusion) 
   and compiled via XLA (Accelerated Linear Algebra).
3. ConcreteFunction: TF caches this compiled graph as a 'ConcreteFunction', keyed 
   by the input argument types and shapes (signature).
4. Subsequent Calls: If inputs match a cached signature, TF executes the compiled 
   ConcreteFunction directly, bypassing Python entirely.
5. Polymorphic Redundant Tracing: If you pass different shapes or Python types (like ints/floats 
   instead of tf.Tensors), TF will re-trace and re-compile a new graph. 
   Redundant tracing slows down training and wastes host memory! Prevent this by specifying 
   input_signature Specs.
"""

def demonstrate_tf_function_tracing():
    print("\n" + "="*80)
    print("1. TENSORFLOW GRAPH COMPILATION & TRACING")
    print("="*80)

    # We define a function and print to trace when compilation actually runs
    def raw_computation(x):
        print(f"  [PYTHON TRACING] Executing Python code for shape {x.shape}")
        return tf.reduce_sum(tf.square(x))

    # Compile with tf.function
    compiled_func = tf.function(raw_computation)

    # 1. Call with shape (3,)
    print("\nFirst Call (Shape (3,)):")
    x1 = tf.constant([1.0, 2.0, 3.0])
    res1 = compiled_func(x1)
    print(f"  Result: {res1}")

    # 2. Call again with identical shape (3,) -> Reuses cached ConcreteFunction!
    # Note: "[PYTHON TRACING]" will NOT print!
    print("\nSecond Call (Identical Shape (3,)):")
    res2 = compiled_func(tf.constant([4.0, 5.0, 6.0]))
    print(f"  Result: {res2}")

    # 3. Call with different shape (4,) -> Forces a completely new compilation trace!
    print("\nThird Call (New Shape (4,)):")
    x3 = tf.constant([1.0, 2.0, 3.0, 4.0])
    res3 = compiled_func(x3)
    print(f"  Result: {res3}")


# ============================================================================
# 2. AUTOMATIC DIFFERENTIATION WITH tf.GradientTape
# ============================================================================
"""
TensorFlow uses `tf.GradientTape` to record operations for automatic differentiation.
Key Mechanics:
1. Active Variable Watching: By default, the tape automatically "watches" all trainable 
   variables (`tf.Variable` created with `trainable=True`). It does NOT watch constant 
   tensors (`tf.constant`) or non-trainable variables unless you call `tape.watch(tensor)` explicitly.
2. Resource Release (Non-Persistent Tapes): By default, a GradientTape releases its recorded 
   resources as soon as `tape.gradient()` is called. To calculate multiple gradients 
   (e.g., separate gradients for different outputs), you must instantiate it as 
   persistent: `tf.GradientTape(persistent=True)`. You then manually release resources 
   using `del tape`.
"""

def demonstrate_gradient_tape():
    print("\n" + "="*80)
    print("2. UNDER THE HOOD OF TF.GRADIENTTAPE")
    print("="*80)

    # Variables vs Constants
    w = tf.Variable(3.0, trainable=True)
    b = tf.Variable(1.0, trainable=False) # Not watched automatically!
    x = tf.constant(2.0)                  # Not watched automatically!

    # 1. Non-persistent tape
    with tf.GradientTape() as tape:
        # Manually watch non-trainable variable b and constant x
        tape.watch(b)
        tape.watch(x)
        
        # Function: y = w*x^2 + b
        y = w * (x ** 2) + b

    # Compute gradients of y with respect to [w, b, x]
    # d_y/d_w = x^2 = 4.0
    # d_y/d_b = 1.0
    # d_y/d_x = 2*w*x = 12.0
    grads = tape.gradient(y, [w, b, x])
    print("Calculated Gradients:")
    print(f"  dy/dw: {grads[0].numpy()} (Expected: 4.0)")
    print(f"  dy/db: {grads[1].numpy()} (Expected: 1.0)")
    print(f"  dy/dx: {grads[2].numpy()} (Expected: 12.0)")

    # 2. Persistent tape demonstration
    # Needed for calculating higher-order gradients (Hessians) or multiple objectives
    x_val = tf.Variable(2.0)
    with tf.GradientTape(persistent=True) as tape:
        # y = x^3
        y = x_val ** 3
        # dy/dx = 3*x^2
        dy_dx = tape.gradient(y, x_val)

    # Compute second-order gradient (d^2y/dx^2 = 6*x) inside the persistent block or outside
    d2y_dx2 = tape.gradient(dy_dx, x_val)
    print(f"\nPersistent Tape High-Order Gradients:")
    print(f"  First-order derivative dy/dx:   {dy_dx.numpy()} (Expected: 12.0)")
    print(f"  Second-order derivative d2y/dx2: {d2y_dx2.numpy()} (Expected: 12.0)")
    
    # Must manually clean up persistent tapes!
    del tape


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is Eager Execution in TensorFlow, and how does it compare to Graph Mode?
   - Answer: Eager mode executes operations immediately when called, returning concrete values 
     to Python, which simplifies debugging and prototyping. Graph mode builds an abstract 
     symbolic dataflow graph first, which is compiled and optimized (node fusion, pipeline 
     parallelism) before execution. Graph mode is far faster for large production workloads.

2. MEDIUM: Why does passing Python integers to a tf.function decorated method trigger 
   repeated re-tracing? How do you prevent it?
   - Answer: JAX/TF traces look at concrete values of raw Python types (int, float) rather 
     than abstract shapes. If you pass different integers, TF compiles a new static graph 
     for each distinct value. To prevent this, either wrap Python inputs in `tf.constant()`, 
     or define a static signature using `input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)]` 
     inside the decorator.

3. HARD: What is a ConcreteFunction in TensorFlow, and what is its relation to tf.function?
   - Answer: `tf.function` is a polymorphic wrapper. When called with a specific signature, 
     it traces the function to build a specialized static graph. This tracked graph is stored 
     as a `ConcreteFunction` object. A single `tf.function` maintains a cache of several 
     `ConcreteFunction` instances, routing incoming inputs to their corresponding matching 
     compiled binary.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom training optimization step `custom_gd_update(w, x, y_true, lr)` 
   using a GradientTape that computes the gradients of Mean Squared Error (MSE) for a single 
   linear neuron ($y = w \cdot x$) and applies a manual gradient descent weight update.
   - Code Template to execute:
"""

def custom_gd_update(w: tf.Variable, x: tf.Tensor, y_true: tf.Tensor, lr: float) -> float:
    with tf.GradientTape() as tape:
        # Forward pass: y_pred = w * x
        y_pred = w * x
        # Loss: MSE = (y_pred - y_true)^2
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        
    # Calculate gradient of loss with respect to w
    grad_w = tape.gradient(loss, w)
    
    # Apply manual gradient descent update: w = w - lr * grad_w
    # Note: must do in-place update using .assign() to update tf.Variable!
    w.assign_sub(lr * grad_w)
    
    return loss.numpy()


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_tf_function_tracing()
    demonstrate_gradient_tape()
    
    # Testing the manual gradient update
    print("\nTesting Custom Gradient Descent Update:")
    w = tf.Variable(5.0)
    x = tf.constant(2.0)
    y_true = tf.constant(8.0) # Optimal weight is 4.0
    
    initial_loss = (5.0 * 2.0 - 8.0)**2 # (10-8)^2 = 4.0
    loss = custom_gd_update(w, x, y_true, lr=0.1)
    
    print(f"  Initial Loss: {initial_loss:.2f}")
    print(f"  Loss returned by step: {loss:.2f} (Expected: 4.0)")
    print(f"  Updated Weight (w):    {w.numpy():.2f} (Expected: 5.0 - 0.1 * [2*(10-8)*2] = 5.0 - 0.1 * 8.0 = 4.2)")