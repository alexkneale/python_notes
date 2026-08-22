# ============================================================================
# JAX TRANSFORMATIONS: ADVANCED JIT COMPILATION, AUTOMATIC DIFF, AND VMAP
# ============================================================================
# Target Audience: Senior ML Engineers, Applied Mathematicians, Researchers.
# Topic: Tracing/Tracers, Static Arguments, Reverse-Mode AD, composable Jacobians,
#        and vectorization with vmap.
# Compatibility: Python 3.10+, JAX 0.4+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is JIT (Just-In-Time) Compilation?
   - Standard Python runs code line-by-line, which is slow. JIT compilation (via `jax.jit`) 
     reads your Python/NumPy functions, converts them into an abstract mathematical map, 
     and compiles them into highly optimized machine code when the function is first called. 
     Subsequent calls run at native hardware speeds.
2. What is Vectorization (vmap)?
   - If you write a function designed to process a single sample, running it on a batch of 
     samples usually requires a slow Python `for` loop. `jax.vmap` automatically vectorizes 
     your function, compiling it so it can process batches of arbitrary sizes in parallel 
     at the compiler level.
"""



import jax
import jax.numpy as jnp
import time

# ============================================================================
# 1. JIT COMPILATION & TRACING UNDER THE HOOD
# ============================================================================
"""
`jax.jit` compiles your Python function into an optimized machine binary using XLA.
Tracing:
When you compile a function and run it for the first time, JAX passes symbolic 
abstract shapes called 'Tracers' into your function instead of actual values. 
JAX records all JAX operations performed on these tracers to build a 'Jaxpr' 
(JAX expression). This Jaxpr is then compiled via XLA.

Static Arguments:
XLA requires shape/size constants to optimize execution pipelines.
If you have a python conditional (e.g. `if condition:`) inside a compiled function, 
the compiler needs to trace it. If 'condition' is a dynamic JAX array, JAX cannot trace 
both paths simultaneously, raising a ConcretizationTypeError.
- Solution: Declare parameters causing branching as 'static' via `static_argnums` or 
  `static_argnames`. This tells JAX to compile a separate binary *for each unique value* 
  of that static argument.
"""

def demonstrate_jit_and_tracing():
    print("\n" + "="*80)
    print("1. JIT COMPILATION & THE TRACING PIPELINE")
    print("="*80)

    # Let's inspect JAX tracers
    def sample_func(x):
        print(f"  [Tracing run] x is: {x}") # This print statement only runs during tracing!
        return x ** 2 + 5.0

    print("Compiling sample_func...")
    jit_func = jax.jit(sample_func)

    # Call 1: Tracing triggers. You will see "[Tracing run]" in the output.
    print("\nFirst execution (triggers tracing + compilation):")
    res1 = jit_func(2.0)
    print(f"Result 1: {res1}")

    # Call 2: Directly executes the compiled XLA binary. 
    # Python code is bypassed, so "[Tracing run]" will NOT print!
    print("\nSecond execution (direct binary execution):")
    res2 = jit_func(4.0)
    print(f"Result 2: {res2}")

    print("\n--- Dealing with Branching and Static Arguments ---")
    
    # This will fail if compiled directly with dynamic arguments because 'is_multiply' controls control flow
    def dynamic_branch(x, is_multiply):
        if is_multiply:
            return x * 10.0
        else:
            return x + 10.0

    # Declare that the second argument (index 1) is static!
    jit_branch = jax.jit(dynamic_branch, static_argnums=(1,))

    # JAX will compile two different binaries behind the scenes
    r_mult = jit_branch(5.0, True)
    r_add = jit_branch(5.0, False)
    print(f"Static-compiled branch (Multiply=True): {r_mult}")
    print(f"Static-compiled branch (Multiply=False): {r_add}")


# ============================================================================
# 2. AUTOMATIC DIFFERENTIATION: GRAD, JACOBIANS, AND HESSIANS
# ============================================================================
"""
JAX implements automatic differentiation (AD).
1. Reverse-Mode AD (backpropagation): Computes dL/dx. Most efficient for functions 
   with few outputs and many inputs (e.g. loss functions: R^N -> R^1). 
   Created via `jax.grad`.
2. Forward-Mode AD: Computes dy/dx. Most efficient for functions with many outputs 
   and few inputs (e.g. R^1 -> R^N). Created via `jax.jacobian` (under-the-hood).

Mathematical Composition:
We can compose JAX transformations to compute high-order derivatives!
Let's find the Hessian Matrix H of a scalar function:
    H_ij = d^2(f(x)) / (dx_i dx_j)
Hessian can be calculated as the Jacobian of the Gradient:
    Hessian(f) = jacobian(gradient(f))
"""

def demonstrate_advanced_differentiation():
    print("\n" + "="*80)
    print("2. AUTOMATIC DIFFERENTIATION (GRAD, JACOBIAN, HESSIAN)")
    print("="*80)

    # Let's write a mathematical function: f(x) = 0.5 * x^T * A * x + b^T * x
    # Hessian of f(x) should be exactly (A + A^T) / 2 (or A if A is symmetric).
    A = jnp.array([[3.0, 1.0],
                   [1.0, 4.0]])
    b = jnp.array([2.0, -1.0])

    def f(x):
        return 0.5 * jnp.dot(x, jnp.dot(A, x)) + jnp.dot(b, x)

    # 1. Gradient: dL/dx
    grad_f = jax.grad(f)
    x_test = jnp.array([1.0, 2.0])

    print(f"Evaluate f(x) at {x_test}:   {f(x_test)}")
    print(f"Gradient grad_f(x):           {grad_f(x_test)}")
    # Analytical gradient: Ax + b
    # A*x + b = [3*1 + 1*2, 1*1 + 4*2] + [2, -1] = [5, 9] + [2, -1] = [7, 8]
    print(f"Analytical gradient check:    {jnp.dot(A, x_test) + b}")

    # 2. Hessian: Jacobian of the Gradient!
    # jax.jacobian uses reverse-mode AD under-the-hood (jacrev).
    hessian_f = jax.jacobian(grad_f)
    print(f"\nHessian matrix H(x):\n{hessian_f(x_test)}")
    print(f"Analytical Hessian (A is symmetric, so should be A):\n{A}")

    # 3. value_and_grad: Computes both f(x) and grad_f(x) in a single unified pass
    # Highly recommended inside optimization training loops!
    val, gradient = jax.value_and_grad(f)(x_test)
    print(f"\nValue and Gradient in one pass:\n  Value: {val}\n  Grad:  {gradient}")


# ============================================================================
# 3. VECTORIZATION VIA VMAP (VECTORIZED MAP)
# ============================================================================
"""
Often, you write code designed for a single sample (e.g. updating a state-vector, 
or modeling a physical particle's Brownian motion).
To map it over a batch of data:
- In NumPy/PyTorch: You have to manually adjust array operations to add batch dimensions, 
  which is highly error-prone.
- In Pure Python: You write loops, which are slow and do not parallelize on devices.
- In JAX: You write the function for a SINGLE sample, then wrap it in `jax.vmap`!
  `vmap` compiles the batch mapping down to optimized vectorized SIMD operations in XLA.
"""

def demonstrate_vmap_speed():
    print("\n" + "="*80)
    print("3. VECTORIZATION: VMAP VS PYTHON LOOPS")
    print("="*80)

    # Suppose we have a custom non-linear function simulating a single particle's 
    # energy dissipation over time:
    def single_particle_simulation(params, time_steps):
        # params is (alpha, beta)
        alpha, beta = params[0], params[1]
        
        # We loop internally for sequential time steps (simulating dynamics)
        val = 1.0
        for t in range(len(time_steps)):
            val = val * alpha - beta * time_steps[t]
        return val

    # Generate test parameters for a massive batch of 10,000 particles
    N_particles = 10000
    params_batch = jnp.stack([jnp.ones(N_particles) * 0.99, jnp.ones(N_particles) * 0.02], axis=1) # (10000, 2)
    time_steps = jnp.linspace(0.0, 1.0, 50)

    # 1. Manual looping in Python over the batch
    t0 = time.perf_counter()
    loop_results = []
    for i in range(N_particles):
        loop_results.append(single_particle_simulation(params_batch[i], time_steps))
    loop_results = jnp.array(loop_results)
    t_loop = time.perf_counter() - t0
    print(f"Manual Python Loop Time:   {t_loop:.5f} seconds")

    # 2. Vectorization using jax.vmap!
    # vmap arguments:
    # in_axes: Defines which axes of the inputs map to the batch. 
    #          - params_batch shape is (10000, 2), so batch is along axis 0.
    #          - time_steps has NO batch dimension, so we pass None to broadcast it!
    vmap_simulation = jax.vmap(single_particle_simulation, in_axes=(0, None))
    
    # Compile with JIT for ultimate speed
    jit_vmap_simulation = jax.jit(vmap_simulation)

    # Warm-up compile
    _ = jit_vmap_simulation(params_batch, time_steps).block_until_ready()

    # Benchmark JIT + VMAP
    t0 = time.perf_counter()
    vmap_results = jit_vmap_simulation(params_batch, time_steps).block_until_ready()
    t_vmap = time.perf_counter() - t0

    print(f"JAX JIT + VMAP Time:        {t_vmap:.5f} seconds")
    print(f"Performance Speedup:        {t_loop / t_vmap:.2f}x")
    
    # Assert correctness
    assert jnp.allclose(loop_results, vmap_results)
    print("SUCCESS: VMAP yields identical results to manual loop but scales to billions of FLOPs on device!")


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the computational difference between Reverse-Mode AD (grad) and Forward-Mode AD?
   - Answer: Reverse-Mode AD (backpropagation) starts at the scalar output (Loss) and 
     propagates adjoints backwards to inputs. It is O(1) in the number of outputs and 
     O(N) in the number of inputs, making it optimal for Neural Networks where parameters >> outputs. 
     Forward-Mode AD propagates tangent vectors forward, making it O(1) in the number of inputs 
     and O(M) in the number of outputs, optimal for mappings where inputs << outputs.

2. MEDIUM: Why does `if x > 0:` throw a ConcretizationTypeError inside a @jax.jit function?
   - Answer: During compilation, JAX passes dynamic placeholder "Tracers" through 
     the function to build the Jaxpr. Tracers do not contain concrete numerical data. 
     Evaluating `x > 0` requires a concrete boolean to decide which branch to compile. 
     Since it doesn't have one, it errors. To fix, either make `x` static using `static_argnums`, 
     or use JAX's structured control-flow primitive: `jax.lax.cond`.

3. HARD: How does `vmap` work under the hood inside XLA?
   - Answer: `vmap` does not execute Python loops. Instead, it parses the function's 
     underlying Jaxpr and transforms each primitive operation (e.g. dot products, additions) 
     into its batched equivalent (e.g. promoting a matrix-vector multiplication into a 
     matrix-matrix multiplication). This batch-promoted Jaxpr is then compiled into high-performance 
     SIMD vector operations by XLA.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a JIT-compiled function `compute_jacobian_at_x(f, x)` that computes 
   the Jacobian matrix of a vector-valued function $f: \mathbb{R}^3 \to \mathbb{R}^2$ 
   using JAX transformations.
   - Code Template to execute:
"""

def sample_vector_func(x):
    # f(x) = [x_0^2 + x_1, x_2 * sin(x_0)]
    return jnp.stack([
        x[0]**2 + x[1],
        x[2] * jnp.sin(x[0])
    ])

@jax.jit
def compute_jacobian_at_x(x: jnp.ndarray) -> jnp.ndarray:
    # Composes jax.jacobian on our function
    jacobian_fn = jax.jacobian(sample_vector_func)
    return jacobian_fn(x)


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_jit_and_tracing()
    demonstrate_advanced_differentiation()
    demonstrate_vmap_speed()
    
    # Testing the compiled Jacobian
    print("\nTesting JIT-Compiled Jacobian:")
    x_test = jnp.array([jnp.pi / 2, 1.0, 2.0])
    # Analytical Jacobian:
    # df_0/dx = [2*x_0, 1, 0] = [pi, 1, 0]
    # df_1/dx = [x_2 * cos(x_0), 0, sin(x_0)] = [2*cos(pi/2), 0, sin(pi/2)] = [0, 0, 1]
    jac = compute_jacobian_at_x(x_test)
    print(f"  Input: {x_test.tolist()}")
    print(f"  Jacobian Matrix:\n{jac}")