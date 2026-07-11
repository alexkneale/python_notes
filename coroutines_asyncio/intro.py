# ============================================================================
# COMPREHENSIVE PYTHON COROUTINES & ASYNCIO TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Topic: async/await, The Event Loop, Tasks, TaskGroups, and Cooperative Multitasking.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import time
import sys
import asyncio
from typing import Coroutine, Any, AsyncGenerator

# ============================================================================
# 1. WHAT ARE COROUTINES? (ASYNC / AWAIT)
# ============================================================================
"""
Historically, Python coroutines were built using generators (`yield`). 
Modern Python (3.5+) introduced NATIVE coroutines using `async def` and `await`.

KEY CONCEPT: COOPERATIVE MULTITASKING
Unlike OS Threads (where the OS forces context switches - "Preemptive Multitasking"),
Asyncio uses a single thread and an Event Loop.
Coroutines explicitly choose when to yield control back to the Event Loop using `await`.
If a coroutine is waiting for I/O (network, file), it says "Hey Event Loop, I'm waiting,
go run some other coroutines in the meantime."
"""

async def basic_coroutine() -> str:
    """
    Defining a function with `async def` makes it a Coroutine Function.
    Calling it DOES NOT execute the body. It returns a Coroutine Object.
    """
    # asyncio.sleep() is a non-blocking sleep. It yields control to the Event Loop.
    await asyncio.sleep(0.1) 
    return "Hello from Coroutine!"

def demonstrate_basics():
    print("--- 1. Coroutine Basics ---")
    
    coro_obj = basic_coroutine()
    print(f"Type of coro_obj: {type(coro_obj)} (Notice it hasn't run yet!)")
    
    # To actually run the top-level coroutine, we must start the Event Loop.
    # asyncio.run() creates a new Event Loop, runs the coroutine, and closes the loop.
    result = asyncio.run(coro_obj)
    print(f"Result: {result}\n")


# ============================================================================
# 2. CONCURRENCY: TASKS AND GATHER
# ============================================================================
"""
If we just `await` coroutines one after the other, they run sequentially.
To run them concurrently, we must wrap them in "Tasks" or use `asyncio.gather()`.

A Task wraps a coroutine and schedules it to run on the Event Loop immediately.
"""

async def mock_network_request(name: str, delay: float) -> str:
    print(f"  [{name}] Started fetching data...")
    await asyncio.sleep(delay) # YIELD CONTROL HERE
    print(f"  [{name}] Finished fetching!")
    return f"{name}_data"

async def sequential_vs_concurrent():
    print("--- 2. Sequential vs Concurrent ---")
    
    # --- SEQUENTIAL (Slow) ---
    print("Running sequentially (expect ~0.6s total)...")
    start = time.perf_counter()
    # Awaiting directly blocks the current coroutine until the awaited one finishes.
    await mock_network_request("Req1", 0.3)
    await mock_network_request("Req2", 0.3)
    print(f"Sequential Time: {time.perf_counter() - start:.2f}s\n")

    # --- CONCURRENT (Fast) ---
    print("Running concurrently via asyncio.gather (expect ~0.3s total)...")
    start = time.perf_counter()
    
    # asyncio.gather schedules multiple awaitables concurrently and waits for all.
    results = await asyncio.gather(
        mock_network_request("Req3", 0.3),
        mock_network_request("Req4", 0.3)
    )
    print(f"Concurrent Results: {results}")
    print(f"Type of result: {type(results)}")
    print(f"Type of first result: {type(results[0])}")
    print(f"Contents of first result: {results[0]}")
    print(f"Concurrent Time: {time.perf_counter() - start:.2f}s\n")


# ============================================================================
# 3. BACKGROUND TASKS & YIELDING CONTROL (COOPERATIVE SCHEDULING)
# ============================================================================
"""
Sometimes, you want to kick off a task in the background and continue running the 
current function without immediately waiting (blocking) for that task to finish.
In asyncio, we do this using `asyncio.create_task(coroutine)`.

THE BEGINNER'S MENTAL MODEL (How Cooperative Scheduling Works):
--------------------------------------------------------------
1. The Chef and the Ticket Wheel (Single-Threaded):
   Since Python asyncio runs on a SINGLE THREAD, think of it like a kitchen with only 
   ONE chef (the Event Loop). At any given split second, the chef can only cook ONE dish.

2. What does `asyncio.create_task(coro)` actually do?
   It writes down a new order (creates a Task) and sticks it on the chef's ticket wheel.
   HOWEVER, the chef is currently busy preparing your current dish (executing the rest 
   of the current function)! The chef WILL NOT stop in the middle of a step to start 
   the new ticket immediately. The background task just sits in the queue.

3. How does the background task get its turn? (Yielding Control):
   The background task can ONLY start executing when the current running function 
   *pauses* and tells the chef: "Hey, I have a step that requires some waiting. 
   You can go look at the other tickets on your wheel while I wait."
   
   We do this using `await` with an asynchronous operation, like `await asyncio.sleep(0.1)`.
   When `await asyncio.sleep()` is called:
   - It pauses the current function.
   - It hands control back to the Event Loop (the chef).
   - The Event Loop scans the ticket wheel, sees the pending `background_worker` task, 
     and runs it!
   - When the sleep time finishes, the Event Loop schedules the main function to resume.
"""

async def background_worker():
    print("  [Background] Heartbeat tick...")
    await asyncio.sleep(0.2)
    print("  [Background] Heartbeat tock...")

async def demonstrate_tasks():
    print("--- 3. Background Tasks ---")
    
    # 1. Schedule background_worker on the Event Loop immediately.
    #    This puts it in the queue, but does NOT start running it yet!
    task = asyncio.create_task(background_worker())
    
    print("  [Main] Task created, doing other work...")
    
    # 2. Yield control! By sleeping, we tell the Event Loop:
    #    "We are pausing for 0.1s. Go run other pending tasks in the queue!"
    #    The Event Loop sees the background task and starts running it.
    await asyncio.sleep(0.1) 
    
    # 3. Resume! When the 0.1s sleep is up, the Event Loop returns here.
    print("  [Main] Back to main! Now let's wait for the background task to finish.")
    
    # 4. We explicitly wait for the background task's completion.
    await task 


# ============================================================================
# 4. PYTHON 3.11+ FEATURE: TASK GROUPS
# ============================================================================
"""
`asyncio.gather` is powerful, but error handling is notoriously tricky (if one
coroutine fails, the others keep running in the background, potentially orphaned).

Python 3.11 introduced `asyncio.TaskGroup`, which provides a robust context manager
for managing multiple tasks. It ensures that if one task fails, all other sibling 
tasks are gracefully cancelled.
"""

async def unstable_request(name: str, delay: float, fail: bool = False):
    await asyncio.sleep(delay)
    if fail:
        raise ConnectionError(f"{name} crashed!")
    return f"{name} Success"

async def demonstrate_task_groups():
    print("\n--- 4. Task Groups (Python 3.11+) ---")
    if sys.version_info < (3, 11):
        print(f"Skipping TaskGroup demo. Python 3.11+ required, you are on {sys.version_info.major}.{sys.version_info.minor}")
        return

    try:
        # The 'async with' block waits for ALL tasks inside it to finish.
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(unstable_request("TaskA", 0.1))
            task2 = tg.create_task(unstable_request("TaskB", 0.2, fail=True)) # This will crash
            task3 = tg.create_task(unstable_request("TaskC", 0.5))
            
    except ExceptionGroup as eg:
        print(f"Caught ExceptionGroup! One of the tasks failed.")
        print(f"TaskA cancelled? {task1.cancelled()} (Finished before crash)")
        # Because TaskB crashed at 0.2s, TaskC (which takes 0.5s) is automatically cancelled!
        print(f"TaskC cancelled? {task3.cancelled()} (Killed by TaskGroup!)")


# ============================================================================
# 5. ASYNC CONTEXT MANAGERS & ITERATORS (DUNDER METHODS)
# ============================================================================
"""
You can create your own async iterators and context managers using specialized
dunder methods: `__aenter__`, `__aexit__`, `__aiter__`, `__anext__`.
"""

class AsyncDatabaseConnection:
    """An Async Context Manager (`async with`)"""
    
    async def __aenter__(self):
        print("  [Async DB] Connecting to DB... (simulated network wait)")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("  [Async DB] Closing connection... (simulated network wait)")
        await asyncio.sleep(0.1)

async def async_data_generator() -> AsyncGenerator[int, None]:
    """An Async Generator (`async for`). Yields values over time."""
    for i in range(3):
        await asyncio.sleep(0.1) # Simulate slow database cursor fetch
        yield i

async def demonstrate_async_dunders():
    print("\n--- 5. Async Context Managers & Generators ---")
    
    async with AsyncDatabaseConnection() as db:
        print("  [Main] Connected! Fetching data stream...")
        
        # Iterating over an async generator requires `async for`
        async for data in async_data_generator():
            print(f"    -> Received chunk: {data}")


# ============================================================================
# 6. THE GOLDEN RULE: NEVER BLOCK THE EVENT LOOP!
# ============================================================================
"""
Because asyncio is SINGLE-THREADED, if you run a blocking synchronous function 
(like `time.sleep()`, `requests.get()`, or heavy CPU math) inside a coroutine, 
YOU FREEZE THE ENTIRE EVENT LOOP. No other coroutines can run!

BUT WAIT! If asyncio is single-threaded, how can we use background threads?
-----------------------------------------------------------------------------
1. The Core Loop is Single-Threaded:
   The main event loop itself, along with your async functions, runs entirely on 
   ONE thread (usually the main thread).

2. Python's Secret Thread Pool:
   Under the hood, `asyncio.to_thread()` offloads the synchronous function to a 
   separate, native operating system (OS) thread managed by a global `ThreadPoolExecutor`.

3. The Concurrency Trick (GIL Release):
   While the background OS thread is blocked (sleeping, querying a database, or waiting 
   on a network request), it releases Python's Global Interpreter Lock (GIL).
   This allows our single-threaded Event Loop (running on the main thread) to keep 
   executing other Python coroutines unhindered.

4. CPU-Bound Gotcha:
   If the blocking function is CPU-bound (e.g. calculating millions of digits of Pi in pure Python),
   it will STILL freeze/slow down the Event Loop even in a background thread because Python 
   only allows one thread to execute Python bytecode at any given moment (due to the GIL).
   For CPU-bound tasks, use Multiprocessing!
"""

def blocking_sync_function():
    """A standard synchronous function that blocks the current thread."""
    time.sleep(0.3)
    return "Blocking work done"

async def background_counter():
    """Just prints numbers to prove if the loop is running or frozen."""
    for i in range(5):
        print(f"    [Counter] {i}")
        await asyncio.sleep(0.1)

async def demonstrate_blocking_vs_unblocking():
    print("\n--- 6. Blocking the Event Loop vs to_thread ---")
    
    print("A: The WRONG Way (Calling sync code directly)")
    counter_task = asyncio.create_task(background_counter())
    await asyncio.sleep(0.05) # Let counter print '0'
    
    # ⚠️ THIS FREEZES THE LOOP! The counter task WILL NOT PRINT while this sleeps.
    print("  [Main] Executing time.sleep(0.3)... (Notice the counter stops!)")
    time.sleep(0.3) 
    print("  [Main] time.sleep finished.")
    
    # Wait for counter to finish before moving to the correct way
    await counter_task 
    
    print("\nB: The RIGHT Way (asyncio.to_thread)")
    counter_task = asyncio.create_task(background_counter())
    await asyncio.sleep(0.05) # Let counter print '0'
    
    # ✅ This offloads the blocking work to a background thread pool, 
    # allowing the Event Loop to keep processing the counter_task!
    print("  [Main] Executing asyncio.to_thread... (Notice the counter keeps going!)")
    result = await asyncio.to_thread(blocking_sync_function)
    print(f"  [Main] Thread finished: {result}")
    
    await counter_task


# ============================================================================
# 7. INTERACTING WITH THE EVENT LOOP (`asyncio.get_running_loop`)
# ============================================================================
"""
Sometimes, an advanced coroutine needs to access the active Event Loop instance directly—
for example, to schedule a synchronous callback, create a Future, or run a task in a 
custom executor pool.

Why `asyncio.get_running_loop()` is preferred over `asyncio.get_event_loop()` (Python 3.7+):
-----------------------------------------------------------------------------------------
- `asyncio.get_event_loop()` (legacy): Has highly complex, context-dependent behavior. If no loop 
  exists in the current thread, it may automatically create and bind one. This causes massive 
  headaches, resource leaks, and test isolation bugs, especially in multi-threaded programs.
- `asyncio.get_running_loop()` (modern): Safe and explicit. It returns the running event loop 
  on the current thread. If no loop is running, it raises a `RuntimeError` immediately.
"""

def synchronous_callback(arg):
    print(f"  [Callback] Loop executed me immediately! Received: {arg}")

async def demonstrate_running_loop():
    print("\n--- 7. Interacting with the Running Loop ---")
    
    # 1. Retrieve the running event loop instance safely
    try:
        loop = asyncio.get_running_loop()
        print(f"  Successfully retrieved running loop: {loop}")
    except RuntimeError as e:
        print(f"  Error: {e}")
        return

    # 2. Schedule a synchronous callback to run on the loop as soon as possible
    loop.call_soon(synchronous_callback, "Hello from the future!")
    
    # Let's pause for a moment to give the loop a chance to execute the scheduled callback
    await asyncio.sleep(0.01)

    # 3. Behind the Scenes of `to_thread`:
    # `asyncio.to_thread()` is actually a modern convenience wrapper around `loop.run_in_executor()`!
    # Here is how you do it manually using the loop's default executor:
    print("  [Main] Running blocking function via loop.run_in_executor...")
    result = await loop.run_in_executor(None, blocking_sync_function)
    print(f"  [Main] run_in_executor result: {result}")


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

def main():
    """
    Standard entry point. We use standard functions to organize our asyncio.run()
    calls to keep the demonstrations clean and isolated.
    """
    demonstrate_basics()
    
    # Run the concurrent demo
    asyncio.run(sequential_vs_concurrent())
    
    # Run the background task demo
    asyncio.run(demonstrate_tasks())
    
    # Run TaskGroups (If Python >= 3.11)
    asyncio.run(demonstrate_task_groups())
    
    # Run Async Dunders
    asyncio.run(demonstrate_async_dunders())
    
    # Run Blocking prevention
    asyncio.run(demonstrate_blocking_vs_unblocking())
    
    # Run Event Loop inspection
    asyncio.run(demonstrate_running_loop())

    print("\n--- Tutorial Complete ---")
    print("Summary:")
    print("1. asyncio is single-threaded, cooperative multitasking.")
    print("2. 'await' suspends the coroutine and gives control back to the Event Loop.")
    print("3. NEVER run blocking code (time.sleep, CPU-heavy math) directly inside async functions.")
    print("4. Use TaskGroups (or gather) to run things concurrently.")
    print("5. Use asyncio.to_thread() to offload blocking sync code to background threads.")
    print("6. Use asyncio.get_running_loop() inside coroutines to safely interact with the loop.")


# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (ASYNCIO & COROUTINES)
# ============================================================================
"""
Q1: What does "cooperative multitasking" mean in the context of asyncio?
A: It means that the operating system does not preemptively interrupt or switch between tasks (unlike threading).
   Instead, tasks must explicitly yield control back to the Event Loop using the `await` keyword.
   If a task never awaits (or runs blocking CPU/IO operations), no other tasks can run, freezing the entire application.

Q2: What happens if you run `time.sleep(5)` inside an async function, and how do you fix it?
A: Since `time.sleep()` is synchronous and blocking, it blocks the main thread. Because asyncio runs on a single thread,
   it blocks the Event Loop entirely, preventing any other coroutine from executing for 5 seconds.
   To fix it, you should either:
   1. Use `await asyncio.sleep(5)` if possible.
   2. Run the blocking function in a separate executor thread using `await asyncio.to_thread()` or `loop.run_in_executor()`.

Q3: What is the difference between `asyncio.gather()` and Python 3.11's `asyncio.TaskGroup`?
A: 
- `asyncio.gather(*aws)` is the older way to group concurrent tasks. If one task fails, the other tasks continue running in the background (unless explicitly canceled), which can cause silent resource leaks and orphaned tasks.
- `asyncio.TaskGroup()` uses an asynchronous context manager. If one task inside the group raises an exception, all other active tasks in that group are automatically and immediately canceled. This implements safer "structured concurrency".

Q4: What is the difference between `asyncio.get_event_loop()` and `asyncio.get_running_loop()`?
A: 
- `asyncio.get_running_loop()` (introduced in Python 3.7) is strict: it only returns the active running event loop for the current thread. If no loop is running, it raises a `RuntimeError` immediately. This is the preferred way to get the loop inside coroutines.
- `asyncio.get_event_loop()` has legacy, context-dependent behavior: if no loop is running, it may create and bind a new one in the current thread. This can cause silent resource leaks, thread safety issues, and test isolation bugs.
"""


if __name__ == "__main__":
    main()
