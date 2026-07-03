# ============================================================================
# COMPREHENSIVE PYTHON THREADING & THE GIL TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Topic: Global Interpreter Lock (GIL), OS Threads, Sync Primitives, and Pools.
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import dis
import time
import threading
import concurrent.futures
from urllib.request import urlopen
from typing import List

# ============================================================================
# 1. WHAT IS THE GIL (GLOBAL INTERPRETER LOCK)?
# ============================================================================
"""
Python threads are REAL OS-level threads (e.g., pthreads on Linux, Windows Threads).
However, CPython has a mutex (lock) called the Global Interpreter Lock (GIL).

WHAT IT DOES:
The GIL ensures that ONLY ONE OS THREAD CAN EXECUTE PYTHON BYTECODE AT A TIME.
Why? Because CPython's memory management (Reference Counting) is not thread-safe.
If two threads incremented a reference count simultaneously, memory would leak 
or crash. The GIL protects CPython's internal state.

THE CATCH:
Because of the GIL, multithreading in Python DOES NOT speed up CPU-bound tasks 
(like heavy math). Multiple threads will just fight for the GIL, resulting in 
context-switching overhead that makes the program SLOWER than a single thread.

THE LOOPHOLE: I/O-BOUND TASKS
When a Python thread performs an I/O operation (Network request, reading a file, 
time.sleep()), IT RELEASES THE GIL. This allows other Python threads to run.
Therefore, Threading is incredibly useful for I/O-bound programs!

*Note: Python 3.13 introduces an experimental "free-threading" (nogil) mode,
but the GIL remains the default standard for the foreseeable future.*
"""


# ============================================================================
# 2. CPU-BOUND VS I/O-BOUND (PROVING THE GIL)
# ============================================================================

def cpu_bound_task():
    """A heavy math task. Keeps the CPU busy."""
    count = 0
    for _ in range(20_000_000):
        count += 1

def io_bound_task():
    """Simulates a network request or file read. Releases the GIL!"""
    time.sleep(0.5)

def measure_execution(task, num_threads: int, name: str):
    print(f"Running {num_threads} threads for: {name}...")
    start = time.time()
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=task)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join() # Wait for all threads to finish
        
    print(f"  -> Finished in {time.time() - start:.2f} seconds\n")

def demonstrate_gil_impact():
    print("--- 1 & 2. The GIL: CPU vs I/O Bound ---")
    
    # I/O Bound: 4 threads take ~0.5s total because the GIL is released during sleep!
    measure_execution(io_bound_task, 1, "I/O Bound (1 Thread)")
    measure_execution(io_bound_task, 4, "I/O Bound (4 Threads) <- MASSIVE SPEEDUP")
    
    # CPU Bound: 2 threads take LONGER than 1 thread due to GIL context switching!
    # (Uncomment below to test, but it takes a few seconds)
    # measure_execution(cpu_bound_task, 1, "CPU Bound (1 Thread)")
    # measure_execution(cpu_bound_task, 2, "CPU Bound (2 Threads) <- SLOWER!")


# ============================================================================
# 3. RACE CONDITIONS (IF THE GIL EXISTS, WHY DO WE NEED LOCKS?)
# ============================================================================
"""
Wait, if the GIL only allows one thread to execute at a time, why do we have 
race conditions? 

Because the GIL protects Python INTERNALS, not YOUR logic.
An operation like `x += 1` is actually 4 separate bytecode instructions:
  1. LOAD_FAST (load x)
  2. LOAD_CONST (load 1)
  3. INPLACE_ADD (add them)
  4. STORE_FAST (save back to x)

The GIL can force a thread context switch BETWEEN any of these bytecodes!
"""

# Shared state
counter = 0

def race_condition_worker():
    global counter
    for _ in range(100_000):
        # We simulate the vulnerability of `counter += 1` by doing it manually,
        # ensuring the context switch ruins our data.
        current_value = counter
        # A context switch here means multiple threads see the same current_value!
        counter = current_value + 1

def demonstrate_race_condition():
    print("--- 3. Race Conditions ---")
    global counter
    counter = 0
    
    t1 = threading.Thread(target=race_condition_worker)
    t2 = threading.Thread(target=race_condition_worker)
    
    t1.start(); t2.start()
    t1.join(); t2.join()
    
    # Expected: 200,000. Actual: Randomly less than 200,000.
    print(f"Expected 200000, got: {counter} (Data lost due to Race Condition!)\n")


# ============================================================================
# 4. SYNCHRONIZATION PRIMITIVES (LOCKS & RLOCKS)
# ============================================================================

safe_counter = 0
# A Lock can be held by only one thread at a time.
counter_lock = threading.Lock()

def thread_safe_worker():
    global safe_counter
    for _ in range(100_000):
        # The lock ensures that ALL bytecodes within this block execute
        # before another thread can enter.
        with counter_lock: # Context manager automatically calls .acquire() and .release()
            safe_counter += 1

# RLock (Reentrant Lock)
# Standard Locks will DEADLOCK if a thread tries to acquire it twice.
# RLocks allow the SAME thread to acquire it multiple times (useful in recursion).
recursive_lock = threading.RLock()

def recursive_worker(n: int):
    with recursive_lock:
        if n > 0:
            recursive_worker(n - 1)

def demonstrate_locks():
    print("--- 4. Thread Safety with Locks ---")
    global safe_counter
    safe_counter = 0
    
    t1 = threading.Thread(target=thread_safe_worker)
    t2 = threading.Thread(target=thread_safe_worker)
    
    t1.start(); t2.start()
    t1.join(); t2.join()
    
    print(f"Expected 200000, got: {safe_counter} (Data safe thanks to Lock!)\n")


# ============================================================================
# 5. ADVANCED PRIMITIVES: EVENTS AND SEMAPHORES
# ============================================================================

def demonstrate_events_and_semaphores():
    print("--- 5. Events and Semaphores ---")
    
    # ---------------------------------------------------------
    # A. threading.Event
    # A simple boolean flag (True/False) shared between threads.
    # Threads can wait() for the flag to become True.
    # ---------------------------------------------------------
    start_event = threading.Event()
    
    def event_worker(id: int):
        print(f"  Worker {id} waiting for green light...")
        start_event.wait() # Blocks until .set() is called
        print(f"  Worker {id} running!")

    t1 = threading.Thread(target=event_worker, args=(1,))
    t2 = threading.Thread(target=event_worker, args=(2,))
    t1.start(); t2.start()
    
    time.sleep(0.5)
    print("  Main thread setting event flag!")
    start_event.set() # Unblocks all waiting threads
    t1.join(); t2.join()

    # ---------------------------------------------------------
    # B. threading.Semaphore
    # Like a Lock, but allows N threads to enter instead of 1.
    # Excellent for rate-limiting (e.g., max 3 active DB connections).
    # ---------------------------------------------------------
    semaphore = threading.Semaphore(2) # Max 2 threads at a time
    
    def sem_worker(id: int):
        with semaphore:
            print(f"    Semaphore: Thread {id} acquired resource.")
            time.sleep(0.2)
            print(f"    Semaphore: Thread {id} releasing resource.")

    threads = [threading.Thread(target=sem_worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    print()


# ============================================================================
# 6. DAEMON THREADS
# ============================================================================
"""
By default, Python will NOT exit until all threads have finished (join).
A Daemon Thread is a background thread tied to the lifecycle of the main program.
When the main thread exits, all Daemon threads are IMMEDIATELY KILLED, 
regardless of what they are doing (they don't get to run `finally` blocks!).
"""

def daemon_worker():
    while True:
        time.sleep(1)
        # This will abruptly stop printing when the main program finishes
        # print("Daemon running...") 

def demonstrate_daemon():
    print("--- 6. Daemon Threads ---")
    dt = threading.Thread(target=daemon_worker, daemon=True)
    dt.start()
    print("Started daemon thread. It will die quietly when this script ends.\n")


# ============================================================================
# 7. CONCURRENT.FUTURES (THE MODERN WAY)
# ============================================================================
"""
Managing raw `threading.Thread` objects, keeping track of lists, and dealing 
with Exceptions/Return values is tedious.
Python 3.2 introduced `concurrent.futures.ThreadPoolExecutor`.
It provides a pool of reusable threads and the `Future` object design pattern.
"""

def fetch_mock_data(url: str) -> str:
    """Mock network request that returns a string."""
    time.sleep(0.3)
    # If an exception happens here, the ThreadPoolExecutor catches it 
    # and attaches it to the Future object.
    if "bad" in url:
        raise ConnectionError(f"Failed to connect to {url}")
    return f"Data from {url}"

def demonstrate_thread_pools():
    print("--- 7. ThreadPoolExecutor (concurrent.futures) ---")
    
    urls = ["site_A.com", "site_bad.com", "site_B.com", "site_C.com"]
    
    # 'with' ensures the executor cleans up (joins all threads) when done
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        
        # ---------------------------------------------------------
        # METHOD 1: submit() and as_completed()
        # Gives you full control. Yields results AS SOON AS THEY FINISH,
        # out of order.
        # ---------------------------------------------------------
        print("Method 1: .submit() and as_completed()")
        
        # .submit() schedules the callable and returns a Future object instantly.
        future_to_url = {executor.submit(fetch_mock_data, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                # .result() blocks until the thread finishes. 
                # If the thread raised an exception, .result() RE-RAISES it here!
                data = future.result() 
                print(f"  Success: {data}")
            except Exception as exc:
                print(f"  Error on {url}: {exc}")

        print("\nMethod 2: .map()")
        # ---------------------------------------------------------
        # METHOD 2: map()
        # Similar to built-in map(). Returns results IN THE EXACT ORDER 
        # they were submitted, regardless of which finished first.
        # ---------------------------------------------------------
        # We use a safe list of URLs to avoid breaking the map iterator
        safe_urls = ["site_A.com", "site_B.com", "site_C.com"]
        
        results = executor.map(fetch_mock_data, safe_urls)
        # Results is a generator. Calling next() or iterating blocks until ready.
        for result in results:
            print(f"  Mapped Result: {result}")


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # 1 & 2. GIL and Threading utility
    demonstrate_gil_impact()
    
    # 3. Showing why Locks are still needed
    demonstrate_race_condition()
    
    # 4. Fixing race conditions
    demonstrate_locks()
    
    # 5. Advanced Primitives
    demonstrate_events_and_semaphores()
    
    # 6. Background tasks
    demonstrate_daemon()
    
    # 7. The recommended way to do multithreading in modern Python
    demonstrate_thread_pools()
    
    print("\n--- Tutorial Complete ---")
    print("Summary:")
    print("1. Threading in Python is only for I/O bound tasks (Network, File, Sleep).")
    print("2. For CPU bound tasks, use the `multiprocessing` module (bypasses GIL).")
    print("3. Always use ThreadPoolExecutor instead of raw threading.Thread when possible.")
