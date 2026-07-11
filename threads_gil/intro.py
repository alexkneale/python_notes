# ============================================================================
# COMPREHENSIVE PYTHON THREADING, CONCURRENCY & THE GIL MASTERCLASS
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience to Advanced/Senior.
# Topic: Processes vs. Threads, the CPython GIL, Thread Safety, Sync Primitives,
#        Thread-Local Storage, the queue module, Pools, and Python 3.13+ Free-Threading.
# Instructions: This file is fully executable Python. Run it to see all demos.
# ============================================================================

import sys
import os
import time
import threading
import queue
import dis
import concurrent.futures
from typing import List, Dict

# ============================================================================
# 1. FOUNDATIONAL CONCEPTS: PROCESSES VS. THREADS
# ============================================================================
"""
Before diving into code, let's establish a clear mental model of how operating 
systems execute code.

THE PROCESS:
------------
A Process is an isolated running instance of a program. It is the operating system's
primary unit of resource allocation.
  - Memory Isolation: Each process has its own private, isolated virtual address space 
    (heap, stack, global variables). One process CANNOT read or write to another 
    process's memory directly.
  - Resource Ownership: A process owns its file descriptors, sockets, security contexts, 
    and environment variables.
  - Overhead: Creating (spawning) a process is relatively slow and memory-intensive 
    because the OS must allocate a new page table and initialize resources.
  - Inter-Process Communication (IPC): To exchange data between processes, you must use 
    special mechanisms like sockets, pipes, shared memory, or message queues.

THE THREAD:
-----------
A Thread (often called a "lightweight process") is the smallest unit of execution 
that an operating system scheduler can coordinate.
  - Shared Memory: Multiple threads exist INSIDE a single process. All sibling threads 
    share the parent process's memory space. They can read and write to the same global 
    variables, heap-allocated objects, and open file descriptors.
  - Thread-Private Memory: Every thread has its own private execution stack, 
    CPU register state, and program counter (PC) so it knows what instruction it is 
    currently executing.
  - Overhead: Spawning and context-switching threads is much faster and cheaper than 
    processes because no new memory space needs to be allocated.

VISUAL MEMORY MODEL:
-------------------
+-----------------------------------------------------------------------+
|  PROCESS (e.g., CPython Interpreter Instance)                         |
|                                                                       |
|  SHARED MEMORY:                                                       |
|  - Global Variables    - Heap-allocated Objects (e.g., List, Dict)    |
|  - Code Segments      - Open File Descriptors                         |
|                                                                       |
|  +----------------------+  +----------------------+  +-------------+  |
|  | THREAD 1             |  | THREAD 2             |  | THREAD 3    |  |
|  |                      |  |                      |  |             |  |
|  | - Private Registers  |  | - Private Registers  |  | - Registers |  |
|  | - Program Counter    |  | - Program Counter    |  | - PC        |  |
|  | - Private Stack      |  | - Private Stack      |  | - Stack     |  |
|  +----------------------+  +----------------------+  +-------------+  |
+-----------------------------------------------------------------------+

ARE PYTHON THREADS "REAL" OS THREADS?
-------------------------------------
Yes! CPython utilizes the underlying host operating system's native threading APIs:
  - POSIX Threads (pthreads) on Linux and macOS.
  - Windows Threads on Microsoft Windows.
When you spawn a `threading.Thread` in Python, the C interpreter invokes the system 
kernel to create a real operating system thread.
"""


# ============================================================================
# 2. THE GLOBAL INTERPRETER LOCK (GIL) - DEEP DIVE
# ============================================================================
"""
If Python threads are real, native operating system threads, why can't we use 
them to run multiple Python instructions in parallel across multiple CPU cores?

The answer is the Global Interpreter Lock (GIL).

WHAT IS THE GIL?
----------------
The GIL is a mutual exclusion lock (mutex) used by the CPython interpreter (the standard 
reference implementation of Python) to ensure that ONLY ONE OS thread executes Python 
bytecode (instructions) at any single moment.

WHY DOES THE GIL EXIST?
-----------------------
CPython's internal memory management is NOT thread-safe. 
CPython manages memory primarily using REFERENCE COUNTING. Every Python object has an 
internal reference counter. When variables are assigned, passed to functions, or added 
to lists, this counter is incremented. When they go out of scope, it is decremented.

If multiple OS threads were allowed to execute Python bytecode simultaneously:
  - Thread A and Thread B might try to increment/decrement the reference count of the 
    same object at the exact same time.
  - Since standard arithmetic operations are not atomic, this would lead to race conditions.
  - Result: Reference counts could become corrupted, leading to silent memory leaks or 
    disastrous segmentation faults/interpreter crashes.

To prevent this, the GIL was introduced as a simple, highly efficient solution: 
lock the entire interpreter. Only the thread holding the GIL can execute Python code.

THE LOOPHOLE (WHEN IS THE GIL RELEASED?):
----------------------------------------
The GIL is not held constantly. CPython explicitly RELEASES the GIL in these situations:
  1. I/O Operations: Reading/writing files, sending/receiving network data via sockets, 
     or calling database queries.
  2. Operating System Waits: Calling `time.sleep()`.
  3. Cython / C Extensions: Highly optimized scientific libraries (like NumPy, SciPy, or 
     PyTorch) explicitly release the GIL before performing intensive mathematical operations, 
     allowing real multi-core parallel execution!
"""

def demonstrate_switch_interval():
    print("--- Section 2: GIL Switch Interval ---")
    # Python's thread scheduler preemptively forces threads to yield the GIL.
    # The interval at which it forces context switching is defined by sys.getswitchinterval()
    # (By default, this is 0.005 seconds, which is 5 milliseconds).
    interval = sys.getswitchinterval()
    print(f"  CPython current thread switch interval: {interval} seconds ({interval * 1000:.1f}ms)")
    print("  Every 5ms, the interpreter forces the running thread to yield the GIL,")
    print("  giving other waiting threads a chance to acquire it. This is cooperative scheduling.\n")


# ============================================================================
# 3. CPU-BOUND VS. I/O-BOUND (CORRECTING THE SPEEDUP ILLUSION)
# ============================================================================
"""
Let's address the most common misconception in Python threading:
  "Spawning 4 threads for a task makes it 4x faster."

This is only true for I/O-bound tasks. For CPU-bound tasks, it actually makes the 
program SLOWER than a single thread due to the overhead of context switching.

THE FIX FOR THE "NO SPEEDUP" ISSUE:
----------------------------------
To demonstrate a genuine multithreaded speedup, we must compare:
  1. Running N tasks SEQUENTIALLY on a single thread.
  2. Running N tasks CONCURRENTLY using N separate threads.

If we run 4 sleep tasks sequentially, it takes 4x the sleep duration.
If we run 4 sleep tasks in parallel threads, they all sleep at the same time,
releasing the GIL, causing a massive, measurable speedup!
"""

def io_bound_task(task_id: int, duration: float = 0.1):
    """Simulates a network request or database write. Releases the GIL!"""
    # During sleep, this thread yields the GIL so other threads can start sleeping.
    time.sleep(duration)

def run_io_sequential():
    """Runs 4 I/O-bound tasks one after the other on the main thread."""
    for i in range(4):
        io_bound_task(i)

def run_io_threaded():
    """Runs 4 I/O-bound tasks concurrently across 4 threads."""
    threads = []
    for i in range(4):
        t = threading.Thread(target=io_bound_task, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join() # Wait for all threads to finish

def cpu_bound_task():
    """A heavy mathematical task that keeps the CPU core busy."""
    # Keeps GIL held constantly. No yield points!
    count = 0
    for _ in range(3_000_000):
        count += 1

def run_cpu_sequential():
    """Runs 2 CPU-bound tasks sequentially."""
    cpu_bound_task()
    cpu_bound_task()

def run_cpu_threaded():
    """Runs 2 CPU-bound tasks concurrently using threads."""
    t1 = threading.Thread(target=cpu_bound_task)
    t2 = threading.Thread(target=cpu_bound_task)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

def run_cpu_multiprocess():
    """Runs 2 CPU-bound tasks in parallel using MULTIPROCESSING (bypasses GIL)."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cpu_bound_task) for _ in range(2)]
        for f in futures:
            f.result()

def demonstrate_cpu_vs_io():
    print("--- Section 3: CPU vs. I/O Bound Demystified ---")
    
    # -------------------------------------------------------------
    # A. Demonstration of I/O Bound Tasks (The Corrected Speedup Test)
    # -------------------------------------------------------------
    print("A. TESTING I/O-BOUND WORK (time.sleep)...")
    
    start_seq = time.perf_counter()
    run_io_sequential()
    seq_duration = time.perf_counter() - start_seq
    print(f"   [1 Thread]  4 tasks sequentially: {seq_duration:.3f} seconds")
    
    start_thr = time.perf_counter()
    run_io_threaded()
    thr_duration = time.perf_counter() - start_thr
    print(f"   [4 Threads] 4 tasks concurrently: {thr_duration:.3f} seconds")
    
    speedup = seq_duration / thr_duration
    print(f"   -> ACTUAL SPEEDUP: {speedup:.1f}x faster! (Expected: ~4x)\n")
    
    # -------------------------------------------------------------
    # B. Demonstration of CPU Bound Tasks (Proving the GIL)
    # -------------------------------------------------------------
    print("B. TESTING CPU-BOUND WORK (Math/Loops)...")
    
    start_cpu_seq = time.perf_counter()
    run_cpu_sequential()
    cpu_seq_duration = time.perf_counter() - start_cpu_seq
    print(f"   [Sequential]     2 tasks sequentially: {cpu_seq_duration:.3f} seconds")
    
    start_cpu_thr = time.perf_counter()
    run_cpu_threaded()
    cpu_thr_duration = time.perf_counter() - start_cpu_thr
    print(f"   [Multithreaded]  2 tasks concurrently: {cpu_thr_duration:.3f} seconds")
    
    start_cpu_mp = time.perf_counter()
    run_cpu_multiprocess()
    cpu_mp_duration = time.perf_counter() - start_cpu_mp
    print(f"   [Multiprocess]   2 tasks in parallel:  {cpu_mp_duration:.3f} seconds")
    
    print("\n   Key Observations:")
    print("   1. Multithreading did NOT speed up CPU-bound tasks. It was likely slower or similar")
    print("      to sequential execution because both threads fought constantly for the single GIL.")
    print("   2. Multiprocessing bypassed the GIL entirely, utilizing 2 physical CPU cores in parallel,")
    print("      providing a massive speedup!\n")


# ============================================================================
# 4. RACE CONDITIONS & BYTECODE ANALYSIS
# ============================================================================
"""
Wait, if the GIL only allows one thread to execute Python bytecode at a time, 
why do we still have race conditions? 

Because the GIL protects the INTERPRETER'S internal state (reference counts, 
dictionary maps), but it does NOT protect your application's logic!

If an operation requires multiple bytecode instructions to execute, a context 
switch can happen midway through. Let's disassemble a simple increment operation.
"""

def disassemble_increment():
    print("--- Section 4: Bytecode Disassembly ---")
    print("Why `counter += 1` is not thread-safe. It is not an ATOMIC operation:")
    
    def sample_add():
        global counter
        counter += 1
        
    dis.dis(sample_add)
    print("\nNotice that `counter += 1` is split into several separate operations:")
    print("  1. LOAD_GLOBAL (Reads the current value of counter)")
    print("  2. LOAD_CONST  (Loads the value 1)")
    print("  3. INPLACE_ADD (Adds the two values)")
    print("  4. STORE_GLOBAL (Writes the result back to memory)")
    print("\nIf Thread A gets preempted (yields the GIL) right after step 1, Thread B")
    print("can run, perform all 4 steps, and store its new value. When Thread A resumes,")
    print("it still holds the OLD stale value from step 1, overwriting Thread B's update!")
    print("This is a Race Condition.\n")


# Global counter for demonstrating race condition
shared_counter = 0

def race_worker(iterations: int):
    global shared_counter
    for _ in range(iterations):
        # We manually amplify the race condition window by splitting the operation
        curr = shared_counter
        time.sleep(0.000001) # Forces a thread context switch (releases the GIL)
        shared_counter = curr + 1

def demonstrate_race_condition():
    print("--- Section 4: Race Conditions in Action ---")
    global shared_counter
    shared_counter = 0
    iterations = 200
    
    t1 = threading.Thread(target=race_worker, args=(iterations,))
    t2 = threading.Thread(target=race_worker, args=(iterations,))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    expected = iterations * 2
    actual = shared_counter
    lost_updates = expected - actual
    print(f"  Expected counter value: {expected}")
    print(f"  Actual counter value:   {actual}")
    print(f"  Data lost:              {lost_updates} increments (corruption!)\n")


# ============================================================================
# 5. SYNCHRONIZATION PRIMITIVES (LOCK, RLOCK, SEMAPHORE, EVENT, CONDITION, BARRIER)
# ============================================================================
"""
To prevent race conditions, threads must coordinate access to shared state. 
Python's `threading` module provides several powerful synchronization primitives.
"""

# A. threading.Lock (Mutex)
# -------------------------
# A Mutual Exclusion lock. Can only be held by one thread at a time.
# If a thread tries to acquire an already-locked Lock, it blocks until released.
lock_counter = 0
counter_lock = threading.Lock()

def locked_worker(iterations: int):
    global lock_counter
    for _ in range(iterations):
        # The 'with' block automatically calls lock.acquire() on entry 
        # and lock.release() on exit, guaranteeing release even if an exception occurs.
        with counter_lock:
            curr = lock_counter
            time.sleep(0.000001)
            lock_counter = curr + 1

# B. threading.RLock (Reentrant Lock)
# -----------------------------------
# A standard Lock will DEADLOCK if a thread tries to acquire it twice:
#   lock = threading.Lock()
#   lock.acquire()
#   lock.acquire() # Blocks forever!
#
# An RLock (Reentrant Lock) remembers which thread acquired it. The owning thread 
# can acquire it recursively multiple times without blocking. It must be released 
# the same number of times it was acquired.
r_lock = threading.RLock()

def recursive_function(depth: int):
    with r_lock:
        print(f"    RLock acquired at depth: {depth}")
        if depth > 1:
            recursive_function(depth - 1)
        # Automatically releases as we unwind the call stack

# C. threading.Semaphore & BoundedSemaphore
# -----------------------------------------
# Limits concurrent access to a resource. It holds an internal counter.
# Each `.acquire()` decrements the counter; each `.release()` increments it.
# If the counter reaches 0, subsequent threads block until a slot is freed.
# BoundedSemaphore raises a ValueError if you release it more times than acquired.
rate_limit_semaphore = threading.Semaphore(2) # Max 2 concurrent workers

def limited_resource_worker(worker_id: int):
    with rate_limit_semaphore:
        print(f"    [Semaphore] Worker {worker_id} acquired resource slot.")
        time.sleep(0.1)
        print(f"    [Semaphore] Worker {worker_id} releasing resource slot.")

# D. threading.Event
# ------------------
# A simple, thread-safe communication signal. It holds an internal boolean flag (initially False).
# Threads call `.wait()` to block until the flag becomes True.
# Another thread calls `.set()` to turn it True, unblocking all waiting threads.
# Call `.clear()` to reset the flag back to False.
start_gate_event = threading.Event()

def athlete_worker(athlete_id: int):
    print(f"    Athlete {athlete_id} at starting gate. Ready...")
    start_gate_event.wait() # Blocks here until gate opens
    print(f"    Athlete {athlete_id} launched!")

# E. threading.Condition
# ----------------------
# A more advanced signaling primitive. Combines a lock with a wait/notify queue.
# Used when threads need to wait for a specific complex state change.
# Always associated with an underlying Lock (default is an RLock).
shared_buffer: List[int] = []
buffer_condition = threading.Condition()
MAX_BUFFER_SIZE = 3

def condition_producer():
    global shared_buffer
    for i in range(5):
        time.sleep(0.05)
        with buffer_condition:
            while len(shared_buffer) >= MAX_BUFFER_SIZE:
                print("    [Producer] Buffer full. Waiting...")
                buffer_condition.wait() # Releases lock and blocks
            
            shared_buffer.append(i)
            print(f"    [Producer] Added item {i}. Buffer: {shared_buffer}")
            buffer_condition.notify_all() # Wake up waiting consumers

def condition_consumer():
    global shared_buffer
    consumed_items = []
    while len(consumed_items) < 5:
        with buffer_condition:
            while len(shared_buffer) == 0:
                print("    [Consumer] Buffer empty. Waiting...")
                buffer_condition.wait() # Releases lock and blocks
            
            item = shared_buffer.pop(0)
            consumed_items.append(item)
            print(f"    [Consumer] Took item {item}. Buffer: {shared_buffer}")
            buffer_condition.notify_all() # Wake up waiting producers
        time.sleep(0.08)

# F. threading.Barrier
# ---------------------
# A synchronization barrier. Forces a specific number of threads ('parties') 
# to wait until ALL of them have reached the barrier before releasing any of them.
sync_barrier = threading.Barrier(3) # Require 3 threads to sync

def barrier_worker(worker_id: int):
    print(f"    [Barrier] Thread {worker_id} performing initialization...")
    time.sleep(worker_id * 0.05) # Simulate different init times
    print(f"    [Barrier] Thread {worker_id} waiting at checkpoint.")
    
    try:
        # Blocks until 3 threads have called wait()
        sync_barrier.wait()
        print(f"    [Barrier] Thread {worker_id} passed checkpoint! Synchronized work starts.")
    except threading.BrokenBarrierError:
        print(f"    [Barrier] Thread {worker_id} barrier was broken.")


def demonstrate_primitives():
    print("--- Section 5: Synchronization Primitives ---")
    
    # 1. Lock Demo
    global lock_counter
    lock_counter = 0
    iterations = 200
    t1 = threading.Thread(target=locked_worker, args=(iterations,))
    t2 = threading.Thread(target=locked_worker, args=(iterations,))
    t1.start(); t2.start()
    t1.join(); t2.join()
    print(f"  A. Lock: Expected {iterations * 2}, Got: {lock_counter} (Perfect thread safety!)")

    # 2. RLock Demo
    print("\n  B. RLock Recursion Demonstration:")
    recursive_function(3)

    # 3. Semaphore Demo
    print("\n  C. Semaphore Resource Limiting (Max 2 Active):")
    sem_threads = [threading.Thread(target=limited_resource_worker, args=(i,)) for i in range(4)]
    for t in sem_threads: t.start()
    for t in sem_threads: t.join()

    # 4. Event Demo
    print("\n  D. Event Signaling (Starting Gate):")
    athletes = [threading.Thread(target=athlete_worker, args=(i,)) for i in range(3)]
    for t in athletes: t.start()
    time.sleep(0.1)
    print("    [Main] BOOM! Pistol fired.")
    start_gate_event.set() # Release all waiting threads
    for t in athletes: t.join()

    # 5. Condition Demo
    print("\n  E. Condition coordination (Producer-Consumer):")
    prod_thread = threading.Thread(target=condition_producer)
    cons_thread = threading.Thread(target=condition_consumer)
    prod_thread.start()
    cons_thread.start()
    prod_thread.join()
    cons_thread.join()

    # 6. Barrier Demo
    print("\n  F. Barrier Synchronization:")
    barrier_threads = [threading.Thread(target=barrier_worker, args=(i,)) for i in range(3)]
    for t in barrier_threads: t.start()
    for t in barrier_threads: t.join()
    print()


# ============================================================================
# 6. THREAD-LOCAL STORAGE & THE THREAD-SAFE QUEUE
# ============================================================================
"""
Using locks everywhere can make code complex and highly prone to deadlocks. 
Fortunately, there are two powerful architectural patterns to avoid locking:
  1. Thread-Local Storage (Isolating state so threads never share memory).
  2. Thread-Safe Queues (Passing immutable data messages between threads).

THREAD-LOCAL STORAGE (`threading.local`):
----------------------------------------
Sometimes you want global or module-level variables, but you want each thread 
to have its own unique, isolated copy. 
For example: database connections, user session contexts, or transaction IDs.
`threading.local()` dynamically assigns attributes unique to the calling thread.

THE `queue.Queue` MODULE:
-------------------------
A standard Python list `[]` is NOT thread-safe for multi-consumer, multi-producer operations.
If multiple threads append and pop concurrently, internals corrupt.
The `queue.Queue` class implements a thread-safe, locked First-In-First-Out (FIFO) queue.
It handles all internal locking automatically!
"""

# Create a Thread-Local object
thread_local_storage = threading.local()

def show_thread_local_isolation(username: str):
    # This attribute 'username' is only visible to the executing thread.
    thread_local_storage.username = username
    time.sleep(0.05)
    print(f"    [Thread-Local] Core Name: {threading.current_thread().name} -> holding user: {thread_local_storage.username}")

# -------------------------------------------------------------
# Safe Producer-Consumer Pipeline using queue.Queue
# -------------------------------------------------------------
job_queue = queue.Queue(maxsize=5) # Thread-safe queue

def pipeline_producer(id: int):
    for i in range(3):
        item = f"Job-{id}.{i}"
        # .put() is blocking by default. It will pause if queue is full.
        job_queue.put(item)
        print(f"    [Queue Pipeline] Producer {id} put: {item}")
        time.sleep(0.02)

def pipeline_consumer(id: int):
    while True:
        try:
            # .get() blocks until an item is available.
            # timeout prevents waiting forever during shutdown.
            item = job_queue.get(timeout=0.1)
            print(f"    [Queue Pipeline] Consumer {id} processed: {item}")
            # Signaling the queue that the item is processed.
            job_queue.task_done()
        except queue.Empty:
            # Queue is empty and no new items arrived within timeout
            break

def demonstrate_local_and_queue():
    print("--- Section 6: Thread Isolation & Thread-Safe Queue ---")
    
    # 1. Thread Local Storage Demo
    t1 = threading.Thread(target=show_thread_local_isolation, args=("Alice",), name="Worker-A")
    t2 = threading.Thread(target=show_thread_local_isolation, args=("Bob",), name="Worker-B")
    t1.start(); t2.start()
    t1.join(); t2.join()

    # 2. Queue Pipeline Demo
    print("\n  Thread-Safe Queue Producer-Consumer Pipeline:")
    producers = [threading.Thread(target=pipeline_producer, args=(i,)) for i in range(2)]
    consumers = [threading.Thread(target=pipeline_consumer, args=(i,)) for i in range(2)]
    
    for p in producers: p.start()
    for c in consumers: c.start()
    
    for p in producers: p.join()
    
    # Block main thread until all queue items are completed (.task_done() called for each .put())
    job_queue.join()
    
    for c in consumers: c.join()
    print("  Pipeline execution complete.\n")


# ============================================================================
# 7. DAEMON THREADS VS. GRACEFUL SHUTDOWN
# ============================================================================
"""
What is a Daemon Thread?
------------------------
By default, when a Python program starts, it won't terminate as long as there are any 
active "non-daemon" threads running. 
A Daemon Thread is a background service thread that runs in the background. 
When the main thread finishes, all daemon threads are IMMEDIATELY AND ABRUPTLY killed 
by the interpreter, regardless of what they are doing.

THE DAEMON DANGER:
------------------
If a daemon thread is executing a database transaction, writing a critical file, 
or holding system sockets, it is terminated mid-sentence. It does not run `finally` 
blocks, and context manager destructors (`__exit__`) are NOT called!

THE SOLUTION: GRACEFUL SHUTDOWN (Using a coordination Event)
-------------------------------------------------------------
Rather than setting `daemon=True` and letting the OS slaughter your threads, 
you should coordinate a clean shutdown using an event.
"""

def unsafe_daemon_worker():
    try:
        while True:
            print("    [Daemon] Running background cleanup...")
            time.sleep(0.05)
    finally:
        # WARNING: This cleanup block will NEVER execute when the script exits!
        print("    [Daemon] Cleaning up database connections safely...")

# Graceful worker coordinating with an event
shutdown_signal = threading.Event()

def graceful_thread_worker():
    try:
        # Check if the main thread asked us to stop
        while not shutdown_signal.is_set():
            print("    [Graceful] Thread is performing background cycle...")
            time.sleep(0.05)
    finally:
        # This IS guaranteed to execute because we join the thread cleanly
        print("    [Graceful] CLEANUP SUCCESS: Safely closed files and released sockets!")

def demonstrate_lifecycle():
    print("--- Section 7: Daemon Threads & Graceful Shutdown ---")
    
    # Spawn unsafe daemon (set daemon=True)
    dt = threading.Thread(target=unsafe_daemon_worker, daemon=True)
    dt.start()
    time.sleep(0.06)
    print("  [Main] Decided to terminate daemon abruptly...")
    # Thread dt is abandoned here.
    
    # Spawning graceful thread
    print("\n  Spawning graceful thread coordinates with event:")
    gt = threading.Thread(target=graceful_thread_worker)
    gt.start()
    time.sleep(0.12)
    
    print("  [Main] Sending shutdown signal...")
    shutdown_signal.set() # Tell the thread to wrap up
    gt.join() # Cleanly wait for the thread's execution loop to finish
    print("  Graceful thread safely exited.\n")


# ============================================================================
# 8. CONCURRENT.FUTURES & REENTRANT EXCEPTION HANDLING
# ============================================================================
"""
Spawning raw `threading.Thread` objects manually can quickly lead to spaghetti code, 
especially when coordinating thread returns or exception propagation.

`ThreadPoolExecutor` (introduced in Python 3.2 via `concurrent.futures`) is the 
modern, recommended way to handle multi-threading.

KEY ADVANTAGES:
---------------
  1. Thread Reusability: Reuses existing threads in a pool, preventing the system 
     overhead of continuously spawning/destroying threads.
  2. Future Pattern: Returns a `Future` object which represents a promise of an 
     eventual return value or error.
  3. Exception Propagation: If an exception is raised inside a thread, it does 
     not crash the thread silently. The exception is captured and safely re-raised 
     when you call `future.result()`.
"""

def simulated_network_fetch(url: str, delay: float) -> str:
    time.sleep(delay)
    if "error" in url:
        raise ConnectionRefusedError(f"HTTP 500 Connection Refused for {url}")
    return f"Success! Payload from {url}"

def demonstrate_thread_pools():
    print("--- Section 8: ThreadPoolExecutor & Modern Threading ---")
    targets = ["https://site-a.com", "https://site-error.com", "https://site-b.com"]
    
    # Max workers limits maximum concurrent threads.
    # The with block acts as a barrier: it blocks on exit until all threads finish (shutdown)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        print("  Submitting tasks to pool...")
        # .submit schedules the function and returns a Future immediately
        future_map = {executor.submit(simulated_network_fetch, url, 0.1): url for url in targets}
        
        # as_completed yields futures as they finish (out of order!)
        for future in concurrent.futures.as_completed(future_map):
            url = future_map[future]
            print(f"  [Main] Future completed for URL: {url}")
            try:
                # result() yields the return value, OR raises the thread's exception
                result = future.result()
                print(f"    -> Result: {result}")
            except Exception as e:
                print(f"    -> Caught Captured Thread Exception: {type(e).__name__}: {e}")
                
    print("  Thread pool safely cleaned up.\n")


# ============================================================================
# 9. THE PYTHON 3.13+ FREE-THREADING REVOLUTION (PEP 703)
# ============================================================================
"""
Python 3.13 introduced an experimental feature called "Free-Threading" (nogil mode),
defined in PEP 703. This is the biggest architecture update to Python in decades.

WHAT IS FREE-THREADING?
-----------------------
It allows running Python bytecode fully in parallel across multiple cores, on 
multiple threads, in a single process, by completely disabling the GIL.

HOW IS MEMORY SAFE WITHOUT A GIL?
---------------------------------
Free-threaded CPython replaces the GIL's sweeping protection with multiple 
fine-grained synchronization mechanisms inside the interpreter itself:
  1. Biased Locking: Optimized locking for objects accessed mostly by a single thread.
  2. Hazard Pointers: Safe memory reclamation for concurrent structures.
  3. Mimalloc-based Garbage Collector: Highly thread-safe memory allocations.
  4. Immortal Objects: Immutable objects (like True, False, None) have their reference 
     counts locked to prevent multi-threaded race contention.

Let's write a helper to inspect the running environment's GIL status.
"""

def demonstrate_nogil_status():
    print("--- Section 9: Python 3.13+ Free-Threading status ---")
    # Python 3.13 introduced sys._is_gil_enabled()
    is_supported = hasattr(sys, "_is_gil_enabled")
    if is_supported:
        status = sys._is_gil_enabled() # Returns True or False
        print(f"  This runtime environment supports PEP 703 Free-Threading.")
        print(f"  Is CPython GIL currently enabled? {status}")
    else:
        print("  This system is running Python < 3.13. PEP 703 Free-Threading is not supported.")
        print(f"  Interpreter version: {sys.version.split()[0]}. The standard GIL is strictly enabled.")
    print()


# ============================================================================
# 10. MASTER INTERVIEW PREP: 30 SENIOR THREADING & CONCURRENCY Q&As
# ============================================================================
"""
This section is a comprehensive technical interview preparation suite covering 
practical, logical, and architectural threading questions for intermediate to 
senior technical interviews.

================================================================================
Q1: What is the primary difference between a Process and a Thread at the OS level?
A: A Process is an independent execution unit with its own isolated virtual memory,
   owning its file descriptors and heap. A Thread is a lightweight subunit of 
   a process. All threads inside a process share its address space (heap, globals),
   which allows extremely fast communication but introduces race condition risks.

================================================================================
Q2: What is the CPython GIL, and why does it exist?
A: The Global Interpreter Lock (GIL) is a mutex that allows only one thread to 
   execute Python bytecode at a time. It exists because CPython's internal memory 
   management (reference counting) is not thread-safe; locking the entire interpreter 
   avoids complex and slower fine-grained locking schemes for objects.

================================================================================
Q3: Since the GIL ensures only one thread executes bytecode at any instant, why do 
    we still need threading synchronization locks like `threading.Lock`?
A: The GIL protects *CPython's internal interpreter state* (like reference counts),
   but NOT your *application's logical state*. A Python logical statement like `x += 1` 
   is non-atomic and compiles to multiple bytecode steps. Python's scheduler can 
   preempt a thread midway through these bytecode steps, causing race conditions 
   and data loss. A `Lock` is required to make logical actions atomic.

================================================================================
Q4: What are the main release points for the GIL in standard CPython?
A: The GIL is released during:
   1. Blocking I/O operations (network socket reads, file operations, DB queries).
   2. Explicit system delays (e.g. `time.sleep()`).
   3. Within compiled C extensions (e.g. NumPy, PyTorch, cryptography) executing 
      heavy non-Python CPU operations.

================================================================================
Q5: Explain how `sys.getswitchinterval()` relates to thread preemption in CPython.
A: CPython implements cooperative multitasking. A running thread is allowed to hold 
   the GIL and run Python bytecode. Every 5 milliseconds (or the custom interval returned 
   by `sys.getswitchinterval()`), the interpreter triggers a switch signal. The running 
   thread checks this, releases the GIL, and waits. Another waiting thread can then acquire it.

================================================================================
Q6: What is a Race Condition? How do you demonstrate it?
A: A race condition occurs when the correctness of a program depends on the timing 
   or interleaving of thread execution. It can be demonstrated by having multiple threads 
   read, modify, and write to a shared global variable concurrently without lock primitives.

================================================================================
Q7: How does `threading.RLock` (Reentrant Lock) differ from `threading.Lock`?
A: A standard `Lock` can only be acquired once. If the thread currently holding the 
   lock tries to acquire it again, it deadlocks. An `RLock` tracks the thread that 
   currently owns it. That owner thread can recursively acquire the RLock multiple times 
   without blocking. It must release it the same number of times it was acquired.

================================================================================
Q8: What is a Deadlock? How can you avoid it?
A: A deadlock occurs when two or more threads are blocked forever, each waiting for 
   the other to release a lock. For example, Thread 1 holds Lock A and waits for Lock B, 
   while Thread 2 holds Lock B and waits for Lock A. You can prevent deadlocks by:
   1. Acquiring locks in a strict global order.
   2. Utilizing context managers (`with`) to prevent orphaned locks.
   3. Using timeout parameters when acquiring locks (e.g., `lock.acquire(timeout=2.0)`).

================================================================================
Q9: What is a Daemon Thread? What is the principal safety concern when using one?
A: A Daemon thread is a background thread whose lifetime is bound to the main application. 
   When all non-daemon threads exit, the Python interpreter terminates instantly, killing 
   all daemon threads mid-execution. The major safety risk is that daemon threads are terminated 
   abruptly, so `finally` blocks, context manager cleanups, and file flushes are never run, 
   risking severe data corruption or resource leaks.

================================================================================
Q10: How do you gracefully shut down threads without using Daemon flags?
A: Use a thread-safe signaling primitive like `threading.Event`. The worker threads 
   should check `event.is_set()` in their execution loops. To shut down, the main thread 
   sets the event (`event.set()`) and joins each worker thread cleanly (`thread.join()`).

================================================================================
Q11: Explain how `threading.Event` works. What is a typical use case?
A: An `Event` acts as a synchronized boolean gate. It begins as `False`. Calling `.wait()` 
   blocks the calling thread. Calling `.set()` sets it to `True` and awakens all blocked 
   threads. Calling `.clear()` resets it to `False`. Typical use cases include a starting 
   gate coordinate system, pause/resume mechanisms, and graceful termination events.

================================================================================
Q12: What is a Semaphore? How does it differ from a BoundedSemaphore?
A: A Semaphore manages an internal resource counter. Each `acquire()` decrements it, 
   and each `release()` increments it. It limits concurrent access to a resource (like database 
   connection pooling). A `BoundedSemaphore` keeps track of its initial starting limit. If a bug 
   causes a thread to call `release()` too many times, raising the counter above the initial 
   limit, BoundedSemaphore raises a `ValueError` to alert you of the programming error.

================================================================================
Q13: What is a Condition variable, and when should you use it?
A: A `Condition` variable combines a lock with a signaling queue. It is used when a 
   thread must wait for a specific complex state or logic condition to become true (e.g. 
   waiting for items to be added to a buffer). It provides `.wait()`, which releases the 
   associated lock and blocks, and `.notify()`, which wakes up waiting threads.

================================================================================
Q14: Explain the purpose and usage of `threading.Barrier`.
A: A `Barrier` coordinates a fixed number of threads ('parties') to synchronize at a 
   specific execution checkpoint. No thread can proceed past the barrier until all parties 
   have arrived at the barrier. When the final thread arrives, all are released simultaneously.

================================================================================
Q15: What is Thread-Local Storage (`threading.local`)? Provide a practical example.
A: Thread-Local Storage is a mechanism to define variables that are globally accessible, 
   but isolated to the specific thread that created them. A practical example is storing 
   thread-specific database connections, user session contexts, or transaction IDs 
   in a web framework so that individual concurrent requests do not corrupt each other's state.

================================================================================
Q16: Why are standard lists or dicts unsafe for parallel modification, even with GIL?
A: While individual C-level list operations (like appending) are technically atomic, 
   logical patterns (like "check-then-act" or iterating while mutating) are not. 
   If Thread A iterates over a dictionary while Thread B mutates it, Python will raise 
   a `RuntimeError: dictionary changed size during iteration`.

================================================================================
Q17: How does `queue.Queue` handle thread safety under the hood?
A: `queue.Queue` encapsulates an internal double-ended list, a lock, and several 
   `Condition` variables (e.g., `not_empty`, `not_full`, `all_tasks_done`). It handles 
   all mutual exclusion and thread synchronization automatically, providing a simple, 
   guaranteed thread-safe FIFO pipeline.

================================================================================
Q18: What is the differences between `submit()` and `map()` in ThreadPoolExecutor?
A: 
   - `submit(fn, *args)` schedules a single callable and returns a `Future` object 
     immediately. It gives you maximum control and allows you to catch exceptions.
   - `map(fn, *iterables)` is an asynchronous equivalent to built-in `map()`. It executes 
     the function on all elements in parallel and yields results as a generator in 
     the exact order they were submitted.

================================================================================
Q19: How are exceptions handled in threads managed by ThreadPoolExecutor?
A: When a thread raises an exception in a `ThreadPoolExecutor`, the thread catches it 
   internally and stores it inside the representing `Future` object. The exception 
   remains silent until the user calls `future.result()` or `future.exception()`, at 
   which point it is re-raised on the main thread.

================================================================================
Q20: Why is a Thread Pool preferred over spawning raw `threading.Thread` objects?
A: Creating raw threads is expensive because it involves kernel-level system calls 
   and stack allocations. Spawning 1000 raw threads will quickly crash the process 
   or slow down the system due to thrashing. A Thread Pool keeps a fixed number of 
   reusable worker threads, avoiding creation overhead and safely bounding resource usage.

================================================================================
Q21: How does Multiprocessing bypass the GIL?
A: Multiprocessing spawns completely separate operating system processes rather than 
   threads. Each spawned process runs its own completely independent CPython interpreter 
   instance, containing its own independent heap space, memory allocations, and its own 
   private GIL. This allows multiple cores to execute Python code in true parallel.

================================================================================
Q22: What is PEP 703 (Free-Threading) and how does it affect Python 3.13+?
A: PEP 703 is a major standard change that enables CPython to compile without the 
   Global Interpreter Lock (GIL). It allows genuine, multi-core parallel thread 
   execution inside a single process, making Python multithreading highly efficient 
   for CPU-bound tasks.

================================================================================
Q23: How does free-threaded Python ensure thread safety of its internals without the GIL?
A: CPython replaces the central GIL lock with highly advanced local concurrency structures:
   1. Biased Locking (objects are locked locally on a per-thread basis to avoid global lock bus contention).
   2. Hazard Pointers (safe garbage collection of active memory segments).
   3. Thread-safe Mimalloc memory allocation.
   4. Immortal Objects (locks reference counts on critical singletons).

================================================================================
Q24: How can you check if the GIL is enabled at runtime in Python 3.13?
A: You can inspect `sys._is_gil_enabled()` (which is defined in compiled builds 
   supporting Free-Threading). If it returns `True`, the GIL is active. If `False`, 
   the runtime is free-threaded.

================================================================================
Q25: Why do libraries like NumPy and PyTorch not suffer from the GIL for heavy work?
A: These libraries are written in C, C++, or Fortran. Before entering intensive numerical 
   calculations, they call CPython API functions like `Py_BEGIN_ALLOW_THREADS`, which releases 
   the GIL. They perform calculations in true multi-core parallel, and re-acquire the GIL 
   via `Py_END_ALLOW_THREADS` only when returning Python objects to the interpreter.

================================================================================
Q26: What is a context manager `with` lock benefit?
A: It provides RAII (Resource Acquisition Is Initialization). It guarantees that 
   the lock is released (`lock.release()`) on exiting the block, even if an exception 
   is raised, avoiding orphaned lock deadlocks.

================================================================================
Q27: What is thread Starvation?
A: Starvation occurs when a thread is perpetually denied necessary CPU scheduling 
   or resources because other threads are prioritized or holding locks indefinitely.

================================================================================
Q28: What is a Livelock? How does it differ from a Deadlock?
A: In a deadlock, threads are physically blocked waiting on locks. In a Livelock, 
   threads are active, changing their states in response to each other, but making 
   no actual forward progress. Think of two polite people repeatedly shifting sides in 
   a hallway to let each other pass, but blocking each other continuously.

================================================================================
Q29: Explain the difference between Concurrency and Parallelism.
A: 
   - Concurrency is about *structure*: managing and coordinating multiple tasks 
     (e.g., interleaving tasks on a single thread). Doing many things at once.
   - Parallelism is about *execution*: running multiple tasks at the physical same 
     instant on separate hardware CPU cores. Doing many things at the same time.

================================================================================
Q30: Why is `queue.Queue` preferred over sharing a global list in thread designs?
A: Spawning queues keeps code cleanly decoupled. Threads exchange messages instead 
   of directly editing a single global structure. This eliminates complex lock 
   orchestrations, prevents synchronization deadlocks, and simplifies testing.
"""


# ============================================================================
# 11. CENTRAL EXECUTION BLOCK (RUNNING THE DEMONSTRATIONS)
# ============================================================================

def main():
    """
    Orchestrates the entire interactive masterclass.
    """
    print("=====================================================================")
    print("      CPYTHON THREADING, CONCURRENCY & THE GIL MASTERCLASS           ")
    print("=====================================================================\n")
    
    # 1. Section 2
    demonstrate_switch_interval()
    
    # 2. Section 3
    demonstrate_cpu_vs_io()
    
    # 3. Section 4
    disassemble_increment()
    demonstrate_race_condition()
    
    # 4. Section 5
    demonstrate_primitives()
    
    # 5. Section 6
    demonstrate_local_and_queue()
    
    # 6. Section 7
    demonstrate_lifecycle()
    
    # 7. Section 8
    demonstrate_thread_pools()
    
    # 8. Section 9
    demonstrate_nogil_status()
    
    print("=====================================================================")
    print("                  MASTERCLASS EXECUTION COMPLETE                    ")
    print("=====================================================================")


if __name__ == "__main__":
    main()
