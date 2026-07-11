# ============================================================================
# BRIDGING COROUTINES AND THREADS: MULTITHREADED ASYNCIO IN PRACTICE
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience to Advanced/Senior.
# Topic: Bridging threads_gil and coroutines_asyncio. Combining Cooperative
#        Multitasking (async/await) with Preemptive Multitasking (OS Threads).
# Instructions: This file is fully executable. Run it to see all hybrid demos.
# ============================================================================

import asyncio
import concurrent.futures
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Coroutine, List

# ============================================================================
# ARCHITECTURAL COMPARISON: PREEMPTIVE VS. COOPERATIVE
# ============================================================================
"""
Before combining threads and coroutines, we must understand how they differ 
and why they are highly complementary when bridged correctly.

+------------------------+--------------------------------------------------+---------------------------------------------------+
| Feature                | Multithreading (threads_gil)                     | Coroutines (coroutines_asyncio)                   |
+------------------------+--------------------------------------------------+---------------------------------------------------+
| Model                  | Preemptive Multitasking (OS schedules switches)  | Cooperative Multitasking (Tasks yield control)    |
| Hardware Utilization   | 1 CPU core (due to CPython GIL) *                | 1 CPU core (single-threaded Event Loop)           |
| Memory Overhead        | Medium (~8MB stack per thread, OS managed)       | Extremely Low (~1KB per Coroutine, heap-allocated)|
| Context Switch Speed   | Slow (requires OS Kernel context switches)       | Ultra-fast (pure user-space function suspension)  |
| Max Concurrency        | Hundreds (limited by thread memory/OS limits)    | Tens of thousands of active concurrent tasks      |
| Best For               | Offloading blocking I/O or GIL-released C work   | High-concurrency network I/O, web sockets, APIs   |
| Code Synchronization   | Complex (requires Locks, Semaphores, Conditions) | Simple (no locks needed for pure async state)     |
+------------------------+--------------------------------------------------+---------------------------------------------------+
* Note: Except in Python 3.13+ with experimental GIL-disabled Free-Threading.

THE ARCHITECTURAL BRIDGE: WHY AND WHEN TO COMBINE BOTH?
------------------------------------------------------
In a modern backend application, you rarely have a "pure" environment. You will face:
  1. High-concurrency network I/O (ideal for Coroutines/Asyncio).
  2. Legacy synchronous database drivers (PostgreSQL/MySQL without async, SQLAlchemy, Django ORM).
  3. Blocking OS-level operations (heavy file read/write, DNS resolution, subprocesses).
  4. Heavy CPU computations (resizing images, parsing large JSONs, cryptography).

If you run synchronous blocking tasks inside your Asyncio Event Loop, the single thread 
freezes, and your high-concurrency server grinds to a halt!

The solution is to build a HYBRID ARCHITECTURE:
- Use the main thread as a high-speed Asyncio coordinator.
- Offload blocking, synchronous, or CPU-intensive work to a Thread Pool.
- Run dedicated background threads to monitor legacy systems, communicating thread-safely 
  with the Asyncio Event Loop.

VISUAL MEMORY & EXECUTION BRIDGE:
---------------------------------
                  +----------------------------------------------+
                  |                 MAIN PROCESS                 |
                  +----------------------------------------------+
                                         |
                +------------------------+------------------------+
                |                                                 |
      +-------------------+                             +-------------------+
      |   MAIN THREAD     |                             | BACKGROUND THREAD |
      | (Runs Event Loop) |                             | (Sync Exec Loop)  |
      +-------------------+                             +-------------------+
      | - Coroutines      |                             | - Blocking I/O    |
      | - Non-blocking IO |                             | - CPU calculations|
      | - High Concurrency|                             | - Legacy Sync DB  |
      +-------------------+                             +-------------------+
                |                                                 |
                | <=========== asyncio.to_thread() =============  | (Offload blocking work)
                |                                                 |
                | =========== asyncio.run_coroutine_threadsafe() ==> (Submit async work)
                |                                                 |
                | =========== loop.call_soon_threadsafe() =========> (Schedule sync callback)
"""


# ============================================================================
# COOPERATIVE VS. PREEMPTIVE ORCHESTRATION: 
# THE HIERARCHY OF PROCESSES, THREADS, EVENT LOOPS, AND TASKS
# ============================================================================
"""
To master advanced Python concurrency, we must understand the containment 
hierarchy and scheduling models of four critical abstractions: Processes, 
Threads, Event Loops, and Tasks.

1. PROCESS (Operating System Isolation Level)
   - Scope: A process is the OS-level container that owns resources (private virtual 
     memory address space, standard file descriptors, socket tables).
   - Python-specifics: Standard CPython executes code inside a process. Each process 
     carries its own independent Global Interpreter Lock (GIL) and its own garbage collector.
   - Concurrency: Processes are fully isolated. To communicate, they must use 
     IPC (pipes, sockets, or shared memory), which carries serialization (pickling) overhead.
   - Scaling: Ideal for CPU-bound tasks, as separate processes run on different CPU cores in parallel.

2. THREAD (Operating System Execution Level)
   - Scope: A Thread is a lightweight, independent flow of execution scheduled by the OS Kernel.
   - Python-specifics: Threads live INSIDE a single process and share all of its memory 
     (global variables, heap). However, standard CPython threads must acquire the 
     Global Interpreter Lock (GIL) to execute Python bytecode.
   - Concurrency: Because they share memory, threads can read/write data with zero serialization 
     overhead. This makes them highly prone to logical race conditions, requiring synchronization 
     primitives (Locks, Semaphores).
   - Scaling: Ideal for blocking I/O (where the GIL is released during waits), but completely 
     bottlenecked for Python CPU-bound tasks.

3. EVENT LOOP (Application/Thread Scheduling Level)
   - Scope: An Event Loop is an infinite control loop that runs inside a *single* Thread. 
     Its job is to monitor system socket/file descriptors (via OS-level multiplexing APIs 
     like select, epoll, or kqueue) and coordinate the execution of concurrent sub-routines.
   - Concurrency: It runs cooperative multitasking. It keeps a queue of active jobs and executes 
     them sequentially on that single thread.
   - Scaling: It is extremely lightweight, fast, and does not require OS thread context-switching.

4. TASK / COROUTINE (User-Space Logical Level)
   - Scope: A Coroutine is a function defined with `async def` that can suspend execution. 
     A Task wraps a coroutine and registers it with the Event Loop, allowing it to execute 
     concurrently.
   - Concurrency: Scheduled cooperatively. A Task runs on the Event Loop thread until it 
     reaches an `await` statement, which explicitly yields control back to the Event Loop, 
     saying: "I am waiting for this external resource, go run other Tasks in the meantime."
   - Scaling: Highly scalable. You can run hundreds of thousands of Tasks on a single Event 
     Loop thread because there is no thread overhead (each Task uses only ~1-2KB of memory).


THE CONCURRENCY NESTING MODEL
-----------------------------
+----------------------------------------------------------------------------------------+
| PROCESS (Private Virtual Memory, Private GIL, OS Sandbox)                              |
|                                                                                        |
|   +----------------------------------------------------------------------------------+ |
|   | THREAD (Shares Process Heap/Globals, Scheduled by Kernel, Carries GIL)           | |
|   |                                                                                  | |
|   |   +----------------------------------------------------------------------------+ | |
|   |   | EVENT LOOP (Control Loop, Single-Threaded Event Multiplexer)               | | |
|   |   |                                                                            | | |
|   |   |   +----------------------------------------------------------------------+ | | |
|   |   |   | TASK (Cooperatively scheduled, yields on `await` - ~2KB size)        | | | |
|   |   |   +----------------------------------------------------------------------+ | | | |
|   |   |   | TASK (Currently running coroutine...)                                | | | |
|   |   |   +----------------------------------------------------------------------+ | | | |
|   |   |   | TASK (Waiting for Event Loop slot...)                                | | | |
|   |   |   +----------------------------------------------------------------------+ | | |
|   |   +----------------------------------------------------------------------------+ | |
|   +----------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------+


HOW THEY ALL WORK TOGETHER: ORCHESTRATION IN PRACTICE
----------------------------------------------------
In a production-grade backend, we don't pick just one; we coordinate all of them:

- Tasks handle High-Concurrency Network Connections: The Event Loop schedules thousands of async tasks 
  for active API/WebSocket connections.
- Threads handle Synchronous I/O: If an async task needs to read a legacy database, it calls 
  `asyncio.to_thread()`, spinning up a background Thread to wait on the synchronous socket, unblocking 
  the event loop.
- Processes handle Heavy CPU Math: If an async task needs to run machine learning inferences or heavy math, 
  it offloads to a separate Process using `loop.run_in_executor()` with a `ProcessPoolExecutor`, 
  bypassing the GIL entirely.
"""

# --- HELPERS FOR THE CONCURRENCY HIERARCHY DEMO ---

def heavy_cpu_math(n: int) -> int:
    """A heavy CPU task that runs in a Process to bypass the GIL."""
    import os
    print(f"      [Sub-Process {os.getpid()} | Thread {threading.get_ident()}] Process started CPU calculations...")
    # Heavy math loop
    total = sum(i * i for i in range(n))
    print(f"      [Sub-Process {os.getpid()} | Thread {threading.get_ident()}] Process finished. Sum of squares calculated.")
    return total

def blocking_thread_task(delay: float) -> str:
    """A blocking sync task that runs in a Thread, unblocking the event loop."""
    import os
    print(f"      [Sync Thread {threading.get_ident()} | Process {os.getpid()}] Thread started blocking database/file write...")
    time.sleep(delay)
    print(f"      [Sync Thread {threading.get_ident()} | Process {os.getpid()}] Thread finished blocking write.")
    return "Write_Success"

async def async_event_loop_task(task_id: int):
    """An async task running directly on the event loop."""
    import os
    print(f"      [Async Task {task_id} | Loop Thread {threading.get_ident()} | Process {os.getpid()}] Async Task starting...")
    await asyncio.sleep(0.15)
    print(f"      [Async Task {task_id} | Loop Thread {threading.get_ident()} | Process {os.getpid()}] Async Task finished.")

async def demonstrate_concurrency_hierarchy():
    print("--- 0. Concurrency Hierarchy (Loops, Threads, Tasks, Processes) Working Together ---")
    import os
    print(f"  [Main Application Process {os.getpid()} | Thread {threading.get_ident()}]")
    print("  Initializing hybrid loop, thread, and process orchestrator...")
    
    loop = asyncio.get_running_loop()
    
    # 1. Start an asynchronous task on the Event Loop
    print("\n  [Action 1] Scheduling concurrent async Tasks on the Event Loop...")
    async_task = asyncio.create_task(async_event_loop_task(1))
    
    # 2. Offload a blocking synchronous task to a Thread Pool
    # We use a ThreadPoolExecutor to handle legacy I/O
    print("\n  [Action 2] Offloading blocking synchronous task to a Thread Pool...")
    thread_future = loop.run_in_executor(None, blocking_thread_task, 0.3)
    
    # 3. Offload a heavy CPU-bound task to a Process Pool
    # We use a ProcessPoolExecutor to run CPU math on a separate core, bypassing the GIL
    print("\n  [Action 3] Offloading heavy CPU task to a Process Pool (Bypassing GIL)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as process_executor:
        process_future = loop.run_in_executor(process_executor, heavy_cpu_math, 5_000_000)
        
        # 4. We can await all three concurrent abstractions simultaneously from our event loop!
        print("\n  [Action 4] Awaiting all concurrency layers to complete concurrently...")
        start_time = time.perf_counter()
        
        # Wait for everything
        _, thread_result, process_result = await asyncio.gather(
            async_task, 
            thread_future, 
            process_future
        )
        
        print(f"\n  All layers resolved in {time.perf_counter() - start_time:.3f} seconds!")
        print(f"  -> Thread Result: {thread_result}")
        print(f"  -> Process Result (Math sum): {process_result}")
        print()


# ============================================================================
# SECTION 1: THE ASYNC-TO-SYNC BRIDGE (OFFLOADING BLOCKING WORK)
# ============================================================================
"""
If a coroutine calls a blocking function (e.g., `requests.get()`, `time.sleep()`), 
it blocks the Event Loop thread. 

To bridge this gap, CPython provides two primary mechanisms to execute sync code 
in a separate thread while allowing the event loop to continue running other tasks:

1. `asyncio.to_thread(func, *args, **kwargs)` (Python 3.9+):
   - A high-level convenience function.
   - Automatically offloads the function to a default, global ThreadPoolExecutor.
   - Safe, clean, and recommended for most standard cases.

2. `loop.run_in_executor(executor, func, *args)` (Lower-level):
   - Allows passing a custom `ThreadPoolExecutor` (or `ProcessPoolExecutor` for CPU-heavy tasks).
   - Essential when you need to limit the size of the thread pool (e.g., to match database connection limits).
"""

# Simulating a legacy blocking database query or file API
def blocking_db_query(query_id: int) -> str:
    print(f"    [Thread {threading.get_ident()}] DB Query {query_id} started (blocking)...")
    time.sleep(0.3)  # Standard blocking sleep (yields GIL, but blocks thread)
    print(f"    [Thread {threading.get_ident()}] DB Query {query_id} finished.")
    return f"Data_Result_{query_id}"

async def background_async_ticker():
    """A background coroutine that proves the Event Loop is NOT frozen."""
    try:
        while True:
            print("  [Async Ticker] Running smoothly...")
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("  [Async Ticker] Stopped.")

async def demonstrate_offloading():
    print("--- 1. Demonstrating Async-to-Sync Bridge (Offloading) ---")
    
    # Start a background async ticker to monitor event loop liveness
    ticker_task = asyncio.create_task(background_async_ticker())
    await asyncio.sleep(0.05)  # Let it tick once
    
    # --- A. THE WRONG WAY: Calling blocking code directly ---
    print("\n[A] Executing blocking query directly on the Event Loop thread...")
    start_time = time.perf_counter()
    # This blocks the entire event loop. The ticker will FREEZE!
    blocking_db_query(999)
    print(f"Direct Execution Time: {time.perf_counter() - start_time:.3f}s (Notice: Ticker was frozen!)")
    
    await asyncio.sleep(0.1)  # Let the ticker catch up

    # --- B. THE RIGHT WAY (HIGH-LEVEL): asyncio.to_thread ---
    print("\n[B] Offloading blocking query via asyncio.to_thread...")
    start_time = time.perf_counter()
    # to_thread immediately schedules the task on a background thread and awaits its completion.
    # The event loop thread is freed, so our async ticker continues running concurrently!
    result = await asyncio.to_thread(blocking_db_query, 101)
    print(f"to_thread Result: {result}")
    print(f"to_thread Time: {time.perf_counter() - start_time:.3f}s (Notice: Ticker kept ticking!)")

    await asyncio.sleep(0.1)

    # --- C. THE CUSTOM WAY (LOW-LEVEL): loop.run_in_executor ---
    print("\n[C] Offloading blocking queries to a custom ThreadPoolExecutor...")
    start_time = time.perf_counter()
    
    # We create a custom executor with a limit of 2 concurrent threads
    custom_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_running_loop()
    
    # Schedule multiple blocking queries concurrently in our pool
    # Passing None to run_in_executor uses the default executor, but here we pass our custom_executor.
    futures = [
        loop.run_in_executor(custom_executor, blocking_db_query, i)
        for i in range(1, 5)
    ]
    
    results = await asyncio.gather(*futures)
    print(f"Custom Executor Results: {results}")
    print(f"Custom Executor Time: {time.perf_counter() - start_time:.3f}s (4 queries executed via 2 threads)")
    
    # Clean up
    custom_executor.shutdown()
    ticker_task.cancel()
    await asyncio.gather(ticker_task, return_exceptions=True)
    print()


# ============================================================================
# SECTION 2: THE SYNC-TO-ASYNC BRIDGE (SUBMITTING WORK TO AN EVENT LOOP)
# ============================================================================
"""
What if your main application is synchronous, but you need to spawn a background 
asynchronous execution environment? 
Common scenarios:
  - A desktop GUI app (Tkinter/PyQt) where the GUI runs on the main thread, 
    but you want an Asyncio event loop running on a background thread to handle network WebSockets.
  - A legacy web app (Flask/Django) that needs to run concurrent async tasks in the background.

To do this, we run an Event Loop *inside* a separate thread and keep it running 
forever using `loop.run_forever()`.

To thread-safely interact with this running loop from our synchronous threads, 
we MUST use the thread-safe API:

1. `asyncio.run_coroutine_threadsafe(coro, loop)`:
   - Submits a coroutine to the specified loop from a separate thread.
   - Thread-safe.
   - Returns a `concurrent.futures.Future` which the sync thread can block on 
     or add callbacks to.

2. `loop.call_soon_threadsafe(callback, *args)`:
   - Schedules a synchronous callback to be executed in the event loop thread 
     at the next opportunity.
"""

# A background thread loop target
def start_background_event_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    print(f"    [Loop Thread {threading.get_ident()}] Event loop is starting...")
    loop.run_forever()  # Keeps thread active, processing events in a loop
    print(f"    [Loop Thread {threading.get_ident()}] Event loop has stopped.")

async def async_worker_task(task_id: int, duration: float) -> str:
    """An asynchronous task running inside our background event loop."""
    print(f"      [Async Loop] Worker {task_id} starting...")
    await asyncio.sleep(duration)
    print(f"      [Async Loop] Worker {task_id} completed!")
    return f"Success_Value_{task_id}"

def synchronous_callback(value: str):
    """A synchronous callback executed thread-safely inside the loop thread."""
    print(f"      [Sync Callback in Loop Thread {threading.get_ident()}] Processed value: {value}")

def demonstrate_background_loop():
    print("--- 2. Demonstrating Sync-to-Async Bridge (Background Loop) ---")
    print(f"  [Main Thread {threading.get_ident()}] Starting demo...")
    
    # 1. Create an isolated event loop
    bg_loop = asyncio.new_event_loop()
    
    # 2. Spawn a daemon thread to run our loop forever
    bg_thread = threading.Thread(
        target=start_background_event_loop, 
        args=(bg_loop,), 
        daemon=True,
        name="AsyncioBgThread"
    )
    bg_thread.start()
    
    # Wait briefly to ensure loop is fully running
    time.sleep(0.1)
    
    # 3. Schedule async coroutines from our synchronous main thread!
    print(f"\n  [Main Thread] Scheduling async_worker_task(101) thread-safely...")
    # asyncio.run_coroutine_threadsafe return a concurrent.futures.Future
    future1 = asyncio.run_coroutine_threadsafe(async_worker_task(101, 0.2), bg_loop)
    
    print(f"  [Main Thread] Scheduling async_worker_task(102) thread-safely...")
    future2 = asyncio.run_coroutine_threadsafe(async_worker_task(102, 0.1), bg_loop)
    
    # 4. We can block on the futures to get the result from our synchronous thread!
    print("  [Main Thread] Waiting for results (blocking on futures)...")
    res2 = future2.result(timeout=1.0)  # Blocks sync thread until future2 completes
    print(f"  [Main Thread] Received Result 2: {res2}")
    
    res1 = future1.result(timeout=1.0)
    print(f"  [Main Thread] Received Result 1: {res1}")
    
    # 5. Schedule a synchronous callback in the background loop thread-safely
    print(f"\n  [Main Thread] Scheduling sync callback via call_soon_threadsafe...")
    bg_loop.call_soon_threadsafe(synchronous_callback, "Hello Threadsafe!")
    time.sleep(0.1)  # Let callback execute
    
    # 6. Graceful Shutdown of the background loop
    print("\n  [Main Thread] Shutting down background event loop...")
    
    # We must stop the loop thread-safely! 
    # Directly calling bg_loop.stop() from another thread is a race condition.
    bg_loop.call_soon_threadsafe(bg_loop.stop)
    
    # Join the thread to wait for it to exit
    bg_thread.join(timeout=1.0)
    
    # Clean up/close the loop
    bg_loop.close()
    print("  [Main Thread] Background loop shut down successfully.\n")


# ============================================================================
# SECTION 3: THE SILENT TRAPS (THREAD-SAFETY IN HYBRID PROGRAMS)
# ============================================================================
"""
⚠️ WARNING: CRITICAL THREAD-SAFETY RULES IN ASYNCIO

Beginners often make the mistake of sharing standard Asyncio primitives across 
threads. This will lead to catastrophic bugs, memory corruption, and random hangs!

Rule 1: ALMOST ALL ASYNCIO OBJECTS ARE NOT THREAD-SAFE!
   - `asyncio.Queue`, `asyncio.Event`, `asyncio.Lock`, `asyncio.Semaphore` are designed 
     to be used *only* inside a single event loop thread. 
   - They do not use operating system mutexes; they rely on the cooperatively scheduled 
     nature of a single-threaded loop.
   - If a background thread calls `asyncio_queue.put_nowait(item)`, the internal state 
     of the queue will get corrupted because the event loop thread might be reading 
     or modifying it simultaneously!

Rule 2: How to safely share queues between threads and asyncio?
   - If you need a FIFO queue to bridge a synchronous thread and an asyncio event loop:
     - DO NOT use `asyncio.Queue` (not thread-safe).
     - DO NOT use `queue.Queue` inside the async loop directly (its `.get()` blocks the thread, freezing the loop!).
     
   The Solution:
   - Use a thread-safe `queue.Queue`, but when reading from it in Asyncio, you must 
     offload the blocking `.get()` call to a thread via `asyncio.to_thread()`.
   - Alternatively, use specialized wrappers or pass messages using `asyncio.run_coroutine_threadsafe`.
"""

# Let's demonstrate the correct thread-safe pipeline bridge
class ThreadAsyncBridgeQueue:
    """
    A bidirectional thread-safe queue bridge.
    Allows synchronous threads to put items, and an asyncio event loop to get items 
    without blocking the loop.
    """
    def __init__(self):
        self._sync_queue = queue.Queue()

    def put_from_thread(self, item: Any):
        """Called by synchronous threads to add items."""
        self._sync_queue.put(item)

    async def get_in_async(self) -> Any:
        """Called by coroutines in the event loop to read items without freezing."""
        # We offload the blocking self._sync_queue.get() to a thread pool!
        return await asyncio.to_thread(self._sync_queue.get)

    def task_done_from_thread(self):
        self._sync_queue.task_done()


# ============================================================================
# SECTION 4: PRACTICAL END-TO-END APPLICATION (HYBRID BACKEND PIPELINE)
# ============================================================================
"""
Let's build a practical, real-world simulation of a hybrid backend service.

SCENARIO:
We are building a highly-scalable Image Processing & Analytics Backend.
1. The Core Server runs on an Asyncio event loop (handling incoming API connection requests).
2. Incoming image processing requests are received asynchronously.
3. Image processing (heavy CPU/IO, legacy PIL/OpenCV sync library) is offloaded to a ThreadPoolExecutor.
4. A legacy external Message Broker (like RabbitMQ) is monitored by a separate background thread.
   When messages arrive on the broker, they are forwarded thread-safely back to our Asyncio Event Loop 
   to trigger notifications to active clients.
5. Implement a robust Graceful Shutdown system that cleanly stops the event loop, 
   the background threads, and drains the thread pool executors.
"""

class HybridBackendService:
    def __init__(self):
        self.loop = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="WorkerPool")
        self.shutdown_event = threading.Event()
        self.active_tasks = []
        self.service_thread = None

    # --- SIMULATED SYNC BLOCKING DB / IMAGE PROCESSING WORK ---
    def process_image_sync(self, request_id: str, image_size: str) -> str:
        """Simulates CPU-heavy sync image processing (e.g. OpenCV, Pillow)"""
        print(f"      [ThreadPool] Thread {threading.current_thread().name} processing image {request_id} ({image_size})...")
        time.sleep(0.4) # Simulate processing time (GIL released or blocked depending on task)
        print(f"      [ThreadPool] Thread {threading.current_thread().name} completed image {request_id}.")
        return f"Processed_{request_id}_at_{image_size}"

    # --- SIMULATED EXTERNAL BROKER LISTENER (SYNC BACKGROUND THREAD) ---
    def legacy_broker_listener_thread(self):
        """Simulates a background thread listening to an external MQ (RabbitMQ, Kafka)"""
        print(f"  [MQ Thread] Started listening to external broker...")
        msg_count = 0
        while not self.shutdown_event.is_set():
            try:
                # Mock receiving a message from MQ every 0.3s
                time.sleep(0.3)
                if self.shutdown_event.is_set():
                    break
                msg_count += 1
                payload = f"External MQ Event #{msg_count}"
                print(f"  [MQ Thread] Received broker message: '{payload}'")
                
                # We MUST push this message thread-safely back to our active Asyncio Event Loop!
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.handle_incoming_broker_notification(payload), 
                        self.loop
                    )
            except Exception as e:
                print(f"  [MQ Thread] Error: {e}")
        print("  [MQ Thread] Shutting down cleanly.")

    # --- ASYNC EVENT HANDLERS ---
    async def handle_incoming_broker_notification(self, payload: str):
        """Coroutine executed inside the main event loop when MQ message is received."""
        print(f"    [Async Event Loop] Routing MQ message to active websocket clients: {payload}")
        # In a real app, you would broadcast this to open websockets
        await asyncio.sleep(0.01)

    async def handle_api_request(self, request_id: str, image_size: str):
        """Handles an incoming HTTP API request asynchronously."""
        print(f"  [Async Server] Received API upload request: {request_id}")
        
        # We offload the blocking image processing to our Thread Pool
        # loop.run_in_executor allows us to specify our dedicated backend executor pool.
        print(f"  [Async Server] Offloading processing for {request_id} to Thread Pool...")
        
        # run_in_executor(executor, func, *args)
        processed_result = await self.loop.run_in_executor(
            self.executor, 
            self.process_image_sync, 
            request_id, 
            image_size
        )
        
        print(f"  [Async Server] API request {request_id} finished processing! Result: {processed_result}")

    # --- CORE WORKER PIPELINE ---
    async def run_main_pipeline(self):
        self.loop = asyncio.get_running_loop()
        print("  [Async Server] Core API event loop is running.")
        
        # Start the background MQ listener thread
        self.service_thread = threading.Thread(target=self.legacy_broker_listener_thread, daemon=True)
        self.service_thread.start()

        # Simulate receiving API requests concurrently
        # Standard asyncio task creation to run things concurrently
        task1 = asyncio.create_task(self.handle_api_request("Req_A", "1920x1080"))
        task2 = asyncio.create_task(self.handle_api_request("Req_B", "4K_Ultra"))
        
        await asyncio.sleep(0.2) # Wait a bit
        task3 = asyncio.create_task(self.handle_api_request("Req_C", "1024x768"))
        
        # Gather all client API request tasks to complete
        await asyncio.gather(task1, task2, task3)
        
        # Allow MQ messages to flow for a few moments
        await asyncio.sleep(0.6)

    def shutdown(self):
        """Initiates a clean, graceful shutdown of all threads and pools."""
        print("\n  [Shutdown Initiated] Starting graceful shutdown...")
        
        # 1. Signal background sync thread to stop
        self.shutdown_event.set()
        if self.service_thread:
            self.service_thread.join(timeout=1.0)
            
        # 2. Shut down the ThreadPoolExecutor
        # wait=True blocks until all running worker threads finish their current image
        print("  [Shutdown] Shutting down thread pool executor...")
        self.executor.shutdown(wait=True)
        print("  [Shutdown] Thread pool executor closed safely.")
        print("  [Shutdown] Hybrid system shutdown complete.\n")


def run_hybrid_backend_demo():
    print("--- 3. Running Practical Hybrid Backend Pipeline ---")
    service = HybridBackendService()
    try:
        # Run top-level async pipeline
        asyncio.run(service.run_main_pipeline())
    finally:
        # Shut down legacy threads and executor pools
        service.shutdown()


# ============================================================================
# MASTER CLASS INTERVIEW PREP: COROUTINES-THREADS BRIDGE Q&AS
# ============================================================================
"""
================================================================================
Q1: Can I share an `asyncio.Queue` or `asyncio.Event` with a background thread?
A: Absolutely NOT. Standard Asyncio synchronization primitives are not thread-safe.
   They are optimized for cooperative single-threaded speed and lack thread-safety 
   mutexes. If modified concurrently from outside the event loop thread, their internal 
   pointers will corrupt, causing state corruption or deadlock.

================================================================================
Q2: How do I schedule an async coroutine to run on an event loop from a synchronous 
    background thread?
A: Use `asyncio.run_coroutine_threadsafe(coro, loop)`. This function is thread-safe.
   It schedules the coroutine on the target loop and returns a standard 
   `concurrent.futures.Future` object, which the synchronous thread can block on 
   to retrieve the return value.

================================================================================
Q3: What is the difference between `asyncio.to_thread()` and `loop.run_in_executor()`?
A: 
   - `asyncio.to_thread()` (introduced in Python 3.9) is a high-level wrapper. 
     It runs the sync function in the event loop's default ThreadPoolExecutor. 
     You cannot specify a custom executor.
   - `loop.run_in_executor()` is a lower-level API. It allows you to pass a custom 
     `ThreadPoolExecutor` or `ProcessPoolExecutor` as the first argument, giving you 
     fine-grained control over the size and lifecycle of the executor.

================================================================================
Q4: What happens to the GIL when `asyncio.to_thread()` offloads a function to 
    a background thread?
A: The background thread is a standard OS thread. When running Python code, it 
   must acquire and contend for CPython's Global Interpreter Lock (GIL). 
   However, if the offloaded function performs blocking system operations 
   (like network request, file I/O, or `time.sleep()`), the Python interpreter 
   temporarily *releases* the GIL during the wait, allowing the main Event Loop thread 
   to run Python bytecode without interruption.

================================================================================
Q5: If I run a heavy CPU-bound Python loop inside `asyncio.to_thread()`, will it 
    unblock the Event Loop?
A: Partially, but not efficiently. Because it is a CPU-bound task in *pure Python*, 
   the background thread must continuously hold the CPython GIL. Every 5 milliseconds, 
   CPython will force the thread to release the GIL (`sys.getswitchinterval()`), 
   allowing the main Event Loop thread a tiny window to execute active coroutines. 
   However, the constant GIL contention and context switching will severely degrade 
   performance and make both the Event Loop and CPU task run much slower. 
   The correct solution for heavy CPU work is to run it in a separate process via 
   `loop.run_in_executor()` with a `ProcessPoolExecutor`.

================================================================================
Q6: How do I shut down an event loop that is running forever inside a background thread?
A: You must invoke the shutdown thread-safely:
   1. Call `loop.call_soon_threadsafe(loop.stop)` from your main or manager thread.
   2. This schedules a stop command inside the loop thread's context. 
   3. Once the loop stops, the thread executing `loop.run_forever()` will exit.
   4. Call `thread.join()` to clean up the OS thread, and then call `loop.close()` 
      to free loop-bound resources.
"""

# ============================================================================
# MAIN EXECUTION ENTRY POINT
# ============================================================================

def main():
    print("======================================================================")
    print("STARTING COMPREHENSIVE COROUTINES & THREADS BRIDGE TUTORIAL")
    print("======================================================================\n")

    # Demo 0: The Concurrency Hierarchy working together (Loops, Threads, Tasks, Processes)
    asyncio.run(demonstrate_concurrency_hierarchy())

    # Demo 1: Offloading blocking work from async (Async -> Sync)
    asyncio.run(demonstrate_offloading())

    # Demo 2: Running an async loop in background thread (Sync -> Async)
    demonstrate_background_loop()

    # Demo 3: Practical hybrid pipeline simulation
    run_hybrid_backend_demo()

    print("======================================================================")
    print("TUTORIAL COMPLETE: Core bridging concepts successfully demonstrated!")
    print("======================================================================")


if __name__ == "__main__":
    main()
