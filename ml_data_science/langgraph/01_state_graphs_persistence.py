# ============================================================================
# LANGGRAPH ENGINEERING: STATE GRAPHS, REDUCERS, AND STATE PERSISTENCE
# ============================================================================
# Target Audience: Senior AI Engineers, Graph Systems Researchers.
# Topic: State Schema Channels, Reducer Operators, Memory Saver Checkpointers, 
#        Thread IDs, and Time-Travel State Updates.
# Compatibility: Python 3.10+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is LangGraph?
   - LangGraph is a library built on top of LangChain to design agentic systems as graphs. 
     Unlike standard linear pipelines, LangGraph supports cyclic architectures, allowing 
     agents to loop, plan, execute tools, and repeat steps dynamically.
2. What are Nodes and Edges?
   - In a graph:
     - Nodes: These are individual Python functions representing an action (e.g., retrieving 
       documents or calling an LLM).
     - Edges: These define the connection paths between nodes, including conditional routes 
       (e.g., "if tool failed, go to error_fixer; else, go to finalizer").
3. What is Graph State & Persistence?
   - LangGraph maintains a single, central state object (like a shared dictionary) that 
     is passed from node to node. Checkpointers (persistence) automatically save snapshots of 
     this state at every transition, allowing multi-turn memory and execution resumes.
"""



from typing import Dict, Any, List, Annotated
import operator

# ============================================================================
# 1. GRAPH STATE SCHEMA & REDUCERS
# ============================================================================
"""
In standard state graphs, writing to a key in the State dictionary overwrites 
the previous value. However, LangGraph allows defining REDUCER functions on 
keys (using `Annotated[Type, reducer_function]`).

A Reducer defines how to merge a new value into the existing state:
- Standard Key (no reducer): Overwrites previous state.
- Annotated[List, operator.add] key: Appends new elements to the existing list, 
  crucial for preserving a history of chat messages or tool executions!
"""

# Let's create a simulated Graph State representing this behavior
class SimulatedGraphState:
    def __init__(self):
        # Equivalent to LangGraph State definition:
        # messages: Annotated[List[str], operator.add]
        # current_step: str
        self.messages: List[str] = []
        self.current_step: str = "init"

    def update_state(self, updates: Dict[str, Any]):
        """
        Emulates a LangGraph state update.
        If a key matches 'messages', it runs the reducer operator.add (appending).
        Other keys overwrite.
        """
        for k, v in updates.items():
            if k == "messages":
                # Reducer: append elements
                if isinstance(v, list):
                    self.messages.extend(v)
                else:
                    self.messages.append(v)
            else:
                # Overwrite
                setattr(self, k, v)


# ============================================================================
# 2. STATE PERSISTENCE & CHECKPOINTING (TIME TRAVEL)
# ============================================================================
"""
In production, multi-turn conversations must survive server restarts and maintain 
session isolation. LangGraph implements this using Checkpointers.

Key Concepts:
1. Thread IDs: A single Graph compiled with a Checkpointer can manage thousands 
   of distinct conversations. We specify a `thread_id` in the execution config. 
   The Checkpointer automatically saves a snapshot of the graph's state under that 
   `thread_id` at every single node transition.
2. Time Travel: Because states are versioned at each node, you can query historical 
   checkpoints, view what the state looked like 3 steps ago, modify a variable (inject 
   state), and resume execution from that point onwards (Time Travel & Forking).
"""

class SimulatedCheckpointer:
    """
    Custom lightweight mock of a Thread-level Checkpointer to demonstrate 
    exact time-travel state indexing mechanisms.
    """
    def __init__(self):
        # Storage: {thread_id: {checkpoint_version_id: state_snapshot}}
        self.storage: Dict[str, List[Dict[str, Any]]] = {}

    def save_checkpoint(self, thread_id: str, state: SimulatedGraphState):
        if thread_id not in self.storage:
            self.storage[thread_id] = []
        
        # Take a deep copy snapshot of the current messages and step
        snapshot = {
            "messages": list(state.messages),
            "current_step": state.current_step,
            "version": len(self.storage[thread_id])
        }
        self.storage[thread_id].append(snapshot)
        return len(self.storage[thread_id]) - 1

    def get_checkpoint(self, thread_id: str, version_id: int) -> Dict[str, Any]:
        return self.storage[thread_id][version_id]


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is a Reducer function in LangGraph and why is it useful?
   - Answer: A Reducer defines how a new update is merged into a state key. 
     By default, keys are overwritten. Reducers (like `operator.add` on a List) 
     allow updates to append or accumulate values, which is necessary for tracking 
     increasing collections like chat messages or lists of execution logs.

2. MEDIUM: What is the purpose of a Thread ID inside LangGraph configuration?
   - Answer: A Compiled Graph is stateless. To support stateful multi-turn interactions 
     across independent users, we pass a `thread_id` in the config dictionary. 
     The Checkpointer uses this ID to load the matching historical state snapshot 
     before running, and saves the updated state back to the same ID afterwards, 
     ensuring thread-level session isolation.

3. HARD: Explain how "Time Travel" is implemented and how state injection can 
   alter a running agent's execution path.
   - Answer: LangGraph compiles state changes as an append-only transaction ledger. 
     At every node transition, a checkpoint is saved. Time Travel queries this ledger 
     to retrieve historical states. By calling `graph.update_state(config, state_updates, as_node="x")`, 
     we inject a manual correction (e.g. replacing a faulty tool output) and commit 
     it as a new state version. Calling `.stream()` from that checkpoint forks the graph 
     on the corrected path, bypassing the original error.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom reducer function `dictionary_merge_reducer(old_dict, new_dict)` 
   that merges a new dictionary into an old dictionary, but if a key exists in both, 
   sums their numerical values instead of overwriting.
   - Code Template to execute:
"""

def dictionary_merge_reducer(old_dict: dict, new_dict: dict) -> dict:
    merged = old_dict.copy()
    
    for k, v in new_dict.items():
        if k in merged:
            # If both are numbers, sum them
            if isinstance(merged[k], (int, float)) and isinstance(v, (int, float)):
                merged[k] += v
            else:
                # Fallback to overwrite
                merged[k] = v
        else:
            merged[k] = v
            
    return merged


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING STATE GRAPH SCHEMAS & CHECKPOINTERS")
    print("="*80)

    # 1. Verify Reducer Action (operator.add)
    state = SimulatedGraphState()
    print("Initial State Messages:", state.messages)
    
    # Update 1: Append list of messages
    state.update_state({"messages": ["User: Hello", "Agent: Welcome"], "current_step": "greeting"})
    print("After Update 1:         ", state.messages)
    
    # Update 2: Append a single message
    state.update_state({"messages": "User: I need help with Cloud SQL"})
    print("After Update 2:         ", state.messages)
    print("Current Step:           ", state.current_step)

    # 2. Verify Simulated Time-Travel Checkpointer
    checkpointer = SimulatedCheckpointer()
    thread_id = "user-session-109"
    
    # Save current state as version 0
    v0 = checkpointer.save_checkpoint(thread_id, state)
    print(f"\nSaved State Checkpoint Version: {v0}")
    
    # Mutate state to simulate further graph transitions
    state.update_state({"messages": ["Agent: Sure, checking DB..."], "current_step": "db_lookup"})
    v1 = checkpointer.save_checkpoint(thread_id, state)
    print(f"Saved State Checkpoint Version: {v1}")
    print("Current Live Messages:         ", state.messages)

    # Time Travel: Load version 0 back!
    checkpoint_v0 = checkpointer.get_checkpoint(thread_id, v0)
    print("\nTime Travelling Back to Version 0...")
    print("Checkpoint Messages loaded:    ", checkpoint_v0["messages"])
    print("Checkpoint Step loaded:        ", checkpoint_v0["current_step"])

    # 3. Verify Dictionary Merge Reducer
    print("\nTesting Custom Dictionary Merge Reducer:")
    old_state = {"tokens_used": 1500, "user_id": "alex"}
    new_update = {"tokens_used": 500, "user_id": "alex_kneale"} # tokens_used should sum to 2000
    merged_res = dictionary_merge_reducer(old_state, new_update)
    print(f"  Old State:  {old_state}")
    print(f"  New Update: {new_update}")
    print(f"  Merged:     {merged_res} (Expected tokens_used: 2000)")