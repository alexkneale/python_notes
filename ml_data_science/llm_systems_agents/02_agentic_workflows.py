# ============================================================================
# LLM SYSTEMS: AGENTIC STATE MACHINES, TOOL CALLING, & STRUCTURED OUTPUTS
# ============================================================================
# Target Audience: AI Engineers, Core Systems Engineers.
# Topic: Schema Grammar Constraints, Function Calling, LangGraph State Patterns,
#        and self-correcting Reflect-and-Refine loops.
# Compatibility: Python 3.10+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is an AI Agent?
   - While traditional LLM apps are simple single-turn prompts (input -> output), an Agent 
     is designed to act autonomously. It can break a goal into steps, decide which tools 
     to use (like running SQL or browsing the web), evaluate its own progress, and self-correct 
     when things go wrong.
2. What is Tool Calling?
   - Tool calling is the mechanism where an LLM is provided with a list of helper functions. 
     If the LLM needs external info (like the weather), it outputs a structured instruction 
     telling your Python script to run the weather function and return the results.
3. What is a State Machine Graph?
   - A state machine structures an agent's workflow into discrete steps (Nodes) connected 
     by rules (Edges). This allows structured loops, like letting the model repeatedly edit 
     its work until a quality grader gives it a passing score.
"""



import json
from typing import Dict, Any, List

# ============================================================================
# 1. UNDER THE HOOD: FUNCTION CALLING & GRAMMAR CONSTRAINTS
# ============================================================================
"""
A common misconception is that LLMs "execute" code or tools directly. In reality:
1. Tool Schema Declaration: The client sends a JSON schema describing the function 
   name, description, and parameter types (using Pydantic models).
2. JSON Generation: The LLM processes the query and outputs a text payload in JSON format, 
   indicating which tool it wants to call and with what arguments.
3. Local Execution: The application intercepts this text, parses the JSON, executes the 
   actual local Python function, and returns the result as a 'Tool Message' back to the LLM.

Ensuring 100% Structured Output (Grammar Constraints):
If we simply ask an LLM to "output JSON", it can still fail due to syntax errors 
(unclosed brackets, missing quotes).
To guarantee syntactically valid structured output, modern runtimes (like llama.cpp, 
Outlines, or vLLM) use Grammar-Constrained Sampling:
- A Context-Free Grammar (CFG) or regex is compiled into a Finite State Machine (FSM).
- At *each token generation step*, the FSM identifies which characters/tokens are 
  syntactically valid next.
- The logits (raw probability scores) for all invalid tokens are set to -infinity, 
  making it mathematically impossible for the model to generate invalid JSON!
"""


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Does an LLM execute the tools we register during "Function Calling"?
   - Answer: No. LLMs do not execute code. Function calling is a communication protocol. 
     The client provides a tool's JSON schema, and the LLM outputs a structured JSON block 
     naming the tool and arguments to call. The orchestration application parses this text, 
     runs the actual local Python code, and feeds the results back to the LLM in a 
     designated Tool Message.

2. MEDIUM: What are the benefits of Graph State Machines over static linear chain prompts?
   - Answer: Linear prompts execute sequentially from start to end without recourse. 
     Graph State Machines (like LangGraph) introduce state-driven cyclic control flow. 
     An agent can draft an answer, call a Critic node, and dynamically loop back 
     to the Generator if validation fails (Self-Correction / Reflect-and-Refine). 
     It also simplifies state persistence, human-in-the-loop approvals, and multi-agent 
     collaboration.

3. HARD: Explain how Grammar-Constrained Sampling guarantees valid JSON formatting.
   - Answer: Standard text generation samples tokens from raw model logits. 
     Grammar-constrained sampling takes a JSON schema and compiles it into a regular expression 
     or Context-Free Grammar (CFG) represented as a Finite State Machine (FSM). 
     At every generation step, the FSM tracks state and identifies which vocabulary tokens 
     are syntactically legal next. The system then forces logits of all invalid tokens 
     to $-\infty$, mathematically preventing the model from ever outputting invalid syntax.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement an agentic conditional router `error_routing_edge(state)` that 
   inspects a dictionary-based state and routes to "draft_generator" if an error key 
   contains a non-empty string, or routes to "finalizer" if the error is empty.
   - Code Template to execute:
"""

def error_routing_edge(state: Dict[str, Any]) -> str:
    # 1. Read 'error' key safely
    error_msg = state.get("error", "")
    
    # 2. Return routing string
    if error_msg and len(error_msg.strip()) > 0:
        return "draft_generator"
    return "finalizer"


# ============================================================================
# 2. STATE GRAPH (LANGGRAPH) PATTERN
# ============================================================================
"""
For complex, multi-step tasks, linear chain-of-thought prompts fail. We instead 
model agents as Directed Graphs (State Machines):
- State: A single, central data structure (usually a typed dict) shared across the graph.
- Nodes: Independent functions that receive the current State, perform work (e.g. call tools, 
  evaluate data), modify the State, and return the updated State.
- Edges: Connections routing execution from one node to the next.
  - Conditional Edges: Decision nodes that evaluate the State and route to different 
    nodes dynamically (e.g. routing to END if information is complete, or routing to 
    RETRY if a tool failed).
"""


# ============================================================================
# 3. RUNNABLE SKELETAL SIMULATION: SELF-CORRECTING REFLECT-AND-REFINE LOOP
# ============================================================================

class AgentState:
    """Central shared graph state tracking agent execution."""
    def __init__(self, query: str):
        self.query: str = query
        self.draft_answer: str = ""
        self.grade_score: float = 0.0
        self.reflection_feedback: str = ""
        self.execution_steps: List[str] = []
        self.loop_count: int = 0


# Define individual Nodes (State modifications)

def draft_generator_node(state: AgentState) -> AgentState:
    """Generates or refines an answer based on query and optional feedback."""
    state.execution_steps.append("DraftGenerator")
    state.loop_count += 1
    
    print(f"\n[Node: DraftGenerator (Attempt {state.loop_count})]")
    
    if state.loop_count == 1:
        # Initial naive attempt
        state.draft_answer = "Virgin Media O2 was formed as a joint venture between Liberty Global and Telefonica."
    else:
        # Refined attempt based on reflection feedback
        print(f"  Incorporating Feedback: '{state.reflection_feedback}'")
        state.draft_answer = "Virgin Media O2 was formed as a joint venture in June 2021 as a merger between Liberty Global's Virgin Media and Telefonica's O2 UK."
        
    print(f"  Draft Answer: '{state.draft_answer}'")
    return state


def critic_node(state: AgentState) -> AgentState:
    """Grades the draft answer and provides constructive feedback."""
    state.execution_steps.append("Critic")
    print("\n[Node: Critic]")
    
    # Simulate a quality evaluation rubric (e.g. checking for precise dates/parents)
    if "June 2021" in state.draft_answer and "O2 UK" in state.draft_answer:
        state.grade_score = 1.0
        state.reflection_feedback = "Perfect. Precise merger details and date provided."
    else:
        state.grade_score = 0.4
        state.reflection_feedback = "Answer is too generic. Must include the exact launch date (June 2021) and clarify that it was O2 UK."
        
    print(f"  Assigned Grade: {state.grade_score * 100:.1f}%")
    print(f"  Feedback:       '{state.reflection_feedback}'")
    return state


# Define Router (Conditional Edge)

def routing_edge(state: AgentState) -> str:
    """
    Decides the next node to route to based on state evaluations.
    """
    print("\n[Routing Edge Evaluation]")
    if state.grade_score >= 0.8:
        print("  Grade exceeds threshold. Routing to: END")
        return "END"
    elif state.loop_count >= 3:
        print("  Max iterations reached. Routing to: END (Safety escape)")
        return "END"
    else:
        print("  Grade below threshold. Routing to: DraftGenerator (Looping back)")
        return "DraftGenerator"


# Run the State Machine Graph
def execute_agent_loop(user_query: str):
    print("\n" + "="*80)
    print("3. RUNNING SELF-CORRECTING STATE MACHINE WORKFLOW")
    print("="*80)

    # Initialize State
    state = AgentState(query=user_query)
    print(f"User Query: {state.query}")

    # Manual Graph compilation loop emulation
    current_node = "DraftGenerator"
    
    while current_node != "END":
        if current_node == "DraftGenerator":
            state = draft_generator_node(state)
            current_node = "Critic" # Standard static transition
        elif current_node == "Critic":
            state = critic_node(state)
            # Evaluate conditional edge transition!
            current_node = routing_edge(state)

    print("\n" + "-"*50)
    print("GRAPH EXECUTION COMPLETED")
    print("-"*50)
    print(f"Final Grade Score: {state.grade_score * 100:.1f}%")
    print(f"Final Answer:      {state.draft_answer}")
    print(f"Execution Path:    {' -> '.join(state.execution_steps)} -> END")


if __name__ == "__main__":
    execute_agent_loop("Detail the formation and structure of Virgin Media O2.")