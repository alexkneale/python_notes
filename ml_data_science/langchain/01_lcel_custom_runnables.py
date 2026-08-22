# ============================================================================
# LANGCHAIN SYSTEM DESIGN: LCEL PIPES, RUNNABLES, AND CUSTOM TOOLS
# ============================================================================
# Target Audience: AI Engineers, Solution Architects.
# Topic: LCEL __or__ Magic, RunnableParallel/Passthrough/Lambda, 
#        Structured Output Parsing, and Custom Tool creation.
# Compatibility: Python 3.10+, LangChain 0.1+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is LangChain?
   - LangChain is an open-source framework designed to help developers build applications 
     with Large Language Models. It provides standard wrappers for LLMs, databases, and prompt 
     templates, letting you connect them easily.
2. What is LCEL (LangChain Expression Language)?
   - LCEL is a declarative syntax for chaining different components. It uses Python's pipe 
     operator `|` (e.g. `prompt | model | parser`) to define sequential streams of data, 
     automating complex parameters, fallbacks, and parallel calls behind the scenes.
3. What are Runnables?
   - Almost all components in LangChain inherit from the `Runnable` class. Runnables have 
     a standardized interface: they can all be invoked (`invoke`), run on batches (`batch`), 
     streamed (`stream`), or executed asynchronously (`ainvoke`).
"""



from typing import Dict, Any, List
import json

# ============================================================================
# 1. UNDER THE HOOD OF LCEL: PYTHON'S __OR__ AND __ROR__ MAGIC
# ============================================================================
"""
LangChain Expression Language (LCEL) allows you to define chains declaratively:
    `chain = prompt | model | parser`

How does this work in Python?
LangChain implements this by overriding Python's bitwise OR operator magic methods 
inside the `Runnable` base class:
1. `__or__(self, other)`: Called when `self | other` is evaluated (Left-to-Right).
2. `__ror__(self, other)`: Called when `other | self` is evaluated (Right-to-Left, 
   used when the left element is a standard Python dictionary or primitive).

By implementing these methods, LangChain constructs an internal execution graph 
where each component passes its output directly to the input of the next. All 
`Runnable` objects also support standard async, batch, and streaming methods 
(`invoke`, `ainvoke`, `batch`, `stream`) out of the box.
"""

class MockRunnable:
    """
    A custom simulation showing how easily you can implement LCEL logic 
    using standard Python magic methods!
    """
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        # self | other: return a new runnable that pipes self's output into other
        if isinstance(other, MockRunnable):
            return MockRunnable(lambda x: other.invoke(self.func(x)))
        elif callable(other):
            return MockRunnable(lambda x: other(self.func(x)))
        raise TypeError("Unsupported pipe type")

    def invoke(self, x):
        return self.func(x)


# ============================================================================
# 2. RUNNABLE PRIMITIVES: PASSTHROUGH, PARALLEL, AND LAMBDA
# ============================================================================
"""
When composing complex LCEL chains, LangChain provides key wrappers:
1. RunnablePassthrough: Passes inputs through unchanged or allows injecting new keys.
2. RunnableParallel: Runs multiple runnables in parallel (highly useful for concurrently 
   fetching vector database context and preparing prompts).
3. RunnableLambda: Converts a raw Python function into a fully compatible `Runnable` 
   supporting invoke, batch, and stream.
"""

# Let's mock LangChain's internal behavior using standard Python objects
class SimulatedLCELPipeline:
    @staticmethod
    def run_parallel_simulation(query: str) -> Dict[str, Any]:
        # Emulating:
        # chain = RunnableParallel(
        #     context=RunnableLambda(retrieve_context),
        #     question=RunnablePassthrough()
        # ) | prompt_formatter
        
        # 1. Define component Lambdas
        retrieve_context = lambda q: f"Retrieved documents matching: '{q}' in Cloud SQL."
        passthrough = lambda q: q
        
        # 2. RunnableParallel execution: Runs both lookups concurrently
        parallel_outputs = {
            "context": retrieve_context(query),
            "question": passthrough(query)
        }
        
        # 3. Next step in pipe: format prompt
        prompt_formatter = lambda data: f"Context: {data['context']}\nQuestion: {data['question']}"
        return prompt_formatter(parallel_outputs)


# ============================================================================
# 3. CUSTOM TOOLS AND DECORATORS
# ============================================================================
"""
Agents rely on tools to interact with external environments. LangChain provides 
the `@tool` decorator to convert standard Python functions into structured tools 
that expose proper descriptions and JSON schemas to LLMs.
"""

def mock_cloud_sql_query(query_str: str) -> str:
    """
    Mock database search. 
    In production, this would execute a raw SQL script or query a BigQuery database.
    """
    if "Virgin Media O2" in query_str:
        return json.dumps({
            "status": "Success",
            "company": "Virgin Media O2",
            "hq": "Reading, England",
            "formed": "June 2021"
        })
    return "Company not found."


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is LCEL in LangChain and what is its main benefit?
   - Answer: LCEL (LangChain Expression Language) is a declarative syntax for composing 
     chains using the pipe operator `|`. It automatically implements parallel execution, 
     streaming, async, batching, and fallback mechanisms on all composed nodes, removing 
     boilerplate code.

2. MEDIUM: Explain the difference between `RunnablePassthrough` and `RunnableParallel`.
   - Answer: `RunnablePassthrough` allows you to pass inputs through unchanged or add 
     extra keys to the input dictionary. `RunnableParallel` takes a dictionary of Runnables 
     and executes them concurrently, returning a dictionary with matching keys, which 
     is optimal for parallel lookups (e.g. merging BM25 and vector search context).

3. HARD: How does Python's `__ror__` magic method enable starting an LCEL chain 
   with a standard Python dictionary, as in: `{"query": ...} | prompt`?
   - Answer: A standard Python dict doesn't implement LCEL's `__or__` operator. When 
     Python evaluates `dict | prompt`, it realizes `dict` has no bitwise OR defined 
     for a Prompt object. It then looks at the right-hand operand (`prompt`) and calls its 
     `__ror__(self, left)` (Right-to-Left OR) method. LangChain's Prompt class implements 
     `__ror__`, wrapping the incoming dictionary in a `RunnablePassthrough` and successfully 
     chaining the nodes.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom, runnable-like class `CustomMapRunnable` that implements 
   `__or__` and `__ror__` magic methods, allowing you to pipe dictionaries and run 
   computational map updates (e.g. duplicating string query length).
   - Code Template to execute:
"""

class CustomMapRunnable:
    def __init__(self, key: str, transformation_fn):
        self.key = key
        self.transformation_fn = transformation_fn

    def __ror__(self, left_input: dict) -> dict:
        # dictionary | CustomMapRunnable: we modify the dictionary and return it
        out = left_input.copy()
        if self.key in out:
            out[self.key] = self.transformation_fn(out[self.key])
        return out


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING CUSTOM SIMULATED LCEL PIPELINES")
    print("="*80)

    # 1. Test MockRunnable custom __or__ chaining
    step1 = MockRunnable(lambda x: x + 10)
    step2 = MockRunnable(lambda x: x * 2)
    # Complete custom chain! (x + 10) * 2
    chain = step1 | step2
    print(f"Custom Magic-Or Chaining: (5 + 10) * 2 = {chain.invoke(5)} (Expected: 30)")

    # 2. Test Parallel Simulation
    fused_prompt = SimulatedLCELPipeline.run_parallel_simulation("Analyze Virgin Media O2")
    print(f"\nSimulated Parallel Retrieval and Prompting:\n{fused_prompt}")

    # 3. Test Custom Map Runnable
    print("\nTesting Custom Ror Map Runnable:")
    state = {"query": "VMO2"}
    # Pipe dictionary directly into the CustomMapRunnable using __ror__!
    result = state | CustomMapRunnable(key="query", transformation_fn=lambda q: q + " - AI Squad")
    print(f"  Input:  {state}")
    print(f"  Output: {result} (Expected query key appended with ' - AI Squad')")