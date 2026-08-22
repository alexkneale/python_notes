# ============================================================================
# HAYSTACK 2.x DEEP DIVE: PIPELINES, DOCUMENT STORES, AND CUSTOM COMPONENTS
# ============================================================================
# Target Audience: AI Engineers, RAG Developers.
# Topic: Component Architecture, InMemoryDocumentStore, Declarative Pipelines,
#        and custom Component Output Typing.
# Compatibility: Python 3.10+, Haystack 2.0+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Haystack?
   - Haystack is an end-to-end framework for building search engines, RAG pipelines, and 
     conversational agents. It emphasizes clean modularity and enterprise-level speed.
2. What are Components in Haystack 2.x?
   - Components are the building blocks of Haystack. Each component has a single purpose 
     (like extracting text from PDFs, generating embeddings, or fetching files). They 
     are highly decoupled, declaring exact data inputs and outputs.
3. What is a Document Store?
   - A Document Store is a database wrapper (like InMemory, Elasticsearch, or Pinecone) 
     designed to hold text, tables, and embeddings. Components like Retrievers query 
     this store to fetch relevant documents.
"""



from typing import Dict, Any, List, Optional
import json

# ============================================================================
# 1. THE HAYSTACK 2.x ARCHITECTURAL PARADIGM
# ============================================================================
"""
Haystack 2.x represents a complete, clean rewrite of the classic framework, 
orienting everything around two simple, highly decoupled concepts:

1. COMPONENTS: 
   - A Component is a highly focused Python class decorated with `@component`.
   - Every component must declare its inputs and outputs using strict type annotations 
     (e.g., `Optional[List[str]]`) so that Haystack can perform static type validation 
     during pipeline compilation.
   - The execution entry point must be named `run(self, ...)` and return a dictionary 
     declaring its outputs.

2. PIPELINES:
   - A Pipeline is an empty DAG structure. You register components using `pipeline.add_component()`.
   - You build the DAG by establishing connections:
     `pipeline.connect("component_A.output_field_name", "component_B.input_field_name")`
   - Haystack statically validates these connections: if component A's output type 
     (e.g. `List[Document]`) doesn't match component B's input type (e.g. `str`), 
     it throws a clean compilation error *before* execution starts, preventing runtime failures.
"""

# Let's emulate Haystack's component and pipeline DAG validation model in Python
class SimulatedComponent:
    """Mock component structure to illustrate component-level registration."""
    def __init__(self, name: str):
        self.name = name

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# ============================================================================
# 2. DOCUMENT STORES & THE INDEXING PATHWAY
# ============================================================================
"""
In Haystack 2.x, Document Stores acts purely as state repositories. They are 
not responsible for embedding or retrieving documents themselves:
- Document Store: Holds raw `Document` objects (id, content, metadata, embedding).
- Document Writer: A standard component used to write documents to the store.
- Retriever: A separate component (e.g., `InMemoryBM25Retriever`) that connects 
  to the store and runs search queries inside pipelines.
"""

class SimulatedDocument:
    def __init__(self, id: str, content: str, meta: Optional[dict] = None):
        self.id = id
        self.content = content
        self.meta = meta or {}


# ============================================================================
# 3. IMPLEMENTING A CUSTOM COMPONENT IN HAYSTACK 2.x STYLE
# ============================================================================

class CustomQueryExpander:
    """
    Simulating a Haystack 2.x Custom Component.
    In real Haystack 2.x, this class would have the `@component` decorator.
    
    It accepts a raw search query and expands it using predefined domain synonyms 
    (e.g., mapping 'O2' -> 'O2 UK Telefonica').
    """
    def __init__(self, synonyms: dict = None):
        self.synonyms = synonyms or {}

    def run(self, query: str) -> Dict[str, Any]:
        """
        Input: query (str)
        Output: Dict mapping 'expanded_query' -> str
        """
        words = query.split()
        expanded_words = []
        for word in words:
            expanded_words.append(word)
            if word in self.synonyms:
                expanded_words.append(f"({self.synonyms[word]})")
        
        return {"expanded_query": " ".join(expanded_words)}


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the main design difference between Haystack 1.x and Haystack 2.x?
   - Answer: Haystack 1.x tightly coupled Document Stores with search algorithms 
     (e.g., calling retriever.retrieve() directly on the store). Haystack 2.x completely 
     decouples them. Document Stores are now simple, passive database wrappers. Retrievers 
     are standalone pipeline components that are passed a reference to the Document Store 
     during initialization, making components highly modular and swap-able.

2. MEDIUM: Why must Haystack 2.x components declare strict typing annotations on inputs 
   and outputs?
   - Answer: Typing annotations are analyzed during pipeline validation. When you connect 
     nodes via `pipeline.connect("node1.output", "node2.input")`, Haystack verifies that 
     node1's output data type matches node2's expected input type. If there is a mismatch 
     (e.g. connecting a Document list output to a string input), Haystack raises a compilation 
     error instantly, avoiding hard-to-debug crashes halfway through multi-step executions.

3. HARD: How do you build a multi-modal pipeline in Haystack 2.x?
   - Answer: In Haystack 2.x, `Document` objects are multi-modal and support holding 
     diverse datatypes (text, tables, or binary data references like images). To construct 
     a multi-modal pipeline, you create separate extraction branches in the DAG: 
     (1) An image extraction component that parses images and calls a multi-modal model 
         (like Gemini Pro Vision) to yield summaries.
     (2) A text parser branch.
     The outputs are merged using a `DocumentJoiner` component and piped into a unified 
     Vector Document Store.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom, runnable-compliant component class `TextCleanerComponent` 
   that cleans text inputs by stripping leading/trailing whitespace, and replacing 
   all occurrences of multiple consecutive spaces with a single space.
   - Code Template to execute:
"""

class TextCleanerComponent:
    def run(self, text: str) -> Dict[str, str]:
        # 1. Clean the text
        # Strip and join elements split by whitespace to remove redundant spaces
        cleaned = " ".join(text.strip().split())
        
        # 2. Return output dict matching expected signature
        return {"cleaned_text": cleaned}


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING HAYSTACK 2.x COMPONENT ARCHITECTURE")
    print("="*80)

    # 1. Instantiate the custom query expander component
    synonyms_map = {"VMO2": "Virgin Media O2", "SQL": "PostgreSQL Database"}
    expander = CustomQueryExpander(synonyms=synonyms_map)

    # 2. Run query expansion
    raw_query = "Troubleshoot SQL replication error on VMO2"
    result = expander.run(query=raw_query)
    
    print(f"Raw Input Query:  {raw_query}")
    print(f"Expanded Output:  {result['expanded_query']}")

    # 3. Test Custom Text Cleaner Component
    print("\nTesting Custom Text Cleaner Component:")
    dirty_text = "   Virgin    Media   O2   UK   "
    cleaner = TextCleanerComponent()
    cleaned_res = cleaner.run(text=dirty_text)
    
    print(f"  Dirty Input: '{dirty_text}'")
    print(f"  Cleaned:     '{cleaned_res['cleaned_text']}' (Expected: 'Virgin Media O2 UK')")