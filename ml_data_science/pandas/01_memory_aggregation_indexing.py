# ============================================================================
# PANDAS UNDER THE HOOD: MEMORY OPTIMIZATION, INDEXING, AND GROUPBY
# ============================================================================
# Target Audience: Data Engineers, Quant Analysts, AI Engineers.
# Topic: Category memory footprints, SettingWithCopy Warning, Copy-on-Write, 
#        Split-Apply-Combine, custom Aggregations, and .transform().
# Compatibility: Python 3.10+, Pandas 1.4+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Pandas?
   - Pandas is Python's primary library for data analysis. It provides two core data structures:
     - Series: A 1D labeled array (like a single column).
     - DataFrame: A 2D tabular data structure (like an Excel spreadsheet or SQL table).
2. What is GroupBy?
   - GroupBy implements the "Split-Apply-Combine" workflow:
     - Split: Divides the dataset into groups based on keys (e.g. grouping sales by store).
     - Apply: Evaluates a function on each group (e.g. summing sales within each group).
     - Combine: Merges the independent sums back into a clean summary table.
3. What is the difference between .loc and .iloc?
   - `.loc` is label-based indexing (looking up rows and columns by their written names). 
     `.iloc` is integer-position indexing (looking up rows and columns by their numeric index 
     positions: 0, 1, 2...).
"""



import pandas as pd
import numpy as np
import sys

# ============================================================================
# 1. COMPRESSING DATAFRAME MEMORY FOOTPRINTS
# ============================================================================
"""
When loading large datasets, default datatypes waste massive amounts of memory:
1. Object Strings: Standard string columns are stored as lists of pointers pointing 
   to Python string objects scattered in memory. This is highly inefficient.
   - Categorical Encoding: Converting low-cardinality columns (like 'Store_ID', 'Gender') 
     to `category` dtype replaces string pointers with small integer indices, 
     compressing memory usage by up to 90% and accelerating merge/join performance.
2. Integer/Float Downcasting: Standard loaders assign float64 or int64 by default. 
   Using `pd.to_numeric(..., downcast='float')` downcasts 64-bit numerical representations 
   to 32 or 16-bit precision if values stay within bounds.
"""

def demonstrate_memory_optimization():
    print("\n" + "="*80)
    print("1. PANDAS MEMORY OPTIMIZATION")
    print("="*80)

    # 1. Create a large DataFrame with redundant strings
    num_rows = 50000
    store_options = ["Store_London_Central", "Store_Reading_HQ", "Store_Bristol_Hub", "Store_Brighton_Beach"]
    
    df = pd.DataFrame({
        "Store_ID": np.random.choice(store_options, size=num_rows),
        "Sales": np.random.randint(100, 10000, size=num_rows).astype(np.int64)
    })

    # Read memory usage of object string column
    mem_store_obj = df["Store_ID"].memory_usage(deep=True)
    mem_sales_int64 = df["Sales"].memory_usage(deep=True)

    print(f"Original Object-String Column Memory: {mem_store_obj / 1024 / 1024:.2f} MB")
    print(f"Original Int64 Column Memory:         {mem_sales_int64 / 1024 / 1024:.2f} MB")

    # 2. Optimize types
    df["Store_ID"] = df["Store_ID"].astype("category")
    df["Sales"] = pd.to_numeric(df["Sales"], downcast="unsigned")

    mem_store_cat = df["Store_ID"].memory_usage(deep=True)
    mem_sales_downcasted = df["Sales"].memory_usage(deep=True)

    print(f"\nAfter Conversion:")
    print(f"  Categorical Store_ID Memory: {mem_store_cat / 1024 / 1024:.2f} MB")
    print(f"  Downcasted Sales Memory:     {mem_sales_downcasted / 1024 / 1024:.2f} MB")
    print(f"  Store_ID Compression Ratio:  {mem_store_obj / mem_store_cat:.1f}x compression!")


# ============================================================================
# 2. DEMYSTIFYING THE SETTINGWITHCOPYWARNING
# ============================================================================
"""
SettingWithCopyWarning occurs when you modify a sliced DataFrame using Chained Indexing:
    `df[df['Sales'] > 5000]['Sales'] = 999`

Why does Pandas warn?
Because chained indexing involves two sequential indexing operations: 
`__getitem__` followed by `__setitem__`. Pandas cannot guarantee if the intermediate 
object returned by the first get-item is a direct pointer to the original memory (view) 
or a newly allocated array (copy). If it's a copy, setting the value changes the copy 
and is silently lost.

Prevention:
- Use `.loc` to modify columns in a single, atomic operation: `df.loc[mask, 'col'] = val`.
- In Pandas 2.0+, Copy-on-Write (CoW) is enabled by default or option, removing 
  this ambiguity entirely by making any copy write-isolated.
"""

def demonstrate_setting_with_copy():
    print("\n" + "="*80)
    print("2. THE SETTINGWITHCOPYWARNING PARADIGM")
    print("="*80)

    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})

    print("Chained Indexing (Unsafe):")
    # This might print a warning depending on current pandas settings:
    # df[df["A"] > 1]["B"] = 999
    print("  Use .loc instead! e.g., df.loc[df['A'] > 1, 'B'] = 999")
    
    # Safe operation:
    df.loc[df["A"] > 1, "B"] = 999
    print("Safe Updated DataFrame:\n", df)


# ============================================================================
# 3. SPLIT-APPLY-COMBINE: AGGREGATIONS VS TRANSFORMS
# ============================================================================
"""
When grouping data using `.groupby()`:
1. `.agg()` (Aggregation): Condenses rows. It takes $N$ grouped rows and outputs 
   a single value (e.g. mean, sum), reducing the total size of the DataFrame.
2. `.transform()` (Transformation): Retains row count. It executes calculation within 
   each group but returns an array of the *exact same size and index shape* as the input. 
   This is highly powerful for centering group variables (e.g., subtracting group mean 
   from each row).
"""

def demonstrate_groupby_transform():
    print("\n" + "="*80)
    print("3. GROUPBY SPLIT-APPLY-COMBINE TECHNIQUES")
    print("="*80)

    df = pd.DataFrame({
        "Department": ["HR", "HR", "IT", "IT", "IT"],
        "Employee": ["Alex", "Alexander", "Kneale", "Bob", "John"],
        "Salary": [50000, 60000, 90000, 70000, 80000]
    })

    # Deduct individual salaries from their department median using transform
    # Transform guarantees we can perform aligned subtraction directly!
    dept_medians = df.groupby("Department")["Salary"].transform("median")
    
    df["Salary_Diff_From_Dept_Median"] = df["Salary"] - dept_medians
    print("DataFrame with Centered Group Salaries:\n", df)


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: What is the difference between `.loc` and `.iloc` in Pandas?
   - Answer: `.loc` is label-based indexing (looking up index labels and column names). 
     `.iloc` is integer-position based indexing (looking up 0-indexed positions, 
     equivalent to standard slicing on NumPy matrices).

2. MEDIUM: Explain what the SettingWithCopyWarning is, and how to avoid it.
   - Answer: It occurs during chained indexing operations (e.g., `df[mask]['col'] = 5`). 
     Because the operation executes in two steps, Pandas cannot guarantee if the first slice 
     is a view (pointing to original memory) or a copy. If it is a copy, modifications are lost. 
     Avoid it by performing single-pass indexing via `.loc` or `.iloc` (e.g., `df.loc[mask, 'col'] = 5`).

3. HARD: Explain how Pandas GroupBy implements the "Split-Apply-Combine" pattern.
   - Answer:
     - Split: Pandas splits the DataFrame rows into groups based on matching key values 
       (creating mapping dictionaries from key to index positions).
     - Apply: Evaluates a custom function on each group subset independently.
     - Combine: Concatenates the independent groups back into a single aligned 
       pandas DataFrame structure, reconstructing indices.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a function `calculate_relative_shares(df, group_col, target_col)` 
   that computes the percentage share of each record's target value relative to the sum total 
   of its matching group, without manual loops.
   - Code Template to execute:
"""

def calculate_relative_shares(df: pd.DataFrame, group_col: str, target_col: str) -> pd.Series:
    # 1. Use transform to broadcast group sum across rows, matching the shape of input
    group_sums = df.groupby(group_col)[target_col].transform("sum")
    
    # 2. Divide target elements by their corresponding group sums
    return df[target_col] / group_sums


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    demonstrate_memory_optimization()
    demonstrate_setting_with_copy()
    demonstrate_groupby_transform()
    
    # Testing relative shares
    print("\nTesting Custom Group relative shares:")
    test_df = pd.DataFrame({
        "Store": ["London", "London", "London", "Bristol", "Bristol"],
        "Revenue": [100.0, 200.0, 700.0, 400.0, 400.0]
    })
    # London total: 1000. Bristol total: 800.
    # Expected relative shares:
    # London: [0.1, 0.2, 0.7]
    # Bristol: [0.5, 0.5]
    shares = calculate_relative_shares(test_df, "Store", "Revenue")
    test_df["Revenue_Share_Of_Store"] = shares
    print(test_df)