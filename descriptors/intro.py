# ============================================================================
# COMPREHENSIVE PYTHON DESCRIPTORS TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers preparing for Mid-to-Senior level roles.
# Topic: The Descriptor Protocol, Data vs Non-Data Descriptors, Attribute Resolution Order, 
#        and how Python uses them under the hood (methods, property, classmethod).
# Instructions: You can run this entire file. It is syntactically valid Python.
# ============================================================================

import sys
import weakref

# ============================================================================
# 1. WHAT IS A DESCRIPTOR?
# ============================================================================
"""
A descriptor is any object that defines at least one of the following methods in
its class (the Descriptor Protocol):
- __get__(self, instance, owner)
- __set__(self, instance, value)
- __delete__(self, instance)
- __set_name__(self, owner, name)  (Introduced in Python 3.6)

Descriptors let you customize attribute access (getting, setting, deleting)
when they are bound to a CLASS attribute of another class.

Key concept: Descriptors must be defined at the CLASS level, not instance level!
"""

class SimpleDescriptor:
    """A descriptor that simply logs access."""
    def __get__(self, instance, owner):
        # self: The descriptor instance itself.
        # instance: The instance of the owner class calling the descriptor (or None if called from the class).
        # owner: The owner class itself.
        print(f"--> __get__ called: self={self}, instance={instance}, owner={owner}")
        if instance is None:
            return self
        return "Descriptor Value!"

    def __set__(self, instance, value):
        print(f"--> __set__ called: self={self}, instance={instance}, value={value}")

class Owner:
    # Bound at class level
    attribute = SimpleDescriptor()

def demonstrate_simple_descriptor():
    print("\n--- 1. Simple Descriptor Basics ---")
    obj = Owner()
    
    # Access via instance
    print("Accessing attribute via instance:")
    val = obj.attribute
    print(f"Returned value: {val}")
    
    # Access via class
    print("\nAccessing attribute via Class:")
    val_class = Owner.attribute
    print(f"Returned value: {val_class}")
    
    # Setting attribute
    print("\nSetting attribute on instance:")
    obj.attribute = "New Value"


# ============================================================================
# 2. DATA vs NON-DATA DESCRIPTORS
# ============================================================================
"""
Descriptors are categorized based on which protocol methods they implement:

1. Non-Data Descriptors: Implement ONLY __get__.
   - Example: Regular methods, @classmethod, @staticmethod.
   - Overrideability: If an instance dictionary contains a key with the same name,
     the instance dictionary takes precedence.

2. Data Descriptors: Implement __get__ AND (__set__ or __delete__).
   - Example: @property, custom validators.
   - Overrideability: A data descriptor ALWAYS takes precedence over the instance dictionary!
"""

class NonDataDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return "Default Non-Data Value"

class DataDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return "Default Data Value"
        
    def __set__(self, instance, value):
        print(f"Setting DataDescriptor to {value}")

class DemoClass:
    non_data = NonDataDescriptor()
    data = DataDescriptor()

def demonstrate_data_vs_nondata():
    print("\n--- 2. Data vs Non-Data Descriptors (Precedence) ---")
    obj = DemoClass()
    
    # Override in instance dictionary
    obj.__dict__['non_data'] = "Instance Overridden Non-Data"
    obj.__dict__['data'] = "Instance Overridden Data"
    
    # What happens when we access them?
    print(f"Accessing 'non_data': {obj.non_data}") 
    # Output: "Instance Overridden Non-Data" (Instance dict won because non_data is Non-Data!)
    
    print(f"Accessing 'data': {obj.data}")
    # Output: "Default Data Value" (Data descriptor won because it implements __set__!)


# ============================================================================
# 3. THE ATTRIBUTE RESOLUTION ORDER (UNDER THE HOOD)
# ============================================================================
"""
When you run `obj.name`, Python's `object.__getattribute__()` is triggered.
It implements the following strict resolution hierarchy:

1. Search the Class Hierarchy for a DATA DESCRIPTOR. If found, call its `__get__`.
2. Search the Instance Dictionary (`obj.__dict__`). If found, return that value.
3. Search the Class Hierarchy for a NON-DATA DESCRIPTOR. If found, call its `__get__`.
4. Search the Class Hierarchy for standard class attributes. If found, return.
5. If nothing is found, call `__getattr__(self, 'name')` if defined, else raise `AttributeError`.

This explains why you can shadow class methods with instance variables, 
but you CANNOT shadow properties!
"""


# ============================================================================
# 4. HOW PYTHON USES DESCRIPTORS INTERNALLY
# ============================================================================
"""
Descriptors power the core features of Python's OOP layer.

A. Python Functions (Methods)
   Every function in Python is a NON-DATA descriptor because it implements `__get__`.
   When you do `obj.method()`, `method.__get__(obj, MyClass)` is called, which 
   binds the function to the instance, turning it into a "bound method" and 
   automatically prepending `self` as the first argument!

B. Properties
   `property()` is a built-in DATA descriptor. It implements `__get__`, `__set__`,
   and `__delete__` to map attribute access to getter/setter/deleter functions.

C. @classmethod and @staticmethod
   These are descriptors.
   - `@classmethod`'s `__get__` returns a bound method with the Class (cls) prepended.
   - `@staticmethod`'s `__get__` returns the underlying function directly, without binding.
"""

class MethodDemo:
    def greet(self):
        return "Hello!"

def demonstrate_internal_descriptors():
    print("\n--- 4. How Python Uses Descriptors Internally ---")
    obj = MethodDemo()
    
    # Let's inspect the type of the function on the class vs the instance
    func_on_class = MethodDemo.greet
    method_on_instance = obj.greet
    
    print(f"Type of function on class: {type(func_on_class)}") # <class 'function'>
    print(f"Type of method on instance: {type(method_on_instance)}") # <class 'method'>
    
    # Under the hood, Python did this:
    bound_manually = MethodDemo.greet.__get__(obj, MethodDemo)
    print(f"Manually bound method: {bound_manually}")
    print(f"Do they match? {bound_manually == method_on_instance}")


# ============================================================================
# 5. PRACTICAL USE CASES & THE SHARING PROBLEM (CRITICAL PITFALL)
# ============================================================================
"""
Pitfall: Because descriptors are bound at the CLASS level, there is only ONE 
descriptor instance shared across all instances of the owner class!

If we store instance-specific data in the descriptor itself (e.g., self.val = value),
every instance of the owner class will overwrite each other's data!
"""

class BadValidator:
    """STORE STATE IN DESCRIPTOR -> HUGE BUG!"""
    def __init__(self):
        self.val = None
        
    def __get__(self, instance, owner):
        return self.val
        
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError("Must be positive")
        self.val = value

class GoodValidator:
    """
    SOLUTION 1 (Modern, Python 3.6+): Use `__set_name__`.
    `__set_name__` is called when the owner class is created, passing the owner 
    class and the name of the attribute. This allows us to store the value inside 
    the instance's own __dict__ under a unique key, preventing shared state bugs 
    and avoiding hardcoded attribute names.
    """
    def __set_name__(self, owner, name):
        # We store the state in the owner instance using this key:
        self.private_name = f"_{name}"
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)
        
    def __set__(self, instance, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Must be a non-negative integer.")
        setattr(instance, self.private_name, value)

class Product:
    # Both instances of Product share these class attributes:
    bad_price = BadValidator()
    good_price = GoodValidator()

def demonstrate_descriptor_pitfall():
    print("\n--- 5. Descriptor State Pitfall & __set_name__ ---")
    p1 = Product()
    p2 = Product()
    
    # --- DEMONSTRATING THE BUG ---
    print("Setting p1.bad_price to 100")
    p1.bad_price = 100
    print(f"p1.bad_price = {p1.bad_price}")
    
    print("Setting p2.bad_price to 200")
    p2.bad_price = 200
    print(f"p1.bad_price = {p1.bad_price} (P1 was overwritten because of shared class-level state!)")
    
    # --- DEMONSTRATING THE CORRECT WAY ---
    print("\nSetting p1.good_price to 10")
    p1.good_price = 10
    print("Setting p2.good_price to 20")
    p2.good_price = 20
    print(f"p1.good_price = {p1.good_price} (Remains 10 - state correctly isolated in the instance!)")
    print(f"p1's internal __dict__: {p1.__dict__}")


# ============================================================================
# 6. ADVANCED USE CASE: LAZY / CACHED PROPERTY
# ============================================================================
"""
Another common use case is implementing a @lazy_property or @cached_property.
This runs a costly computation once, caches the result in the instance's 
dictionary under the same name, and future lookups fetch it directly from the 
instance dictionary (bypassing the non-data descriptor's __get__).
"""

class LazyProperty:
    def __init__(self, func):
        self.func = func
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        print(f"--> [LazyProperty] Running expensive calculation for '{self.func.__name__}'...")
        value = self.func(instance)
        # Cache the value in the instance's dictionary with the property's name.
        # Future attribute lookups will hit the instance dict directly, bypassing __get__!
        instance.__dict__[self.func.__name__] = value
        return value

class DeepComputation:
    def __init__(self, data):
        self.data = data
        
    @LazyProperty
    def heavy_result(self):
        # Simulate a heavy math computation
        return sum(x * x for x in self.data)

def demonstrate_lazy_property():
    print("\n--- 6. Advanced Use Case: Lazy/Cached Property ---")
    comp = DeepComputation(range(100000))
    
    print("First Access:")
    val1 = comp.heavy_result
    print(f"Result: {val1}")
    
    print("\nSecond Access (should be cached, no calculation message printed!):")
    val2 = comp.heavy_result
    print(f"Result: {val2}")
    
    print(f"\nInternal __dict__ of comp object: {comp.__dict__}")


# ============================================================================
# COMMON TECHNICAL INTERVIEW QUESTIONS & ANSWERS (DESCRIPTORS)
# ============================================================================
"""
Q1: What is a Descriptor in Python, and how does it relate to properties?
A: A descriptor is an object that customizes attribute access (lookup, modification, deletion) 
   by implementing any of the following dunder methods: `__get__()`, `__set__()`, or `__delete__()`.
   Under the hood, Python's built-in `@property`, `@classmethod`, `@staticmethod`, and even bound methods 
   are implemented using the descriptor protocol!

Q2: What is the difference between a Data Descriptor and a Non-Data Descriptor?
A: 
- A Data Descriptor implements both `__get__()` and `__set__()` (and/or `__delete__()`).
  It takes precedence over the instance's dictionary (`__dict__`). Even if the instance has a key with the same name, 
  Python will route attribute access through the data descriptor.
- A Non-Data Descriptor only implements `__get__()` (e.g., bound methods).
  It has lower precedence than `__dict__`. If a key with the same name is added to the instance's `__dict__`, 
  it will shadow/override the non-data descriptor.

Q3: What is the purpose of the `__set_name__` method introduced in Python 3.6?
A: Historically, a descriptor had no way of knowing the variable name it was assigned to in the parent class 
   without passing the name explicitly to its `__init__`.
   `__set_name__(self, owner, name)` is called automatically when the class is defined, allowing the descriptor 
   to automatically capture the field name (e.g., storing the value inside `owner.__dict__` using the exact field name).
"""


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    demonstrate_simple_descriptor()
    demonstrate_data_vs_nondata()
    demonstrate_internal_descriptors()
    demonstrate_descriptor_pitfall()
    demonstrate_lazy_property()
    
    print("\n--- Tutorial Complete ---")
    print("Key takeaway: Descriptors power properties, classmethods, and standard methods.")
    print("Remember to use __set_name__ (Python 3.6+) or weakrefs to avoid shared state bugs!")
