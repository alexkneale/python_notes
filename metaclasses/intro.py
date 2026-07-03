# ============================================================================
# COMPREHENSIVE PYTHON METACLASSES TUTORIAL (ADVANCED LEVEL)
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Topic: Classes as objects, `type`, __new__ vs __call__, and __init_subclass__.
# Instructions: You can run this entire file. It is syntactically valid Python.
# Read through the comments and functions to understand the deep mechanics.
# ============================================================================

from typing import Any, Dict, Tuple

# ============================================================================
# 1. CLASSES ARE JUST OBJECTS (THE `type` FUNCTION)
# ============================================================================
"""
In Python, EVERYTHING is an object.
When you instantiate a class, you create an object.
But what is the class itself? The class is an instance of a METACLASS.

By default, the metaclass of all classes in Python is `type`.
You can actually use `type()` to create a class dynamically at runtime, 
without ever using the `class` keyword!

Syntax: type(name: str, bases: tuple, namespace: dict)
"""

def demonstrate_type_creation():
    print("--- 1. Dynamic Class Creation with `type` ---")
    
    # 1. Define a function to act as a method
    def speak(self):
        return f"Hello from {self.name}!"

    # 2. Dynamically create a class named 'DynamicRobot'
    # Inherits from 'object' (empty tuple defaults to object)
    # Namespace dict assigns the 'name' attribute and 'speak' method.
    DynamicRobot = type(
        'DynamicRobot', 
        (object,), 
        {'version': 1.0, 'speak': speak}
    )
    
    # 3. Instantiate and use it just like a normal class!
    robot = DynamicRobot()
    robot.name = "R2-D2"
    
    print(f"Class Name: {DynamicRobot.__name__}")
    print(f"Robot speaks: {robot.speak()}")
    print(f"Type of robot: {type(robot)}")
    print(f"Type of DynamicRobot (The Class): {type(DynamicRobot)}") # It's 'type'!


# ============================================================================
# 2. ANATOMY OF A CUSTOM METACLASS
# ============================================================================
"""
A metaclass intercepts the creation of a class. 
When Python reads a `class` definition, it asks the metaclass to build it.

Execution Order:
1. Python reads the class body and builds a namespace dictionary.
2. Metaclass `__new__` is called to allocate memory for the CLASS.
3. Metaclass `__init__` is called to initialize the CLASS.
4. Later, when you instantiate the class (e.g., `obj = MyClass()`), 
   the Metaclass `__call__` is triggered to create the OBJECT.
"""

class AnatomyMeta(type):
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
        """
        __new__ creates the CLASS object. 
        'mcs' refers to the metaclass itself (AnatomyMeta).
        Must return the new class object (usually via super()).
        """
        print(f"[AnatomyMeta.__new__] Allocating memory for class '{name}'")
        
        # We can modify the namespace before the class is even created!
        namespace['injected_by_metaclass'] = "Secret Data"
        
        # Call type.__new__ to actually create the class in memory
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __init__(cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
        """
        __init__ initializes the newly created CLASS object.
        'cls' refers to the class that was just created.
        """
        print(f"[AnatomyMeta.__init__] Initializing class '{name}'")
        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        """
        __call__ intercepts INSTANTIATION (when someone does `MyClass()`).
        'cls' is the class being instantiated.
        Must return the new instance.
        """
        print(f"[AnatomyMeta.__call__] Intercepting instantiation of '{cls.__name__}'")
        
        # 1. Allocate object memory (calls MyClass.__new__)
        instance = super().__call__(*args, **kwargs)
        
        # 2. Modify the instance before returning it to the user
        instance.creation_timestamp = "2023-10-27"
        
        return instance

# The moment Python reads this definition, AnatomyMeta.__new__ and __init__ run!
class DemoClass(metaclass=AnatomyMeta):
    def __init__(self, x):
        self.x = x

def demonstrate_anatomy():
    print("\n--- 2. Anatomy of a Metaclass ---")
    print("Instantiating DemoClass now...")
    
    # This triggers AnatomyMeta.__call__
    obj = DemoClass(10) 
    
    print(f"obj.x: {obj.x}")
    print(f"obj.injected_by_metaclass: {obj.injected_by_metaclass}")
    print(f"obj.creation_timestamp: {obj.creation_timestamp}")


# ============================================================================
# 3. USE CASE 1: ENFORCING CODING STANDARDS
# ============================================================================
"""
Metaclasses are great for strict framework rules. For example, ensuring that 
all methods in a subclass have a docstring, or preventing certain naming conventions.
"""

class EnforceDocstringsMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Skip checking the base class itself (optional, but good practice)
        if name != "BaseAPI":
            for attr_name, attr_value in namespace.items():
                # If it's a method/function and not a dunder method
                if callable(attr_value) and not attr_name.startswith("__"):
                    if not attr_value.__doc__:
                        raise TypeError(f"Method '{attr_name}' in class '{name}' is missing a docstring!")
                        
        return super().__new__(mcs, name, bases, namespace)

class BaseAPI(metaclass=EnforceDocstringsMeta):
    pass

def demonstrate_enforcement():
    print("\n--- 3. Enforcing Standards ---")
    
    try:
        # This will crash AT COMPILE TIME (when Python reads the class definition)
        class BadAPI(BaseAPI):
            def fetch_data(self):
                # No docstring!
                pass
    except TypeError as e:
        print(f"Caught Metaclass Error: {e}")
        
    # This will succeed
    class GoodAPI(BaseAPI):
        def fetch_data(self):
            """Fetches data from the server."""
            pass
    print("GoodAPI created successfully.")


# ============================================================================
# 4. USE CASE 2: AUTOMATIC PLUGIN REGISTRY
# ============================================================================
"""
When building large systems (like Django models or testing frameworks), you 
often want subclasses to automatically register themselves into a central dictionary 
without the developer having to remember to do it.
"""

class PluginRegistryMeta(type):
    registry = {}

    def __new__(mcs, name, bases, namespace):
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Don't register the base plugin class itself
        if name != "BasePlugin":
            mcs.registry[name] = cls
            
        return cls

class BasePlugin(metaclass=PluginRegistryMeta):
    def run(self):
        pass

class AudioPlugin(BasePlugin):
    pass

class VideoPlugin(BasePlugin):
    pass

def demonstrate_registry():
    print("\n--- 4. Automatic Registry ---")
    print("Registered Plugins:")
    for name, plugin_cls in PluginRegistryMeta.registry.items():
        print(f"  - {name}: {plugin_cls}")


# ============================================================================
# 5. USE CASE 3: THE SINGLETON PATTERN
# ============================================================================
"""
A Singleton ensures a class only ever has ONE instance.
If you call the class again, it returns the previously created instance.
We do this by overriding the metaclass's `__call__` method.
"""

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            print(f"[SingletonMeta] Creating the single instance for {cls.__name__}...")
            # Use super() to actually create and init the instance
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            print(f"[SingletonMeta] Returning existing instance for {cls.__name__}...")
            
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        self.connected = True

def demonstrate_singleton():
    print("\n--- 5. The Singleton Pattern ---")
    
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    
    print(f"db1 is db2? {db1 is db2}") # True, they occupy the exact same memory address


# ============================================================================
# 6. THE MODERN ALTERNATIVE: `__init_subclass__` (PYTHON 3.6+)
# ============================================================================
"""
"Metaclasses are deeper magic than 99% of users should ever worry about. 
If you wonder whether you need them, you don't." - Tim Peters

Before Python 3.6, metaclasses were required for auto-registry and enforcement.
Now, Python provides the `__init_subclass__` classmethod, which is MUCH simpler
and avoids Metaclass Conflicts (when a class inherits from two base classes 
with different metaclasses, which causes a TypeError).
"""

class ModernRegistryBase:
    registry = {}

    # This is called whenever a class inherits from ModernRegistryBase
    @classmethod
    def __init_subclass__(cls, **kwargs):
        # We must call super() in case of multiple inheritance
        super().__init_subclass__(**kwargs)
        
        print(f"[__init_subclass__] Registering {cls.__name__}")
        cls.registry[cls.__name__] = cls

class FastPlugin(ModernRegistryBase):
    pass

class SlowPlugin(ModernRegistryBase):
    pass

def demonstrate_init_subclass():
    print("\n--- 6. Modern __init_subclass__ ---")
    print("This achieved the exact same result as Use Case 2, but without metaclasses!")
    print(f"Modern Registry: {ModernRegistryBase.registry}")


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Note: Sections 2, 3, 4, and 6 will output text BEFORE this block even runs!
    # This is because Metaclasses and __init_subclass__ execute at COMPILE/IMPORT time 
    # (when Python reads the module from top to bottom).
    
    print("\n================== SCRIPT EXECUTION START ==================\n")
    demonstrate_type_creation()
    demonstrate_anatomy()
    demonstrate_enforcement()
    demonstrate_registry()
    demonstrate_singleton()
    demonstrate_init_subclass()
    
    print("\n--- Tutorial Complete ---")
    print("Summary:")
    print("1. Metaclasses define HOW a class is constructed.")
    print("2. `__new__` creates the class, `__init__` initializes the class, `__call__` intercepts instantiation.")
    print("3. For most modern use cases (registry/validation), use `__init_subclass__` instead.")
