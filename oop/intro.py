# ============================================================================
# COMPREHENSIVE PYTHON OBJECT-ORIENTED PROGRAMMING (OOP) TUTORIAL
# ============================================================================
# Target Audience: Developers with ~1.5 years of experience.
# Instructions: You can run this entire file. It is syntactically valid Python.
# Read through the comments and classes to understand the concepts.
# ============================================================================

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field

# ============================================================================
# 1. CORE OOP BASICS IN PYTHON
# ============================================================================

class CoreConcepts:
    """
    Python classes are blueprints for creating objects.
    Everything in Python is an object, including classes themselves (they are 
    instances of 'type').
    """
    
    # Class Attribute: Shared across all instances of the class.
    # Stored in the class's __dict__.
    species = "Homo Sapiens"

    def __init__(self, name: str, age: int):
        # Instance Attributes: Unique to each instance.
        # Stored in the instance's __dict__.
        self.name = name
        self.age = age

    def instance_method(self):
        """
        Standard instance method. The first parameter is always 'self', 
        which represents the specific object instance calling the method.
        """
        return f"{self.name} is {self.age} years old."


# ============================================================================
# 2. THE 4 PILLARS OF OOP
# ============================================================================

# ----------------------------------------------------------------------------
# Pillar 1: Encapsulation
# ----------------------------------------------------------------------------
# Bundling data and methods that operate on that data within a single unit (class).
# Restricting direct access to some of an object's components to prevent state mutation.
# Python doesn't have true private/protected keywords. It relies on conventions 
# and name mangling.

class BankAccount:
    def __init__(self, owner: str, balance: float):
        self.owner = owner               # Public attribute
        self._account_type = "Checking"  # Protected attribute (Convention: single underscore)
                                         # Indicates "internal use only", but not enforced.
        self.__balance = balance         # Private attribute (Name Mangling: double underscore)
                                         # Python renames this internally to _BankAccount__balance

    # Properties are the Pythonic way to implement getters and setters.
    # They wrap attribute access in methods, allowing logic (like validation) on assignment.
    @property
    def balance(self) -> float:
        """Getter for the private __balance attribute."""
        return self.__balance

    @balance.setter
    def balance(self, amount: float):
        """Setter with validation logic."""
        if amount < 0:
            raise ValueError("Balance cannot be negative.")
        self.__balance = amount

    def get_internal_balance(self):
        # Accessing the mangled variable inside the class works normally.
        return self.__balance


# ----------------------------------------------------------------------------
# Pillar 2: Abstraction
# ----------------------------------------------------------------------------
# Hiding complex implementation details and showing only the essential features.
# In Python, we use the `abc` (Abstract Base Classes) module.
# Abstract classes cannot be instantiated.

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        """
        Any child class MUST implement this method, or it will also be 
        considered abstract and cannot be instantiated.
        """
        pass

class StripeProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        # Complex Stripe API logic hidden from the user
        print(f"Processing ${amount} via Stripe API...")
        return True


# ----------------------------------------------------------------------------
# Pillar 3: Inheritance
# ----------------------------------------------------------------------------
# A class can derive attributes and methods from another class.
# Python supports single and multiple inheritance.

class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self):
        return "..."

class Dog(Animal):
    def __init__(self, name: str, breed: str):
        # super() delegates method calls to the parent or sibling class.
        # Here, we initialize the parent class's attributes.
        super().__init__(name)
        self.breed = breed

    def speak(self):
        # Method Overriding: Replacing the parent's implementation
        return "Woof!"

# Multiple Inheritance & MRO (Method Resolution Order)
# Python uses the C3 Linearization algorithm to determine the order 
# in which base classes are searched when looking for a method.
class Flyer:
    def move(self):
        return "Flying"

class Swimmer:
    def move(self):
        return "Swimming"

class Duck(Flyer, Swimmer):
    pass 
    # MRO determines which move() is called. 
    # Left-to-right priority: Duck -> Flyer -> Swimmer -> object.
    # Duck().move() will return "Flying".
    # You can view MRO via Duck.__mro__ or Duck.mro()


# ----------------------------------------------------------------------------
# Pillar 4: Polymorphism
# ----------------------------------------------------------------------------
# The ability of different objects to respond to the same method call in their own way.
# Python achieves this through Method Overriding (shown above) and "Duck Typing".
# Duck Typing: "If it walks like a duck and quacks like a duck, it's a duck."
# We don't care about the object's actual type, only that it has the required methods.

class Cat:
    def speak(self):
        return "Meow!"

def make_animal_speak(animal):
    # Polymorphism in action. 'animal' can be a Dog, Cat, or anything 
    # that implements a 'speak' method. No strict type checking needed.
    print(animal.speak())


# ============================================================================
# 3. SOLID PRINCIPLES
# ============================================================================
# Five design principles intended to make software designs more understandable, 
# flexible, and maintainable.

# ----------------------------------------------------------------------------
# S - Single Responsibility Principle (SRP)
# ----------------------------------------------------------------------------
# A class should have one, and only one, reason to change. 
# It should do exactly one thing.

# BAD:
# class UserReport:
#     def get_data(self): ...
#     def format_pdf(self): ...
#     def save_to_db(self): ... 
# (Handles logic, formatting, and persistence - 3 reasons to change)

# GOOD:
class UserDataFetcher:
    def get_data(self) -> dict:
        return {"user": "John"}

class ReportFormatter:
    def format_pdf(self, data: dict) -> str:
        return f"PDF Data: {data}"

class ReportRepository:
    def save(self, report: str):
        print("Saving to DB...")


# ----------------------------------------------------------------------------
# O - Open/Closed Principle (OCP)
# ----------------------------------------------------------------------------
# Software entities (classes, modules, functions) should be open for extension, 
# but closed for modification. 
# You should be able to add new functionality without altering existing code.

# BAD:
# class DiscountCalculator:
#     def calculate(self, customer_type, amount):
#         if customer_type == "regular": return amount * 0.9
#         elif customer_type == "vip": return amount * 0.8  
#         # Adding a new type requires modifying this class!

# GOOD: (Using Polymorphism/Strategy Pattern)
class DiscountStrategy(ABC):
    @abstractmethod
    def apply_discount(self, amount: float) -> float:
        pass

class RegularDiscount(DiscountStrategy):
    def apply_discount(self, amount: float) -> float:
        return amount * 0.9

class VIPDiscount(DiscountStrategy):
    def apply_discount(self, amount: float) -> float:
        return amount * 0.8

class Checkout:
    # We can inject any discount strategy. If we need a 'HolidayDiscount', 
    # we just create a new class. Zero modification to `Checkout`.
    def __init__(self, strategy: DiscountStrategy):
        self.strategy = strategy

    def calculate_total(self, amount: float) -> float:
        return self.strategy.apply_discount(amount)


# ----------------------------------------------------------------------------
# L - Liskov Substitution Principle (LSP)
# ----------------------------------------------------------------------------
# Objects of a superclass shall be replaceable with objects of its subclasses 
# without breaking the application.
# Subclasses must behave exactly like the parent class expects.

# BAD: The classic Square/Rectangle problem.
# If Square inherits from Rectangle, setting width to 5 and height to 10 
# on a Square violates the math of a Square, breaking expected Rectangle behavior.

# GOOD: Design based on behaviors, not real-world taxonomy.
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    def area(self) -> float:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: float):
        self.side = side
    def area(self) -> float:
        return self.side ** 2
# Now, anywhere a Shape is expected, both Rectangle and Square work perfectly.


# ----------------------------------------------------------------------------
# I - Interface Segregation Principle (ISP)
# ----------------------------------------------------------------------------
# Clients should not be forced to depend upon interfaces that they do not use.
# Keep interfaces small and specific.

# BAD:
# class Machine(ABC):
#     @abstractmethod
#     def print_doc(self): pass
#     @abstractmethod
#     def scan_doc(self): pass
#     @abstractmethod
#     def fax_doc(self): pass
# (A simple Printer class would be forced to implement scan and fax methods it doesn't support)

# GOOD:
class Printer(ABC):
    @abstractmethod
    def print_doc(self): pass

class Scanner(ABC):
    @abstractmethod
    def scan_doc(self): pass

class SimplePrinter(Printer):
    def print_doc(self):
        print("Printing...")

class MultiFunctionDevice(Printer, Scanner):
    def print_doc(self):
        print("Printing...")
    def scan_doc(self):
        print("Scanning...")


# ----------------------------------------------------------------------------
# D - Dependency Inversion Principle (DIP)
# ----------------------------------------------------------------------------
# 1. High-level modules should not depend on low-level modules. Both should depend on abstractions.
# 2. Abstractions should not depend on details. Details should depend on abstractions.
# TL;DR: Depend on Interfaces/ABCs, not concrete implementations (Dependency Injection).

# BAD:
# class EmailSender:
#     def send(self, msg): print("Sending email")
# class NotificationService:
#     def __init__(self):
#         self.sender = EmailSender() # Tightly coupled to concrete class!

# GOOD:
class MessageSender(ABC):
    @abstractmethod
    def send(self, msg: str): pass

class EmailService(MessageSender):
    def send(self, msg: str):
        print(f"Email sent: {msg}")

class SMSService(MessageSender):
    def send(self, msg: str):
        print(f"SMS sent: {msg}")

class NotificationService:
    # High-level module depends on the abstraction (MessageSender)
    # The actual implementation is injected at runtime.
    def __init__(self, sender: MessageSender):
        self.sender = sender

    def notify(self, msg: str):
        self.sender.send(msg)


# ============================================================================
# 4. ADVANCED PYTHON OOP ("And More")
# ============================================================================

# ----------------------------------------------------------------------------
# A. Class Methods vs Static Methods
# ----------------------------------------------------------------------------
class MathOperations:
    pi_approximation = 3.14

    @classmethod
    def change_pi(cls, new_pi: float):
        """
        Takes 'cls' (the class itself) instead of 'self'.
        Used to mutate class state, or as Alternative Constructors (Factory Methods).
        """
        cls.pi_approximation = new_pi

    @classmethod
    def from_string(cls, pi_string: str):
        """Alternative constructor example."""
        cls.pi_approximation = float(pi_string)
        return cls()

    @staticmethod
    def add(a: float, b: float) -> float:
        """
        Takes neither 'self' nor 'cls'.
        Behaves like a regular function, but belongs in the class's namespace 
        because it makes logical sense for it to live here. Cannot access instance/class state.
        """
        return a + b


# ----------------------------------------------------------------------------
# B. Magic / Dunder (Double Underscore) Methods
# ----------------------------------------------------------------------------
# Python's Data Model allows you to override built-in behavior using dunders.
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        """Human-readable string representation (called by print() and str())."""
        return f"Vector({self.x}, {self.y})"

    def __repr__(self):
        """Unambiguous developer representation (used in REPL, debugging)."""
        # Convention: Should look like valid Python code to recreate the object.
        return f"Vector(x={self.x}, y={self.y})"

    def __add__(self, other):
        """Operator overloading for the '+' symbol."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        """Operator overloading for the '==' symbol."""
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y

    def __len__(self):
        """Called by len(). Let's define it as the hypotenuse length rounded to int."""
        return int(math.hypot(self.x, self.y))

    def __call__(self, scalar: float):
        """Makes the object callable like a function: vec(5)."""
        return Vector(self.x * scalar, self.y * scalar)


# ----------------------------------------------------------------------------
# C. Context Managers (The 'with' statement)
# ----------------------------------------------------------------------------
class DatabaseConnection:
    def __enter__(self):
        """Setup logic before entering the 'with' block."""
        print("Connecting to DB...")
        return self # Often returns the resource itself

    def __exit__(self, exc_type, exc_val, traceback):
        """Teardown logic. Executes even if an exception was raised."""
        print("Closing DB connection...")
        if exc_type is not None:
            print(f"An exception occurred: {exc_val}")
        # Return True to swallow the exception, False to let it propagate
        return False


# ----------------------------------------------------------------------------
# D. Dataclasses
# ----------------------------------------------------------------------------
# Introduced in Python 3.7. Generates boilerplate code automatically 
# (__init__, __repr__, __eq__, etc.) based on type hints.

@dataclass(order=True) 
# order=True automatically generates __lt__, __le__, __gt__, __ge__ 
class InventoryItem:
    # Sort index used by the generated rich comparison methods
    sort_index: float = field(init=False, repr=False)
    
    name: str
    unit_price: float
    quantity_on_hand: int = 0 # Default value
    
    def __post_init__(self):
        """Called automatically right after the generated __init__."""
        # Let's say we want to order items by total value
        self.sort_index = self.unit_price * self.quantity_on_hand

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand


# ----------------------------------------------------------------------------
# E. Memory Optimization: __slots__
# ----------------------------------------------------------------------------
# By default, Python stores instance attributes in a dynamic dictionary (__dict__).
# This consumes more memory. If you have millions of objects, you can define 
# __slots__ to pre-allocate memory strictly for defined variables, preventing 
# the creation of __dict__ and '__weakref__'.

class Point:
    __slots__ = ['x', 'y'] # Restricts attributes to exactly these two.

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        # self.z = 10 # This would throw an AttributeError: 'Point' object has no attribute 'z'


# ----------------------------------------------------------------------------
# F. Metaclasses
# ----------------------------------------------------------------------------
# A metaclass is the "class of a class". It defines how a class behaves.
# Just as an object is an instance of a class, a class is an instance of a metaclass.
# Default metaclass is 'type'. You can intercept class creation.

class SingletonMeta(type):
    """
    A metaclass for the Singleton design pattern.
    Ensures only one instance of a class ever exists.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # Intercept the instantiation process
        if cls not in cls._instances:
            # If not created, call the super() (type) to allocate memory and init
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseClient(metaclass=SingletonMeta):
    def __init__(self):
        print("Initializing Database Client...")


# ============================================================================
# EXECUTION / DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("--- Encapsulation ---")
    account = BankAccount("Alice", 1000)
    print(account.balance)
    account.balance = 1500 # Uses setter
    # print(account.__balance) # Raises AttributeError
    print(account._BankAccount__balance) # Mangled access (Frowned upon, but works)

    print("\n--- MRO & Inheritance ---")
    d = Duck()
    print("Duck moves by:", d.move())
    print("Duck MRO:", [cls.__name__ for cls in Duck.__mro__])

    print("\n--- SOLID: Dependency Inversion ---")
    email_service = EmailService()
    notifier = NotificationService(email_service)
    notifier.notify("Server is down!")

    print("\n--- Dunder Methods ---")
    v1 = Vector(2, 4)
    v2 = Vector(3, 1)
    v3 = v1 + v2
    print("Vector addition:", v3)
    print("Callable vector:", v3(2)) # Multiplies by scalar

    print("\n--- Context Managers ---")
    with DatabaseConnection() as db:
        print("Performing DB operations...")
        # Raising an error here would still trigger __exit__

    print("\n--- Dataclasses ---")
    item1 = InventoryItem("Laptop", 1000.0, 5)
    item2 = InventoryItem("Mouse", 20.0, 50)
    print(item1) # Notice __repr__ is automatically clean
    print("Item1 > Item2?", item1 > item2) # Compares based on sort_index (total value)

    print("\n--- Metaclasses (Singleton) ---")
    db1 = DatabaseClient()
    db2 = DatabaseClient()
    print("Are both DB instances the same object in memory?", db1 is db2)
