---
layout: post
title: Decorators
category: Technique
tags: "Python-3-Tutorial"
---
---
# Decorators
Two different kinds of decorators in Python:
* Function decorators
* Class decorators

A decorator in Python is any callable Python object **that is used to modify a function or a class**. **A reference** to a function "func" or a class "C" is passed to a decorator and the decorator returns a modified function or class. The modified functions or classes usually contain calls to the original function "func" or class 'C'.

## First Steps to Decorators
First we have to know or remember that **function names are references to functions** and that **we can assign multiple names to the same function**:

```python
>>> def succ(x):
...     return x + 1
... 
>>> successor = succ
>>> successor(10)
11
>>> succ(10)
11
```

	* We have two names, i.e. "succ" and "successor" for the same function.
	* We can delete either "succ" or "successor" without deleting the function itself, as following snippet code shows below


```python
>>> del succ
>>> successor(10)
11
```

## Functions inside Functions

```python
def f():
    
    def g():
        print("Hi, it's me 'g'")
        print("Thanks for calling me")
        
    print("This is the function 'f'")
    print("I am calling 'g' now:")
    g()
 

f()
```

Another example using "proper" return statements in the functions:

```python
def temperature(t):
    def celsius2fahrenheit(x):
        return 9 * x / 5 + 32

    result = "It's " + str(celsius2fahrenheit(t)) + " degrees!" 
    return result

print(temperature(20))
```

## Functions as Parameters
Due to the fact that **every parameter of a function is a reference to an object** and **functions are objects as well**, we can pass functions -or better "references to functions"- as parameters to a function.

```python
def g():
    print("Hi, it's me 'g'")
    print("Thanks for calling me")
    
def f(func):
    print("Hi, it's me 'f'")
    print("I will call 'func' now")
    func()
          
f(g)
```

Sometimes, we need to know what the 'real' name of func is. For this purpos, we can use the attribute `__name__`, as it contains this name.

## Functions returning Functions
**The output of a function is also a reference to an object.** Therefore functions can return references to function objects.

[polynomial "factory" function](https://www.python-course.eu/python3_decorators.php)
[Polynomials](https://www.python-course.eu/polynomial_class_in_python.php)

---

## A Simple Decorator

```python
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return function_wrapper

def foo(x):
    print("Hi, foo has been called with " + str(x))

print("We call foo before decoration:")
foo("Hi")
    
print("We now decorate foo with f:")
foo = our_decorator(foo)

print("We call foo after decoration:")
foo(42)
```

And the output is shown as below:

```python
We call foo before decoration:
Hi, foo has been called with Hi
We now decorate foo with f:
We call foo after decoration:
Before calling foo
Hi, foo has been called with 42
After calling foo
```

After the decoration `foo=our_decorator(foo)`, `foo` is a reference to the `function_wrapper`. `foo` will be called inside of `function_wrapper`, but before and after the call some additional code will be executed, i.e. in our case two print functions.

## The Usual Syntax for Decorators in Python
The decorations occurs in the line before the function header. The `@` is followed by the decorator function name.

```python
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return function_wrapper

@our_decorator
def foo(x):
    print("Hi, foo has been called with " + str(x))

foo("Hi")
```

It is also possible to decorate third party functions, e.g. functions imported from a module. It can't  use the Python syntax with the `@` sign in this case:

```python
from math import sin, cos

def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        res = func(x)
        print(res)
        print("After calling " + func.__name__)
    return function_wrapper

sin = our_decorator(sin)
cos = our_decorator(cos)

for f in [sin, cos]:
    f(3.1415)
```

Summarizing we can say that a decorator in Python is a callable Python object that is used to modify a function, method or class definition. The original object, the one which is going to be modified, is passed to a decorator as an argument. The decorator returns a modified object, e.g. a modified function, which is bound to the name in the definition.

A generalized version of function_wrapper, which accepts functions with arbitrary parameters. The example is shown as below:

```python
from random import random, randint, choice

def our_decorator(func):
    def function_wrapper(*args, **kwargs):
        print("Before calling " + func.__name__)
        res = func(*args, **kwargs)
        print(res)
        print("After calling " + func.__name__)
    return function_wrapper

random = our_decorator(random)
randint = our_decorator(randint)
choice = our_decorator(choice)

random()
randint(3, 8)
choice([4, 5, 6])
```

## Use Cases for Decorators
### 1. Checking Arguments with a Decorator
The following program uses a decorator function to ensure that the argument passed to the function factorial is a positive integer.

```python
def argument_test_natural_number(f):
    def helper(x):
        if type(x) == int and x > 0:
            return f(x)
        else:
            raise Exception("Argument is not an integer")
    return helper
    
@argument_test_natural_number
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)

for i in range(1,10):
	print(i, factorial(i))

print(factorial(-1))
```

### 2. Counting Function Calls with Decorators
The following example uses a decorator to count the number of times a function has been called.

```python
def call_counter(func):
    def helper(x):
        helper.calls += 1
        return func(x)
    helper.calls = 0

    return helper

@call_counter
def succ(x):
    return x + 1

print(succ.calls)
for i in range(10):
    succ(i)
    
print(succ.calls)
```

## Decorators with Parameters

## Using wraps from functools
The way we have defined decorators so far hasn't taken into account that the attributes
* `__name__` (name of the function)
* `__doc__` (the docstring)
* `__module__` (The module in which the function is defined)

of the original functions will be lost after the decoration.

The original attributes of the function f can be saved, if they are assigned inside of the decorator.

```python
def greeting(func):
    def function_wrapper(x):
        """ function_wrapper of greeting """
        print("Hi, " + func.__name__ + " returns:")
        return func(x)
    function_wrapper.__name__ = func.__name__
    function_wrapper.__doc__ = func.__doc__
    function_wrapper.__module__ = func.__module__
    return function_wrapper
```

Fortunately, we can import the decorator `wraps` from `functools` instead of add all assignment codes to keep the original attributes and decorate functions in the decorator with it:

```python
from functools import wraps

def greeting(func):
    @wraps(func)
    def function_wrapper(x):
        """ function_wrapper of greeting """
        print("Hi, " + func.__name__ + " returns:")
        return func(x)
    return function_wrapper
```


## Class instead of Functions
### 1. The `__call__` method
A callable object is an object which can be used and behaves like a function but not be a function. It is possible to define classes in a way that the instances will be callable objects. The `__call__` method is called, if the instance is called "like a function", i.e. using brackets.

```python
class A:
    
    def __init__(self):
        print("An instance of A was initialized")
    
    def __call__(self, *args, **kwargs):
        print("Arguments are:", args, kwargs)
              
x = A()
print("now calling the instance:")
x(3, 4, x=11, y=10)
print("Let's call it again:")
x(3, 4, x=11, y=10)
```

### 2. Using a Class as a Decorator
The following decorator can be rewrite as a class:

```python
def decorator1(f):
    def helper():
        print("Decorating", f.__name__)
        f()
    return helper

@decorator1
def foo():
    print("inside foo()")

foo()
```

The following decorator implemented as a class does the same "job":

```python
class decorator2:
    
    def __init__(self, f):
        self.f = f
        
    def __call__(self):
        print("Decorating", self.f.__name__)
        self.f()

@decorator2
def foo():
    print("inside foo()")

foo()
```
