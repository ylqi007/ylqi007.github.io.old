---
layout: post
title: Keyword Argument Unpacking
category: Technique
tags: "Python-3-Tutorial"
---

# [What is the name of ** in python?](https://softwareengineering.stackexchange.com/questions/131403/what-is-the-name-of-in-python)

> It should be called **the keyword argument unpacking syntax**.

If you have a list of arguments, `*args`, it's called **"argument unpacking"**, in the same manner `**kwargs` is called **"keyword argument unpacking"**.

If you use it on the left hand side of an `=`, as in `a, *middle, end = my_tuple`, you'd say **"tuple unpacking"**.

In total, there are three types of (single parameter) arguments:

```
def f(x)			# x: positional argument
def f(x, y=0)		# y: keyword argument
def f(x, *xs, y=0)	# y: keyword-only argument
```

The `*args` argument is called the **"variable positional parameter"** and `**kwargs` is the **"variable keyword parameter"**. Keyword-only arguments can't be given positionally, because a variable positional parameter will take all of the arguments you pass.

So the `*` and `**` arguments just unpack their respective data structures:

```
args = (1, 2, 3)  # usually a tuple, always an iterable[1]
f(*args) → f(1, 2, 3)
# and 
kwargs = {"a": 1, "b": 2, "c": 3}  # usually a dict, always a mapping*
f(**kwargs) -> f(a=1, b=2, c=3)
```

[1]: Iterables are objects that implement the `__iter__()` method and mappings are objects that implement `keys()` and `__getitem__()`. Any object that supports this protocol will be understood by the constructors `tuple()` and `dict()`, so they can be used for unpacking arguments.
