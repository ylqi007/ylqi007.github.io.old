---
layout: post
title: Numpy-Introduction
category: Technique
tags: "Python-3-Tutorial"
---

# [Numpy Introduction](https://www.python-course.eu/numpy.php)
@(Numerical Python)[python, numpy]

NumPy is an acronym for "Numeric Python" or "Numerical Python". It is an open source extension module for Python.

SciPy (Scientific Python) is often mentioned in the same breath with NumPy. SciPy extends the capabilities of NumPy with further useful functions for minimization, regression, Fourier-transformation and many others.

Both NumPy and SciPy are usually not installed by default. NumPy has to be installed before installing SciPy. 

Python in combination with Numpy, Scipy and Matplotlib can be used as a replacement for MATLAB.
![Alt text | center](/public/img/posts/python_3_tutorial/NumpyIntroduction/pic_1.png)

## 1. Comparison between Core Python and Numpy
When we say "Core Python", we mean Python without any special modules, i.e. especially without NumPy.

The advantages of **Core Python**:
* high-level number objects: integers, floating point
* containers: lists with cheap insertion and append methods, dictionaries with fast lookup

Advantages of using **Numpy with Python**:
* array oriented computing
* efficiently implemented multi-dimensional arrays
* designed for scientific computation

## 2. A Simple Numpy Example
We have a list with values, e.g. temperatures in Celsius, and turn the list `cvalues` into one-dimensional numpy array:

```python
cvalues = [20.1, 20.8, 21.9, 22.5, 22.7, 22.3, 21.8, 21.2, 20.9, 20.1]
C = np.array(cvalues)	# c in an instance of the class numpy.ndarray
```

Let's assume, we want to turn the values into degrees Fahrenheit. This is very easy to accomplish with a numpy array.

```python
print(C * 9 / 5 + 32)
[ 20.1  20.8  21.9  22.5  22.7  22.3  21.8  21.2  20.9  20.1]
```

Compared to solution with a numpy array, the solution for Python list looks awkward:

```python
fvalues = [ x*9/5 + 32 for x in cvalues] 
print(fvalues)
[68.18, 69.44, 71.42, 72.5, 72.86, 72.14, 71.24000000000001, 70.16, 69.62, 68.18]
```

## 3. Memory Consumption: ndarray and list
The main benefits of using numpy array should be smaller memory consumption and better runtime behaviour.

![Alt text | center](/public/img/posts/python_3_tutorial/NumpyIntroduction/pic_2.png)

To calculate the memory consumption of the list from the above picture, we can use the function `getsizeof` from the module `sys`.

```python
from sys import getsizeof as size
lst = [24, 12, 57]
size_of_list_object = size(lst)   # only green box
size_of_elements = len(lst) * size(lst[0]) # 24, 12, 57
total_list_size = size_of_list_object + size_of_elements
print("Size without the size of the elements: ", size_of_list_object)
print("Size of all the elements: ", size_of_elements)
print("Total size of list, including elements: ", total_list_size)
```

And the output of the snippet of above:

```
Size without the size of the elements:  88
Size of all the elements:  84
Total size of list, including elements:  172
```

The size of a Python list consists of the general list information, the size needed for the references to the elements and the size of all the elements of the list. If we apply `sys.getsizeof` to a list, we get only the size of without the size of the elements.

For every new element, we need another eight bytes for the reference to the new object. The new integer object itself consumes 28 bytes. The size of an empty list `list` is 64 bytes.

The memory consumption of a `numpy.array`.

![Alt text | center](/public/img/posts/python_3_tutorial/NumpyIntroduction/pic_3.png)

We can create the numpy array of the previous diagram and calculate the memory usage:

```python
a = np.array([24, 12, 57])
print(size(a))
120
```

Get the memory usage for the general array information by creating an empty array:

```python
e = np.array([])
print(size(e))
96
```

We can see that the difference between the empty array `e` and the array `a` with three integers consists in 24 Bytes. This means that an arbitrary integer array of length $n$ in numpy needs
$96+n\times 8$ Bytes
whereas a list of integers needs
$64 + 8\times len(lst) + len(lst)\times 28$ Bytes
This is a minimum estimation, as Python integers can use more than 28 bytes.

When we define a Numpy array, numpy automatically chooses a fixed integer size, the default `int64`. We can determine the size of the integers, when we define an array.

```python
a = np.array([24, 12, 57], np.int8)
print(size(a) - 96)
a = np.array([24, 12, 57], np.int16)
print(size(a) - 96)
a = np.array([24, 12, 57], np.int32)
print(size(a) - 96)
a = np.array([24, 12, 57], np.int64)
print(size(a) - 96)
```

And the result of above snippet code is

```python
3
6
12
24
```

## 4. Time Comparison between Python List and Numpy Arrays
One of the main advantages of Numpy is its advantage in time compared to standard Python. 
Define the following functions:

```python
import time
size_of_vec = 1000
def pure_python_version():
    t1 = time.time()
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = [X[i] + Y[i] for i in range(len(X)) ]
    return time.time() - t1
def numpy_version():
    t1 = time.time()
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y
    return time.time() - t1
```

Call these functions and see the time consumption:

```python
t1 = pure_python_version()
t2 = numpy_version()
print(t1, t2)
print("Numpy is in this example " + str(t1/t2) + " faster!")
```

And the result is 

```
0.00025272369384765625 7.367134094238281e-05
Numpy is in this example 3.43042071197411 faster!
```

It's an easier and above all better way to measure the times by using the `timeit` module. The code for using the class `Timer` class is in this [tutorial](https://www.python-course.eu/numpy.php).


