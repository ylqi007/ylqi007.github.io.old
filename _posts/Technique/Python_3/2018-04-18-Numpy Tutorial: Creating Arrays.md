---
layout: post
title: Numpy-Creating-Arrays
category: Technique
tags: "Python-3-Tutorial"
---

# [Numpy Tutorial: Creating Arrays](https://www.python-course.eu/numpy_create_arrays.php)
@(Numerical Python)[python, numpy]

## 1. Creation of Arrays with Evenly Spaced Values
### [numpy.arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)

```python
arange([start,] stop[, step], [, dtype=None])
```

`np.arange()` returns an `ndarray `rather than a list iterator as `range` does.

### [numpy.linspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)

```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

If the optional parameter `retstep` is set, the function will also return the value of the spacing between adjacent values. So, the function will return a tuple.

```python
import numpy as np
samples, spacing = np.linspace(1, 10, retstep=True)
```

## 2. Zero-dimensional Arrays in Numpy
It's possible to create multidimensional arrays in numpy. Scalars are zero dimensional. Applying the ndim to the scalar, we get the dimension of the array.

```python
import numpy as np
x = np.array(42)
print("x: ", x)
print("The type of x: ", type(x))		# numpy.ndarray
print("The dimension of x:", np.ndim(x))	# 0
```

## 3. One-dimensional Arrays in Numpy

```python
F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
V = np.array([3.4, 6.9, 99.8, 12.8])
```

## 4. Two- and Multidimensional Arrays

Multidimensional arrays can be created by passing nested lists or tuples to the array method of numpy.


## 5. Shape of an Array
The function `shape` returns the shape of an array. The shape is a tuple of integers. These numbers denote the lengths of the corresponding array dimension.

```python
x = np.array([ [67, 63, 87],
               [77, 69, 59],
               [85, 87, 99],
               [79, 72, 71],
               [63, 89, 93],
               [68, 92, 78]])
print(np.shape(x))
(6, 3)
```

There is also an equivalent array property:

```python
print(x.shape)
(6, 3)
```

`shape` can also be used to change the shape of an array.

```python
x.shape(3,6)
print(x)
x.shape = (2, 9)
print(x)
```

And the result is

```python
[[67 63 87 77 69 59]
 [85 87 99 79 72 71]
 [63 89 93 68 92 78]]

[[67 63 87 77 69 59 85 87 99]
 [79 72 71 63 89 93 68 92 78]]
```

The shape of a scalar is an empty tuple:

```python
x = np.array(11)
print(np.shape(x))
```

## 6. Indexing and Slicing
Assigning to and accessing the elements of an array is similar to other sequential data types of Python, i.e. `lists` and `tuples`.

1. Single indexing behaves the way.

```python
A = np.array([ [3.4, 8.7, 9.9], 
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])
A[1], A[-1]
A[1][0]
```

We have to be aware of the fact, that way of accessing multi-demensional `A[1][0]` can be highly inefficient. The reason is that we have to an intermediate array `A[1]` from which we access the element `A[1][0]`. It behaves similar to this:

```python
tmp = A[1]
print(tmp)
print(tmp[0])
```

There is another way to access elements of multi-dimensional arrays in Numpy: only one pair of square brackets and all the indices  are separated by commas:

```python
print(A[1, 0])
```

**slicing** of lists `lists` and `tuples`. The syntax is the same in numpy for one-dimensional arrays, and it can applied to multiple dimensions as well.
The general syntax for a one-dimensional array A looks like this:

```python
A[start:stop:step]
```

Slicing for multidimensional is illustrated as following:

```python


A = np.array([
[11, 12, 13, 14, 15],
[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45],
[51, 52, 53, 54, 55]])
print(A[:3, 2:])
```

The third parameters `step` can also be used in slicing. The `reshape` function is used to construct the two-dimensional array.

```python
X = np.arange(28).reshape(4, 7)
print(X)
```

This code snippet shows
```python
[[ 0  1  2  3  4  5  6]
 [ 7  8  9 10 11 12 13]
 [14 15 16 17 18 19 20]
 [21 22 23 24 25 26 27]]
```

```python
print(X[::2, ::3])
[[ 0  3  6]
 [14 17 20]]
```
 
 ![Alt text|center](/public/img/posts/python_3_tutorial/NumpyArrays/pic_1.png)

If the number of objects in the selection is less than the dimension N, then `:` is assumed for any subsequent dimensions:

```python
A = np.array(
    [ [ [45, 12, 4], [45, 13, 5], [46, 12, 6] ], 
      [ [46, 14, 4], [45, 14, 5], [46, 11, 5] ], 
      [ [47, 13, 2], [48, 15, 5], [52, 15, 1] ] ])
A[1:3, 0:2]  # equivalent to A[1:3, 0:2, :]
```

Attention: Whereas slicings on lists and tuples create new objects, a slicing operation on an array creates a view on the original array. So we get an another possibility to access the array, or better a part of the array. From this follows that if we modify a view, the original array will be modified as well.

```python
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
S = A[2:6]
S[0] = 22
S[1] = 23
print(A)
[ 0  1 22 23  4  5  6  7  8  9]
```

Doing the similar thing with lists, we can see that we get a copy:

```python
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
lst2 = lst[2:6]
lst2[0] = 22
lst2[1] = 23
print(lst)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

If you want to check, if two array names share the same memory block, you can use the function `np.may_share_memory`.

```python
np.may_share_memory(A, B)
```

## 7. Arrays of Ones and of Zeros
```python
import numpy as np
E = np.ones((2,3))
print(E)
F = np.ones((3,4),dtype=int)
print(F)
```

## 8. Copying Arrays

### numpy.copy()

`numpy.copy(a, order='K')`

### numpy.ndarray.copy

` ndarray.copy(order='C')`

## 9. Identity Array
### The identity Function

### The eye Function

`eye(N, M=None, k=0, dtype=float)`

```python
import numpy as np
np.eye(5, 8, k=1, dtype=int)

array([[0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0]])
```




