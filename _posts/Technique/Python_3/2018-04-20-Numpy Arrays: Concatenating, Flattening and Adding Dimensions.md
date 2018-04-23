---
layout: post
title: Numpy Arrays--Concatenating, Flattening and Adding Dimensions
category: Technique
tags: "Python-3-Tutorial"
---

# Numpy Arrays: Concatenating, Flattening and Adding Dimensions

## Flatten and Reshape Arrays
There are two methods to flatten a multidimensional array:
* `flatten()`
* `ravel()`

### 1. flatten
`flatten` is a ndarray method with an optional keyword parameter "order". order can have the values "C", "F" and "A". 
1. The default of order is "C". "C" means to flatten C style in row-major ordering, i.e. the rightmost index "changes the fastest" or in other words: In row-major order, the row index varies the slowest, and the column index the quickest, so that a[0,1] follows [0,0].
2. "F" stands for Fortran column-major ordering. 
3. "A" means preserve the the C/Fortran ordering.

```python
import numpy as np
A = np.array([[[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7]],
              [[ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15]],
              [[16, 17],
               [18, 19],
               [20, 21],
               [22, 23]]])
Flattened_X = A.flatten()
print(Flattened_X)
print(A.flatten(order="C"))
print(A.flatten(order="F"))
print(A.flatten(order="A"))
```

The output is

```python
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
[ 0  8 16  2 10 18  4 12 20  6 14 22  1  9 17  3 11 19  5 13 21  7 15 23]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
```

### 2.ravel
The order of the elements in the array returned by ravel() is normally "C-style".
`ravel(a, order='C')`
ravel returns a flattened one-dimensional array. A copy is made only if needed.
The optional keyword parameter "order" can be 'C','F', 'A', or 'K'
`C`: C-like order, with the last axis index changing fastest, back to the first axis index changing slowest. "C" is the default!
`F`: Fortran-like index order with the first index changing fastest, and the last index changing slowest.
`A`: Fortran-like index order if the array "a" is Fortran contiguous in memory, C-like order otherwise.
`K`: read the elements in the order they occur in memory, except for reversing the data when strides are negative.

```python
print(A.ravel())
print(A.ravel(order="A"))
print(A.ravel(order="F"))
print(A.ravel(order="A"))
print(A.ravel(order="K"))
```

And the output is 

```python
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
[ 0  8 16  2 10 18  4 12 20  6 14 22  1  9 17  3 11 19  5 13 21  7 15 23]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
```

### 3. reshape
The method `reshape()` gives a new shape to an array without changing its data, i.e. it returns a new array with a new shape.
`reshape(a, newshape, order='C')`
Parameter 	Meaning
`a` 			array_like, Array to be reshaped.
`newshape` 	int or tuple of ints
`order`		'C', 'F', 'A', like in flatten or ravel

## Concatenating Arrays
Concatenate three one-dimensional arrays to one array.

```python
x = np.array([11,22])
y = np.array([18,7,6])
z = np.array([1,3,5])
c = np.concatenate((x,y,z))
print(c)
[11 22 18  7  6  1  3  5]
```

If we are concatenating multidimensional arrays, we can concatenate the arrays according to axis. Arrays must have the same shape to be concatenate with `concatenate()`. In the case of multidimensional arrays, we can arrange them according to the axis. The default value is `axis=0`.

## Adding New Dimensions
New dimensions can be added to an array by using `slicing` and `np.newaxis`.

```python
x = np.array([2,5,18,14,4])
y = x[:, np.newaxis]
print(y)
```

The output is

```python
[[ 2]
 [ 5]
 [18]
 [14]
 [ 4]]
```

## Vector Stacking
`np.row_stack()`
`np.column_stack()`
`np.dstack()`

## Repeating Patterns, The `tile` Method
`tile(A, reps)`
An array is constructed by repeating A the number of given by `reps`.
`reps` is usually a tuple (or list) which defines the number of repetitions along the corresponding axis/directions.

If `A.ndim<n`, `A` is promoted to be n-dimensional by prepending new axes. So a shape (5,) array is promoted to (1, 5) for 2-D replication, or shape (1, 1, 5) for 3-D replication. If this is not the desired behavior, promote `A` to n-dimensions manually before calling this function.
If `A.ndim > d'`, `reps` is promoted to `A.ndim` by pre-pending 1's to it.
Thus for an array 'A' of shape (2, 3, 4, 5), a 'reps' of (2, 2) is treated as (1, 1, 2, 2).

```python
import numpy as np
x = np.array([[1, 2], [3, 4]])
print(np.tile(x, 2))
import numpy as np
x = np.array([[1, 2], [3, 4]])
print(np.tile(x, (2, 1)))
import numpy as np
x = np.array([[1, 2], [3, 4]])
print(np.tile(x, (2, 2)))
```
 
