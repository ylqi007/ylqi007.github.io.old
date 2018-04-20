---
layout: post
title: Numerical Operations on Numpy Arrays
category: Technique
tags: "Python-3-Tutorial"
---

## Matrices v.s. Two-Dimensional Arrays
Some may have taken two-dimensional arrays of Numpy as matrices. This is principially all right, because they behave in most aspects like our mathematical idea of a matrix. We even saw that we can perform matrix multiplication on them. Yet, there is a subtle difference . There are "real" matrices in Numpy. They are a subset of the two-dimensional arrays. We can turn a two-dimensional array into a matrix by applying the `mat` function. The main difference shows, if you multiply two two-dimensional arrays or two matrices. We get real matrix multiplication by multiplying two matrices, but the two-dimensional arrays will be only multiplied component-wise:

```python
import numpy as np
A = np.array([ [1, 2, 3], [2, 2, 2], [3, 3, 3] ])
B = np.array([ [3, 2, 1], [1, 2, 3], [-1, -2, -3] ])
R = A * B
print(R)
```

The output of snippet code above is

```python
[[ 3  4  3]
 [ 2  4  6]
 [-3 -6 -9]]
```

```python
MA = np.mat(A)
MB = np.mat(B)
R = MA * MB
print(R)
```

The output of snippet code above is

```python
[[ 2  0 -2]
 [ 6  4  2]
 [ 9  6  3]]
```

## Comparison Operators
If we compare two arrays, we don't get a simple `True` or `False` as a return value. The comparisons are performed elementswise.

```python
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([ [11, 102, 13], [201, 22, 203], [31, 32, 303] ])
A == B
```

The output of snippet code above is

```python
array([[ True, False,  True],
       [False,  True, False],
       [ True,  True, False]], dtype=bool)
```

It is possible to compare complete arrays for equality as well. `array_equal` is used for this purpose. `array_equal` returns `True` if two arrays have the same shape and elements, otherwise `False` will be returned.

```python
print(np.array_equal(A, B))	# False
print(np.array_equal(A, A))	# True
```

## Applying Operators on Arrays with Different Shapes
Two basic cases with basic operators like `+` or `*`:
* an operator applied to an array and a scalar
* an operator applied to two arrays of the same shape

Let's see some cases that we can apply operators on arrays, if they have different shapes.

### 1. Broadcasting
Numpy provides a powerful mechanism, called **Broadcasting**, which allows to perform arithmetic operations on arrays of different shapes. This means that we have **a smaller array** and **a larger array**, and we transform or apply the smaller array multiple times to perform some operation on the larger array. In other words: Under certain conditions, the smaller array is "broadcasted" in a way that it has the same shape as the larger array.

1. First example of broadcasting

```python
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([1, 2, 3])
print("Multiplication with broadcasting: ")
print(A * B)
print("... and now addition with broadcasting: ")
print(A + B)
```

The output of snippet code above is

```python
Multiplication with broadcasting: 
[[11 24 39]
 [21 44 69]
 [31 64 99]]
... and now addition with broadcasting: 
[[12 14 16]
 [22 24 26]
 [32 34 36]]
```

The following diagram illustrates the way of working of broadcasting:

![Alt text | center](/public/img/posts/python_3_tutorial/NumpyOperations/pic_1.png)

2. The Second Example:

Turn a row vector into a column vector:

```python
B = np.array([1, 2, 3])
B[:, np.newaxis]
```

The output is

```python
array([[1],
       [2],
       [3]])
```

Now we are capable of doing the multiplication using broadcasting:

```python
A * B[:, np.newaxis]
```

![Alt text | center](/public/img/posts/python_3_tutorial/NumpyOperations/pic_2.png)

3. The Third Example:

```python
A = np.array([10, 20, 30])
B = np.array([1, 2, 3])
A[:, np.newaxis]
```

The output is

```python
array([[10],
       [20],
       [30]])
```

```python
A[:, np.newaxis] * B
```

The previous code returned the following result:

```python
array([[10, 20, 30],
       [20, 40, 60],
       [30, 60, 90]])
```

![Alt text | center](/public/img/posts/python_3_tutorial/NumpyOperations/pic_3.png)

### 2. Another Way to Do it
1. Doing it without broadcasting:

```python
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([1, 2, 3])
B = B[np.newaxis, :]
B = np.concatenate((B, B, B))
print("Multiplication: ")
print(A * B)
print("... and now addition again: ")
print(A + B)
```

The output is

```python
Multiplication: 
[[11 24 39]
 [21 44 69]
 [31 64 99]]
... and now addition again: 
[[12 14 16]
 [22 24 26]
 [32 34 36]]
```

2. Using 'tile':

```python
import numpy as np
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.tile(np.array([1, 2, 3]), (3, 1))
print(B)
print("Multiplication: ")
print(A * B)
print("... and now addition again: ")
print(A + B)
```

The output is

```python
[[1 2 3]
 [1 2 3]
 [1 2 3]]
Multiplication: 
[[11 24 39]
 [21 44 69]
 [31 64 99]]
... and now addition again: 
[[12 14 16]
 [22 24 26]
 [32 34 36]]
```

## Distance Matrix
A practical example of a distance matrix is a distance matrix between geographic locations, in our example Eurpean cities:

```python
cities = ["Barcelona", "Berlin", "Brussels", "Bucharest",
          "Budapest", "Copenhagen", "Dublin", "Hamburg", "Istanbul",
          "Kiev", "London", "Madrid", "Milan", "Moscow", "Munich",
          "Paris", "Prague", "Rome", "Saint Petersburg", 
          "Stockholm", "Vienna", "Warsaw"]
dist2barcelona = [0,  1498, 1063, 1968, 
                  1498, 1758, 1469, 1472, 2230, 
                  2391, 1138, 505, 725, 3007, 1055, 
                  833, 1354, 857, 2813, 
                  2277, 1347, 1862]
dists =  np.array(dist2barcelona[:12])
print(dists)
print(np.abs(dists - dists[:, np.newaxis]))
```

The output is

```python
[   0 1498 1063 1968 1498 1758 1469 1472 2230 2391 1138  505]
[[   0 1498 1063 1968 1498 1758 1469 1472 2230 2391 1138  505]
 [1498    0  435  470    0  260   29   26  732  893  360  993]
 [1063  435    0  905  435  695  406  409 1167 1328   75  558]
 [1968  470  905    0  470  210  499  496  262  423  830 1463]
 [1498    0  435  470    0  260   29   26  732  893  360  993]
 [1758  260  695  210  260    0  289  286  472  633  620 1253]
 [1469   29  406  499   29  289    0    3  761  922  331  964]
 [1472   26  409  496   26  286    3    0  758  919  334  967]
 [2230  732 1167  262  732  472  761  758    0  161 1092 1725]
 [2391  893 1328  423  893  633  922  919  161    0 1253 1886]
 [1138  360   75  830  360  620  331  334 1092 1253    0  633]
 [ 505  993  558 1463  993 1253  964  967 1725 1886  633    0]]
```

## 3-Dimensional Broadcasting

```python
A = np.array([ [[3, 4, 7], [5, 0, -1] , [2, 1, 5]],
      [[1, 0, -1], [8, 2, 4], [5, 2, 1]],
      [[2, 1, 3], [1, 9, 4], [5, -2, 4]]])
B = np.array([ [[3, 4, 7], [1, 0, -1], [1, 2, 3]] ])
B * A
```

The output is

```python
array([[[ 9, 16, 49],
        [ 5,  0,  1],
        [ 2,  2, 15]],
       [[ 3,  0, -7],
        [ 8,  0, -4],
        [ 5,  4,  3]],
       [[ 6,  4, 21],
        [ 1,  0, -4],
        [ 5, -4, 12]]])
```

The second example,

```python
B = np.array([1, 2, 3])
B = B[np.newaxis, :]
print(B.shape)
B = np.concatenate((B, B, B)).transpose()
print(B.shape)
B = B[:, np.newaxis]
print(B.shape)
print(B)
print(A * B)
```

The output is 

```python
(1, 3)
(3, 3)
(3, 1, 3)
[[[1 1 1]]
 [[2 2 2]]
 [[3 3 3]]]
[[[ 3  4  7]
  [ 5  0 -1]
  [ 2  1  5]]
 [[ 2  0 -2]
  [16  4  8]
  [10  4  2]]
 [[ 6  3  9]
  [ 3 27 12]
  [15 -6 12]]]
```
