---
layout: post
title: Numpy--Broadcasting
category: Technique
tags: "Python-3-Tutorial"
---

# [Broadcasting](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html)

## General Broadcasting Rules
When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
1. they are equal, or
2. one of them is 1

If these conditions are not met, a `ValueError: frames are not aligned` exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.

In the following example, both the `A` and `B` arrays have axes with length one that are expanded to a larger size during the broadcast operation:

```
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

> From the above example, lining up the sizes of the trailing axes of these arrays according to the broadcast rules. The last dimensions of `A` and `B` are 1 and 5 separately, therefore the length of this dimension broadcast to the larger one, which is 5. The same for the second from last dimensions that will broadcast to 6, and the third from the last to 7.

![Alt text](/public/img/posts/python_3_tutorial/NumpyBroadcasting/fig_1.png)

Figure 1: In the simplest example of broadcasting, the scalar `b` is stretched to become an array of with the same shape as `a` so the shapes are compatible for element-by-element multiplication.

![Alt text](/public/img/posts/python_3_tutorial/NumpyBroadcasting/fig_2.png)

Figure 2: A two dimensional array multiplied by a one dimensional array results in broadcasting if number of 1-d array elements matches the number of 2-d array columns.

![Alt text](/public/img/posts/python_3_tutorial/NumpyBroadcasting/fig_3.png)

Figure 3: When the trailing dimensions of the arrays are unequal, broadcasting fails because it is impossible to align the values in the rows of the 1st array with the elements of the 2nd arrays for element-by-element addition. 

![Alt text](/public/img/posts/python_3_tutorial/NumpyBroadcasting/fig_4.png)

Figure 4: In some cases, broadcasting stretches both arrays to form an output array larger than either of the initial arrays. 

