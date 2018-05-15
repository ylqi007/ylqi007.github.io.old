---
layout: post
title: Lecture_3 -- Linear and Logistic Regression
category: 深度学习
tags: TensorFlow, CS20
---

# Lecture 3: Linear and Logistic Regression in TensorFlow

---
## 1. Linear Regression: Predict life expectancy from birth rate


---
## 2. Control flow: Huber loss
Several outliers pull the fitted line towards them, making the model perform worse. One way to deal with outliers is to use Huber loss. Intuitively, squared loss has the disadvantage of giving too much weights to outliers (you square the difference - the larger the difference, the larger its square). Huber loss was designed to give less weight to outliers. [Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)

This basically means that if the condition is true, use the true function. Else, use the false function.

```
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)
```

Control flow ops defined by TensorFlow, [The full list of Control Flow](https://www.tensorflow.org/versions/master/api_guides/python/control_flow_ops)

---
## 3. tf.data
### 3.1 tf.data.Dataset
> With `tf.data`, instead of storing our input data in non-TensorFlow object, we store it in a `tf.data.Dataset` object. We can create a Dataset from tensors with:

```
tf.data.Dataset.from_tensor_slices((features, labels))
```

> `features` and `labels` are supposed to be tensors, but remember that since **TensorFlow** and **numpy** are seamlessly integrated, they can be **Numpy** arrays.

### 3.2 tf.data.Iterator
> After we have turned our data into a magical `Dataset` object, we can iterate through samples in this `Dataset` using an `iterator`. An `iterator` iterates through the `Dataset` and returns a new sample or batch each time we call `get_next()`.


---
## 4. Optimizers
* Why is optimizer in the fetches list of `tf.Session.run()`?
* How does TensorFlow know what variables to update?
> optimizer is an op whose job is to minimize loss. To execute this op, we need to pass it into the list of fetches of tf.Session.run(). When TensorFlow executes optimizer, it will execute the part of the graph that this op depends on. In this case, we see that optimizer depends on loss, and loss depends on inputs X,  Y, as well as two variables weights and bias. 

[Optimizers--official documentation](https://www.tensorflow.org/api_guides/python/train)
[A pretty great comparison of these optimizers](http://ruder.io/optimizing-gradient-descent/)

---
## 5. Logistic Regression with MNIST
["mini-batch size" in Bengio's practical tips](https://arxiv.org/pdf/1206.5533v2.pdf)
