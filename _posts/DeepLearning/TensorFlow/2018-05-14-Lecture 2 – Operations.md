---
layout: post
title: Lecture_2 -- Operations
category: 深度学习
tags: TensorFlow, CS20
---

# Lecture 2 -- Operations

---
## 1. TensorBoard: Visualizing Learning
> The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. To make it easier to understand, debug, and optimize TensorFlow programs, we've included a suite of visualization tools called TensorBoard.

To visualize the program with TensorBoard, we need to write log files of the program. To write event files, we first need to create a writer for those logs, using this code:

```
writer = tf.summary.FileWriter([logdir], [graph])
```

You can either call it using `tf.get_default_graph()`, which returns the default graph of the program, or through `sess.graph`, which returns the graph the session is handling. The latter requires you to already have created a session.

Next, go to Terminal, run the program. And then open browser and go to http://localhost:6006/ 

```
$ python3 [my_program.py] 
$ tensorboard --logdir="./graphs" --port 6006
```

* **Note:** If you've run your code several times, there will be multiple event files in your [logdir]. TF will show only the latest graph and display the warning of multiple event files. To get rid of the warning, delete the event files you no longer need.

---
## 2. Constant op
[Constants, Sequences, and Random Values](https://www.tensorflow.org/api_guides/python/constant_op)
* `tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)`
* `tf.zeros(shape, dtype=tf.float32, name=None)`
* `tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)`
* `tf.ones(shape, dtype=tf.float32, name=None)`
* `tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)`
* `tf.fill(dims, value, name=None)`
* `tf.lin_space(start, stop, num, name=None)`
* `tf.range([start], limit=None, delta=1, dtype=None, name='range')`
* `tf.random_normal`
* `tf.truncated_normal`
* `tf.random_uniform`
* `tf.random_shuffle`
* `tf.random_crop`
* `tf.multinomial`
* `tf.random_gamma`
* `tf.set_random_seed`

---
## 3. Math Operations
[Math](https://www.tensorflow.org/api_guides/python/math_ops)

---
## 4. Data Types
* Python Native Types
* TensorFlow Native Types
	* A full list of TensorFlow data type, [tf.DType class](https://www.tensorflow.org/api_docs/python/tf/DType)
* Numpy Data Types

> Most of the times, you can use TensorFlow types and NumPy types interchangeably.

---
## 5. Variables
Differences between a constant and a variable:
1. A constant is, well, constant. Often, you’d want your weights and biases to be updated during training.
2. A constant's value is stored in the graph and replicated wherever the graph is loaded. A variable is stored separately, and may live on a parameter server.

---
## 6. Interactive Session
> You sometimes see `InteractiveSession` instead of `Session`. The only difference is an InteractiveSession makes itself the default session so you can call `run()` or `eval()` without explicitly call the session.

---
## 7. Control Dependencies
> Sometimes, we have two or more independent ops and we'd like to specify which ops should be run first. In this case, we use tf.Graph.control_dependencies([control_inputs]).

---
## 8. Importing Data
### 8.1 The old way: `place_holder` and `feed_dict`
### 8.2 The new way: tf.data

---
## 9. The trap of lazy loading
> Lazy loading is a term that refers to a programming pattern when you defer declaring/initializing an object until it is loaded. In the context of TensorFlow, it means you defer creating an op until you need to compute it.

> There are two ways to avoid this bug. First, always separate the definition of ops and their execution when you can. But when it is not possible because you want to group related ops into classes, you can use Python @property to ensure that your function is only loaded once when it's first called. This is not a Python course so I won't dig into how to do it. But if you want to know, check out [this wonderful blog post](https://danijar.com/structuring-your-tensorflow-models/) by Danijar Hafner.

