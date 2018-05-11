---
layout: post
title: Lecture_1 -- Introduction to TensorFlow
category: 深度学习
tags: TensorFlow, CS20
---

# Lecture 1: Introduction to Tensorflow

## 1. TensorFlow Basics
The rst thing we need to understand about TensorFlow is its computation graph approach. Any
TensorFlow program consists of two phases:
**Phase 1:** Assemble a graph;
**Phase 2:** Use a session to execute operations in the graph.

* Data Flow Graphs
	* TensorFlow separates definition of computations from their execution.

* Tensor
	* 0-d tensor: scalar (number)
	* 1-d tensor: vector
	* 2-d tensor: matrix
	* and so on

* TensorFlow = tensor + flow = data + flow

* How to get the value of a variable $a$ ?
	* Create a **session**, assign it to variable $sess$ so we can call it later. Within the session, evaluate the graph to fetch the value of $a$.

* `tf.Session()`
	* A Session object encapsulates the environment in which *Operation objects* are executed, and *Tensor objects* are evaluated.
	* Session will also allocate memory to store the current values of variables.

* Subgraphs
	* Possible to break graphs into several chunks and run them parallelly across multiple CPUs, GPUs, TPUs or other devices.

* Distributed Computation
	* To put part of a graph on a specific CPU or GPU.

* **BUG ALERT!**
	* Multiple graphs require multiple sessions, each will try to use all available resources by default.
	* Can't pass data between them without passing them through python/numpy, which doesn't work in distributed.
	* **It's better to have disconnected subgraphs within one graph.**

* Do not mix default graph and user created graphs.

* Why graphs?
	* Save computation. Only run subgraphs that lead to the values you want to fetch.
	* Break computation into small, differential pieces to facilitate auto-differentiation.
	* Facilitate distributed computation, spread the work across multiple CPUs, GPUs, TPUs, or other devices.
	* Many common machine learning models are taught and visualized as directed graphs.

## 2. TensorFlow API -- Graph
TensorFlow uses a **dataflow graph** to represent your computation in terms of the dependencies between individual operations.
	1. Define the dataflow graph.
	2. Create a TensorFlow **session** to run parts of the graph across a set of local and remote devices.
  
  
### Why dataflow graphs?
* **Dataflow** is a common programming model for parallel computing.
	1. The **nodes** represent units of computation;
	2. The **edges** represent the data consumed or produced by a computation.

* **Dataflow** has several advantages:
	1. Parallelism
	2. Distributed execution
	3. Compilation
	4. Portability

### What is a **tf.Graph** ?
A `tf.Graph` contains two relevant kinds of information:
* **Graph structure.** The nodes and edges of the graph, indicating how individual operations are composed together, **but not prescribing how they should be used.**
* **Graph collections.** TensorFlow provides a general mechanism for storing collections of metadata in a `tf.Graph`.
	* The `tf.add_to_collection` function enables you to associate a list of objects with a key (where `tf.GraphKeys` defines some of the standard keys);
	* The `tf.get_collection` enables you to look up all objects associated with a key;
	* eg1. When you create a `tf.Variable`, it is added by default to collections representing **global variables** and **trainable variables**.
	* eg2. When you later come to create a `tf.train.Saver` or `tf.train.Optimizer`, the variables in these collections are used as the default arguments.

### Building a **tf.Graph**
Most TensorFlow programs start with a dataflow graph construction phase. In this phase, you invoke TensorFlow API functions that construct new `tf.Operation` (node) and `tf.Tensor` (edge) objects and add them to a `tf.Graph` instance.
TensorFlow provides a **default graph** that is an implicit argument to all API functions in the same context.

### Naming operations
A `tf.Graph` object defines a **namespace** for the `tf.Operation` objects it contains. TensorFlow automatically chooses a unique name for each operation in your graph, but giving operations descriptive names can make your program easier to read and debug.
The TensorFlow API provides two ways to override the name of an operation:
1. Each API function that creates a new `tf.Operation` or returns a new `tf.Tensor` accepts an optional name argument.
	* `tf.constant(42.0, name="answer")`creates a new `tf.Operation` named **answer** and returns a `tf.Tensor` named **answer:0**.
2. The `tf.name_scope` function makes it possible to add a **name scope** prefix to all operations created in a particular context. The current name scope prefix is a "/" - delimited list of the names of all active **tf.name_scope** context managers.

Note that `tf.Tensor` objects are implicitly named after the `tf.Operation` that produces the tensor as output. A tensor name has the form `<OP_NAME>:<i>` where:
* `<OP_NAME>` is the name of the operation that produces it.
* `<i>` is an integer representing the index of that tensor among the operation's outputs.

### Placing operatons on different devices
If you want your TensorFlow program to use multiple different devices, the tf.device function provides a convenient way to request that all operations created in a particular context are placed on the same device (or type of device).
A device specification has the following form:
`/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>`

### Tensor-like objects
Many TensorFlow operations take one or more `tf.Tensor` objects as arguments.
For convenience, these functions will accept a **tensor-like object** in place of a `tf.Tensor`, and implicitly convert it to a `tf.Tensor` using the `tf.convert_to_tensor` method. Tensor-like objects include elements of the following types:
* `tf.Tensor`
* `tf.Variable`
* `numpy.ndarray`
* list (and lists of tensor-like objects)
* Scalar Python types: bool, float, int, str

## 3. TensorFlow API -- Session
### Executing a graph in a **tf.Session**
TensorFlow uses the `tf.Session` class to represent a connection between the client program and the C++ runtime.

### Creating a **tf.Session**
Since a `tf.Session` owns physical resources (such as GPUs and network connections), it is typically used as a context manager (in a with block) that automatically closes the session when you exit the block. 
It is also possible to create a session without using a with block, but you should explicitly call `tf.Session.close` when you are finished with it to free the resources.

`tf.Session.init` accepts three optional arguments:
* **target.**
* **graph.** By default, a new `tf.Session` will be bound to--and only able to run operations in--the current default graph.
* **config.** This argument allows you to specify a `tf.ConfigProto` that controls the behavior of the session.

### Using **tf.Session.run** to execute operations
The `tf.Session.run` method is the main mechanism for running a `tf.Operation` or evaluating a `tf.Tensor`.

## 4. TensorFlow API -- Visualizing graph
The **graph visualizer** is a component of TensorBoard that renders the structure of your graph visually in a browser. The easiest way to create a visualization is to pass a `tf.Graph` when creating the `tf.summary.FileWriter`

[TensorBoard Tutorial](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)

## 5. Programming with multiple graphs
When training a model, a common way of organizing your code is to use one graph for training your model, and a separate graph for evaluating or performing inference with a trained model.
