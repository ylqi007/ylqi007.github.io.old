---
layout: post
title: Lecture note 4 How to structure your model in TensorFlow
category: DeepLearning
tags: TensorFlow
---
# Lecture note 4: How to structure your model in TensorFlow

We've implemented two simple models in TensorFlow:
* Linear regression => the number of fire and theft in the city of Chicago
* Logistic regression => an Optical Character Recognition task on the MNIST dataset.

However, complex models would require better planning, otherwise our models would be pretty messy and hard to debug. And we will be doing this through an example: word2vec.

In short, we need **a vector representation of words** so that we can input them into our neural networks to do some magic tricks.
* Skip-gram V.S. CBOW (Continuous Bag-of-Words) [refer to the note 4]()
	* Skip-gram predicts source context-words from the center words. **???**
	* CBOW predicts center words from context word. **???**

An explanation/tutorial to the skip-gram model: [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
In the skip-gram model, to get the vector representations of words, which is the weights of the hidden layer. These weights are actually the **word vectors** or **embedding matrix**.

## Softmax, Negative Sampling and Noise Constructive Estimation
Softmax is too computationally because of the normalization. So implementing skip-gram model with **negative sampling**, one of sampling-based approaches. Negative sampling is actually a simplified model of an approach called **Noise Constructive Estimation (NCE)**, e.g. negative sampling makes certain assumption about the number of noise samples to generate (k) and the distribution of noise samples (Q) to simplify computation (negative sampling assumes that $k Q(w)=1$).
   * While negative sampling is useful for the learning word embeddings, it doesn’t have the theoretical guarantee that its derivative tends towards the gradient of the softmax function, which makes it not so useful for language modeling.
   * NCE has this nice theoretical guarantees that negative sampling lacks as the number of noise
samples increases. In this example, NCE will be used because of its nice theoretical guarantee.

## About dataset
**text8** is the first 100 MB of cleaned text of the English Wikipedia dump on Mar. 3, 2006 (whose
link is no longer available). 

## Interface: How to structure your TensorFlow model
The models more or less have the same structure. 
* **Phase 1**: assemble your graph
	1. Define placeholders for input and output
		* Input is the center word and output is the target (context) word. Instead of using one-hot vectors, we input the index of those words directly.
	2. Define the weights (in this case, embedding matrix) 
		* Each row corresponds to the representation vector of one word. If one word is represented with a vector of size EMBED_SIZE, then the embedding matrix will have shape [VOCAB_SIZE, EMBED_SIZE]. We initialize the embedding matrix to value from a random distribution.
	3. Define the inference model
	4. Define loss function
		* TensorFLow has already implemented NCE.
	5. Define optimizer
* **Phase 2**: execute the computation
Create a session then within the session, use the feed_dict to feed inputs and outputs into the placeholders, run the optimizer to minimize the loss, and fetch the loss value back to us.
Which is basically training your model. There are a few steps:
	1. Initialize all model variables for the first time.
	2. Feed in the training data. Might involve randomizing the order of data samples.
	3. Execute the inference model on the training data, so it calculates for each training input
example the output with the current model parameters.
	4. Compute the cost
	5. Adjust the model parameters to minimize/maximize the cost depending on the model.

## Name Scope
Tensorboard doesn't know which nodes are similar to which nodes and should be grouped together. This setback can grow to be extremely daunting when building complex models with hundreds of ops.
TensorFlow can group all ops related together with **name scope.**
```python
withtf.name_scope(name_of_that_scope):
    define 1
    define 2
    ...
```
There are two kinds of edges:
* solid line, represent data flow edges.
* dotted arrow represent control dependence edges.

## Class
In order to make the model most easy to resue, taking advantage of Python's object-oriented-ness, **to build the model as a class.**

Visualize with t-SNE. (t-distributed stochastic neighbor embedding is a machine learning algorith for dimensionality reduction develop by Geoffrey Hinton and Laurens van der Maaten. It is a nonlinear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can be then visualized in a scatter plot.)


---
[Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
[word2vec_no_frills.py](https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/2017/examples/04_word2vec_no_frills.py)
