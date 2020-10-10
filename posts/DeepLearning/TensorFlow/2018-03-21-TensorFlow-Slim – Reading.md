---
layout: post
title: TensorFlow-Slim--Reading Notes
category: 深度学习
tags: TensorFlow
---

# TensorFlow-Slim -- Reading Notes

I read through the whole doc, and keep some useful notes.
[Tensorflow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim#restoring-models-with-different-variable-names)
TF-Slim is a lightweight library for defining, training and evaluating complex models in TensorFlow.

---
## 1. Defining Models
Models can be succinctly defined using TF-Slim by combining its **variables**, **layers** and **scopes**.

### 1.1 Variables
* To create a `Variable` in native tensorflow:
	* Either a predefined value of an initialization mechanism.
	* If a variable needs to be created on a specific device, the specification must be [made explicit](https://www.tensorflow.org/programmers_guide/using_gpu).
* TF-Slim provides a set of thin wrapper functons in [variables.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/framework/python/ops/variables.py) which allow callers to easily define variables.
* Note that in native TensorFlow, there are two types of variables: regular variables and local (transient) variables.
	* **Regular variables**, can be saved to disk using a saver.
	* **Local variables** are those variables that only exist for the duration of a session and are not saved to disk.
* TF-Slim further differentiates variables by defining **model variables**, which are variables that represent parameters of a model. Model variables are trained or fine-tuned during learning and are loaded from a checkpoint during evaluation or inference.

When you create a model variable via TF-Slim's layers or directly via the `slim.model_variable` function, TF-Slim adds the variable to the `tf.GraphKeys.MODEL_VARIABLES` collection.

### 1.2 Layers
* A Convolutional Layer in a neural network is composed of several low level operations:
	1. Creating the weight and bias variables
	2. Convolving the weights with the input from the previous layer
	3. Adding the biases to the result of the convolution.
	4. Applying an activation function.
```
input = ...
with tf.name_scope('conv1_1') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias, name=scope)
```
Actually, this need to duplicate the code repeatedlly.

* TF-Slim provides a number of convenient operations defined at the more abstract level of neural network layers.
```
input = ...
net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')
```

* TF-Slim also provides two meta-operations called `repeat` and `stack` that allow users to repeatedly perform the same operation.
```
# Perform several convolutions to define 3 layers.
net = ...
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```
One way to reduce this code duplication, use `for` loop:
```
# Use for loop to define 3 layers.
net = ...
for i in range(3):
  net = slim.conv2d(net, 256, [3, 3], scope='conv3_%d' % (i+1))
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```
This can be made even cleaner by using TF-Slim's `repeat` operation:
```
# Use repeat to create 3 layers.
net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```
* Furthermore, TF-Slim's `slim.stack` operator allows a caller to repeatedly apply the same operation with different arguments to create a **stack** or tower of layers.

### 1.3 Scopes
In addition to the types of scope mechanisms in TensorFlow ([name_scope](https://www.tensorflow.org/api_docs/python/tf/name_scope), [variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope)). TF-Slim adds a new scoping mechanism called [arg_scope](https://www.tensorflow.org/api_docs/python/tf/contrib/framework/arg_scope). This new scope allows a user to specify one or more operations and a set of arguments which will be passed to each of the operations defined in the `arg_scope`. 
Consider the following code snippet:
```
net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')
```
It should be clear that these three convolution layers share many of the same hyperparameters. This code contains a lot of repeated values that should be factored out. One solution would be to specify default values using variables:
```
padding = 'SAME'
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)
net = slim.conv2d(inputs, 64, [11, 11], 4,
                  padding=padding,
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv1')
net = slim.conv2d(net, 128, [11, 11],
                  padding='VALID',
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv2')
net = slim.conv2d(net, 256, [11, 11],
                  padding=padding,
                  weights_initializer=initializer,
                  weights_regularizer=regularizer,
                  scope='conv3')
```
This solution ensures that all three convolutions share the exact same parameter values but doesn't reduce completely the code clutter. By using an `arg_scope`, we can both ensure that each layer uses the same values and simplify the code:
```
with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2') # overrides padding with the value of 'VALID'.
    net = slim.conv2d(net, 256, [11, 11], scope='conv3')
```
As the example illustrates, the use of arg_scope makes the code cleaner, simpler and easier to maintain.

### 1.4 Working Example: Specifying the VGG16 Layers
By combining TF-Slim Variables, Operations and scopes, a normally very complex network can be implemented with very few lines of codes. For example, the entire [VGG](https://www.robots.ox.ac.uk/%7Evgg/research/very_deep/) architecture can be defined with just with the following snippet:
```
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net
```

---
## 2. Training Models
Training Tensorflow models requires **a model**, **a loss function**, **the gradient computation** and **a training routine** that iteratively computes the gradients of the model weights relative to the loss and updates the weights accordingly. TF-Slim provides both common loss functions and a set of helper functions that run the training and evaluation routines.

### 2.1 Losses
***The loss function defines a quantity that we want to minimize.***
* For **classification problems**, this is typically the cross entropy between the true distribution and the predicted probability distribution across classes.
* For **regression problems**, this is often the sum-of-squares differences between the predicted and true values.

TF-Slim provides an easy-to-use mechanism for defining and keeping track of loss functions via the [`losses`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/losses/python/losses/loss_ops.py) module. 
* A simple case to train the VGG network:
```python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg
#Load the images and labels.
images, labels = ...
#Create the model.
predictions, _ = vgg.vgg_16(images)
#Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy(predictions, labels)
```

* A case with multi-task model that produces multiple outputs:
```python
#Load the images and labels.
images, scene_labels, depth_labels = ...
# Create the model.
scene_predictions, depth_predictions = CreateMultiTaskModel(images)
#Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
#The following two lines have the same effect:
total_loss = classification_loss + sum_of_squares_loss
total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
```
In this example, we have two losses which we add by calling `slim.losses.softmax_cross_entropy` and `slim.losses.sum_of_squares`. We can obtain the total loss by adding them together (`total_loss`) or by calling `slim.losses.get_total_loss()`. How did this work? When you create a loss function via TF-Slim, TF-Slim adds the loss to a special TensorFlow collection of loss functions.

* [loss_ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/losses/python/losses/loss_ops.py) also has a function that adds the **custom loss function** to TF-Slims collection.
```
# Load the images and labels.
images, scene_labels, depth_labels, pose_labels = ...
# Create the model.
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)
# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.
# The following two ways to compute the total loss are equivalent:
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss
# (Regularization Loss is included in the total loss by default).
total_loss2 = slim.losses.get_total_loss()
```
In this example, we can again either produce the total loss function manually or let TF-Slim know about the additional loss and let TF-Slim handle the losses.

### 2.2 Training Loop
TF-Slim provides a simple but powerful set of tools for training models found in [learning.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py). These include a Train function that repeatedly measures the loss, computes gradients and saves the model to disk, as well as several convenience functions for manipulating gradients.
For example, once we've specified the model, **the loss function** and **the optimization scheme**, we can call `slim.learning.create_train_op` and `slim.learning.train` to perform the optimization:
```
g = tf.Graph()

# Create the model and specify the losses...
...

total_loss = slim.losses.get_total_loss()
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are applied too.
train_op = slim.learning.create_train_op(total_loss, optimizer)
logdir = ... # Where checkpoints are stored.

slim.learning.train(
    train_op,
    logdir,
    number_of_steps=1000,
    save_summaries_secs=300,
    save_interval_secs=600):
```
In this example, slim.learning.train is provided with the `train_op` which is used to (a) **compute the loss** and (b) **apply the gradient step**.

### 2.3 Working Example: Training the VGG16 Model
To illustrate this, lets examine the following sample of training the VGG network:
```
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg

...

train_log_dir = ...
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
  # Set up the data loading:
  images, labels = ...

  # Define the model:
  predictions = vgg.vgg_16(images, is_training=True)

  # Specify the loss function:
  slim.losses.softmax_cross_entropy(predictions, labels)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total_loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)
```

## 3 Fine-Tuning Existing Models
### 3.1 Brief Recap on Restoring Variables from a Checkpoint
After a model has been trained, it can be restored using tf.train.Saver() which restores Variables from a given checkpoint.
```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to restore all the variables.
restorer = tf.train.Saver()

# Add ops to restore some variables.
restorer = tf.train.Saver([v1, v2])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```
See [Restoring Variables](https://www.tensorflow.org/programmers_guide/variables#restoring-variables) and [Choosing which Variables to Save and Restore](https://www.tensorflow.org/programmers_guide/variables#choosing-which-variables-to-save-and-restore) sections of the [Variables](https://www.tensorflow.org/programmers_guide/variables) page for more details.

### 3.2 Partially Restoring Models
It is often desirable to fine-tune a pre-trained model on an entirely new dataset or even a new task. In thse situations, one can use TF-Slim's helper functions to select a subset of variables to restore:
```
# Create some variables.
v1 = slim.variable(name="v1", ...)
v2 = slim.variable(name="nested/v2", ...)
...

# Get list of variables to restore (which contains only 'v2'). These are all
# equivalent methods:
variables_to_restore = slim.get_variables_by_name("v2")
# or
variables_to_restore = slim.get_variables_by_suffix("2")
# or
variables_to_restore = slim.get_variables(scope="nested")
# or
variables_to_restore = slim.get_variables_to_restore(include=["nested"])
# or
variables_to_restore = slim.get_variables_to_restore(exclude=["v1"])

# Create the saver which will be used to restore the variables.
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
```

### 3.3 Restoring models with different variable names
* When restoring variables from a checkpoint, the Saver locates the variable names in a checkpoint file and maps them to variables in the current graph. Above, we created a saver by passing to it a list of variables. In this case, the names of the variables to locate in the checkpoint file were implicitly obtained from each provided variable's var.op.name.
* Sometimes, we want to restore a model from a checkpoint whose variables have different names to those in the current graph. In this case, we must provide the Saver a dictionary that maps from each checkpoint variable name to each graph variable. Consider the following example where the checkpoint variables names are obtained via a simple function:

```python
# Assuming than 'cov1/weights' should be restored from 'vgg16/conv1/weights'
def name_in_checkpoint(var):
  return 'vgg16/' + var.op.name
# Assuming than 'conv1/weights' and 'conv1/bias' should be restored from 'conv1/params1' and 'conv1/params2'
def name_in_checkpoint(var):
  if "weights" in var.op.name:
    return var.op.name.replace("weights", "params1")
  if "bias" in var.op.name:
    return var.op.name.replace("bias", "params2")

variables_to_restore = slim.get_model_variables()
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore} 
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
```
The code in line 13, creating a dictionary that maps from **each checkpoint variable name** to **each graph variable**.

### 3.4 Fine-Tuning a Model on a different task
Consider the case where we have a pre-trained VGG16 model. The model was trained on the ImageNet dataset, which has 1000 classes. However, we would like to apply it to the Pascal VOC dataset which has only 20 classes. To do so, we can *initialize our new model using the values of the pre-trained model* **excluding the final layer**:

```
# Load the Pascal VOC data
image, label = MyPascalVocDataLoader(...)
images, labels = tf.train.batch([image, label], batch_size=32)
# Create the model
predictions = vgg.vgg_16(images)
train_op = slim.learning.create_train_op(...)
# Specify where the Model, trained on ImageNet, was saved.
model_path = '/path/to/pre_trained_on_imagenet.checkpoint'
# Specify where the new model will live:
log_dir = '/path/to/my_pascal_model_dir/'
# Restore only the convolutional layers:
variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)
# Start training.
slim.learning.train(train_op, log_dir, init_fn=init_fn)
```
In this snippet of code, line 17 restores variables exclude `fc6`, `fc7` and `fc8`.

---
## 4 Evaluating Models
Once we've trained a model (or even while the model is busy training) we'd like to see how well the model performs in practice. This is accomplished **by picking a set of evaluation metrics**, which will grade the models performance, and **the evaluation code** which actually loads the data, performs inference, compares the results to the ground truth and records the evaluation scores. This step may be performed once or repeated periodically.

### 4.1 Metrics
We define a **metric** to be a performance measure that is not a loss function (losses are directly optimized during training), but which we are still interested in for the purpose of evaluating our model.
TF-Slim provides a set of metric operations that makes evaluating models easy. Abstractly, computing the value of a metric can be divided into three parts:
1. Initialization: initialize the variables used to compute the metrics.
2. Aggregation: perform operations (sums, etc) used to compute the metrics.
3. Finalization: (optionally) perform any final operation to compute metric values. For example, computing means, mins, maxes, etc.

The following example demonstrates the API for declaring metrics. Because metrics are often evaluated on a test set which is different from the training set (upon which the loss is computed), we'll assume we're using test data:
```
images, labels = LoadTestData(...)
predictions = MyModel(images)

mae_value_op, mae_update_op = slim.metrics.streaming_mean_absolute_error(predictions, labels)
mre_value_op, mre_update_op = slim.metrics.streaming_mean_relative_error(predictions, labels)
pl_value_op, pl_update_op = slim.metrics.percentage_less(mean_relative_errors, 0.3)
```
As the example illustrates, the creation of a metric returns two values: a `value_op` and an `update_op`. The `value_op` is an idempotent operation that returns the current value of the metric. The `update_op` is an operation that performs the aggregation step mentioned above as well as returning the value of the metric.
Keeping track of each `value_op` and `update_op` can be laborious. To deal with this, TF-Slim provides two convenience functions:
``` python
# Aggregates the value and update ops in two lists:
value_ops, update_ops = slim.metrics.aggregate_metrics(
    slim.metrics.streaming_mean_absolute_error(predictions, labels),
    slim.metrics.streaming_mean_squared_error(predictions, labels))

# Aggregates the value and update ops in two dictionaries:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
})
```

### 4.2 Working example: Tracking Multiple Metrics
```python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg


# Load the data
images, labels = load_data(...)

# Define the network
predictions = vgg.vgg_16(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
})

# Evaluate the model using 1000 batches of data:
num_batches = 1000

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  for batch_id in range(num_batches):
    sess.run(names_to_updates.values())

  metric_values = sess.run(names_to_values.values())
  for metric, value in zip(names_to_values.keys(), metric_values):
    print('Metric %s has value: %f' % (metric, value))
```

### 4.3 Evaluation Loop
TF-Slim provides an evaluation module ([evaluation.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/evaluation.py)), which contains helper functions for writing model evaluation scripts using metrics from the [metric_ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/metrics/python/ops/metric_ops.py) module. These include a function for periodically running evaluations, evaluating metrics over batches of data and printing and summarizing metric results. For example:
```python
import tensorflow as tf

slim = tf.contrib.slim

# Load the data
images, labels = load_data(...)

# Define the network
predictions = MyModel(images)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    'accuracy': slim.metrics.accuracy(predictions, labels),
    'precision': slim.metrics.precision(predictions, labels),
    'recall': slim.metrics.recall(mean_relative_errors, 0.3),
})

# Create the summary ops such that they also print out to std output:
summary_ops = []
for metric_name, metric_value in names_to_values.iteritems():
  op = tf.summary.scalar(metric_name, metric_value)
  op = tf.Print(op, [metric_value], metric_name)
  summary_ops.append(op)

num_examples = 10000
batch_size = 32
num_batches = math.ceil(num_examples / float(batch_size))

# Setup the global step.
slim.get_or_create_global_step()

output_dir = ... # Where the summaries are stored.
eval_interval_secs = ... # How often to run the evaluation.
slim.evaluation.evaluation_loop(
    'local',
    checkpoint_dir,
    log_dir,
    num_evals=num_batches,
    eval_op=names_to_updates.values(),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=eval_interval_secs)
```






