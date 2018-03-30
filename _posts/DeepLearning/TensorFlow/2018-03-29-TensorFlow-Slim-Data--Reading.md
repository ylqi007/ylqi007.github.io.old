---
layout: post
title: TensorFlow-Slim-Data--Reading
category: DeepLearning
tags: TensorFlow
---

# TensorFlow-Slim-Data -- Reading
TF-Slim providers a data loading library for facilitating the reading of data from various formats.


## 1. Overview
The task of loading data has two main components: 
1. specification of how a dataset is represented so it can be read and interpreted;
2. instruction for providing the data to consumers of the dataset.

One must specify instructions for how the data is actually provided and housed in memory. For example, if the data is sharded over many sources, should it be read in parallel from these sources? Should it be read serially? Should the data be shuffled memory?


## 2. Dataset Specification
TF-Slim's [dataset](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/dataset.py) is a tuple that encapsulates the following elements of a dataset specification:

* `data_sources`: A list of file paths that together make up the dataset
* `reader`: A TensorFlow Reader appropriate for the file type in data_sources.
* `decoder`: A TF-Slim data_decoder class which is used to decode the content of the read dataset files.
* `num_samples`: The number of samples in the dataset.
* `items_to_descriptions`: A map from the items provided by the dataset to descriptions of each.

In a nutshell, a dataset is read by 
1. Opening the files specified by `data_sources` using the given `reader` class 
2. Decoding the files using the given `decoder`
3. Allowing the user to request a list of `items` to be returned as `Tensors`.


## 3. Data Decoders
A `data_decoder` is a class which is given some data, which is possibly serialized/encoded, and returns a list of `Tensors`.

```python
# Load the data
my_encoded_data = ...
data_decoder = MyDataDecoder()

# Decode the inputs and labels:
decoded_input, decoded_labels = data_decoder.Decode(data, ['input', 'labels'])

# Decode just the inputs:
decoded_input = data_decoder.Decode(data, ['input'])

# Check which items a data decoder knows how to decode:
for item in data_decoder.list_items():
	print(item)
```


## 4. Example: TFExampleDecoder
A `TFExample` protocol buffer is a map from *keys (strings)* to either a `tf.FixedLenFeature` or `tf.VarLenFeature`.

Consequently, to decode a `TFExample`, one must provide a mapping from one or more `TFExample` fields to each of the `items` that the `tfexample_decoder` can provide.
The `tfexample_decoder` is constructed by specifying a map of `TFExample` keys as well as a set of `ItemHandlers`. A `TFExample` maps a set of keys to either `tf.FixedLenFeature` or `tf.VarLenFeature` and an `ItemHandler` provides a mapping from `TFExample` keys to the item being provided.

Because a `tfexample_decoder` might return multiple `items`, one often constructs a `tfexample_decoder` using multiple `ItemHandlers`.

```python
keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
}

items_to_handlers = {
    'image': tfexample_decoder.Image(
      image_key = 'image/encoded',
      format_key = 'image/format',
      shape=[28, 28],
      channels=1),
    'label': tfexample_decoder.Tensor('image/class/label'),
}

decoder = tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)
```

In this example, the `TFExample` is parsed using three keys: `image/encoded`, `image/format` and `image/class/label`. Additionally, the first two keys are mapped to a single `item` named **image**, and the third key is mapped to a `item` named **label**. As defined, this `data_decoder` provides two `items` named *image* and *label*.


## 5. Data Provision
A [data_provider](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/data_provider.py) is a class which provides `Tensors` for each item requested:

```python
my_data_provider = ...
image, class_label, bounding_box = my_data_provider.get(
    ['image', 'label', 'bb'])
```

The [dataset_data_provider](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/dataset_data_provider.py) is a `data_provider` that provides data from a given dataset specification:

```python
dataset = GetDataset(...)
data_provider = dataset_data_provider.DatasetDataProvider(
    dataset, common_queue_capacity=32, common_queue_min=8)
```

The `dataset_data_provider` enables control over several elements of data provision:
* How many concurrent readers are used.
* Whether the data is shuffled as its loaded into its queue.
* Whether to take a single pass over the data or read data indefinitely.
