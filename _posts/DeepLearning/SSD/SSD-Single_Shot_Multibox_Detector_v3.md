---
layout: post
title: SSD-Single_Shot_Multibox_Detector 阅读笔记_v3
category: DeepLearning
tags: "Deep-Learning"
---

# SSD-Single_Shot_Multibox_Detector_v3

## Abstract
> Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location.
> At prediction time, the network generates **scores** for the presence of each object category in each default box and produces **adjustments** to the box to better match the object shape.
> Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.
> SSD is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stages and encapsulates.

## 1. Introduction
> Current state-of-the-art object detection systems are variants of the following approach: hypothesize bounding boxes, resample pixels or features for each box, and apply a high quality classifier. [I think attributes referring to Faster-RCNN]
> This paper presents the first deep network based object detector that does not resample pixels or features for bounding box hypotheses and is as accurate as approaches that do. This results in a significant improvement in speed for high-accuracy detection.
> The fundamental improvement in speed comes from eliminating bounding box proposals and the subsequent pixel or feature resampling stage.
> Our improvements include **1)** using a small convolutional filter to predict object categories and offsets in bounding box locations, **2)**using separate predictors(filters) for different aspect ratio detections, and **3)**applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple stage. **Especially using multiple layers for prediction at different scales.**
> **Summarization of contributions**
> 1. Faster than YOLO and significantly more accurate, in fact as accurate as Faster R-CNN
> 2. The core of SSD is predicting **category scores** and **box offsets** for *a fixed set of default bounding boxes* using small convolutional filters applied to feature map.
> 3. To achieve high detection accuracy we produce predictions of **different scales from feature maps of different scales**, and explicitly separate predictions by aspect ratio.
> 4. These design features lead to simple end-to-end training and high accuracy, even on low resolution input images, further improving the speeding vs accuracy trade-off.

---
## 2. The Single Shot Detector (SSD)
![@Fig.1 SSD framework | center](1520629305225.png)
> In a convolutional fashion, we evaluate a small set (e.g. 4) of default boxes of different ascpect ratios at each location in several feature maps with both the shape offsets and the confidences for all object categores($(c_{1}, c_{2},...,c_{p})$).
> At training time, we first match these default boxes to the ground truth boxes. In this sample image, we have matched two default boxes with the cat and dog, which are treated as positives and the rest as negatives. The model loss is a weighted sum between localization loss(e.g. Smooth L1) and confidence loss(e.g. Softmax).

### 2.1 Model
> The SSD approach in based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections.
> **base network** + **auxiliary structure** to produce detection with the following key features:
> 1. **Multi-scale feature maps for detection.** We add convolutional feature layers to the end of the truncated base network. These layers decrease in size progressively and allow predictions of detections at multiple scales.
> 2. **Convolutional predictors for derection.** Each added feature layer (or optionally an existing feature layer from the base network) can produce a fixed set of detection predictions using a set of convolutional filters.
> 3. **Default boxes and aspect ratios.** We associate a set of default bounding boxes with **each feature map cell**, for multiple feature as the top the network. **The default boxes** tile the feature map **in a convolution manner**, so that the position of each box relative to its corresponding cell is fixed. At each feature map cell, we predict the **offsets** relative to the default box shapes in the cell, as well as the **per-class scores** that indicate the presence of a class instance in each of those boxes. Specaifically, for each box out of $k$ at a given location, we compute $c$ class scores and the 4 offsets relative to the original default box shape. The default boxes are similar to the $anchor\ boxes$ used in Faster R-CNN, however we apply them to several feature maps of different resolutions.

### 2.2 Training
> The key difference between training SSD and training a typical detetor that uses a region proposals, is that ground truth information needs to be assigned to specific outputs in the fixed set of detector outputs.
> Training also involves choosing the set of default boxes and scales for detection as well as the hard negative mining and data augmentation strategies.
> **Matching strategy.** During training we need to determine which default boxes correspond to a ground truth detection and train the network accordingly. ==> We begin by matching each ground truth box to the default box with the **best jaccard overlap**, then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5).
> **Training objective.** The SSD training objective is derived from the MultiBox objective but is extended to handle multiple object categories. The overall objective loss function is a weighted sum of the localization loss and the confidence loss. **The localization loss** is a Smooth L1 loss between the predicted box ($l$) and groundtruth box ($g$) parameters. **The confidence loss** is the softmax loss over multiple classes confidences ($c$).
> **Choosing scales and aspect ratios for default boxes.** To handle different object scales, some methods suggest **1)** processing the image at different sizes and **2)** combing the results afterwards. *1. Previous works have shown that using feature maps from the lower layers can improve semantic segmentation quality because the lower layers capture more fine details of the input objects. 2. [12] showed that adding global context pooled from a feature map can help smooth the segmentation results.* Motivated by these methods, we use both the lower and upper feature maps for detection.
> Feature maps from different levels within a network are known to have different (empirical) receptive field sizes. Fortunately, within the SSD framework, the default boxes do not necessary need to correspond to the actual receptive fields of each layer. We design the tiling of default scales of the objects. The scale of the default boxes for each feature map is computed as:
$$
\begin{equation}
s_{k}=s_{min}+\frac{s_{max} - s_{min}}{m-1}(k-1), k \in[1,m]
\end{equation}
$$
> **Hard negative mining.** After the matching step, most of the default boxes are negatives. Instead of using all the negative examples, we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most $3:1$. We found that this leads to faster optimization and a more stable training.
> **Data augmentation.** To make the model more robust to various input object sizes and shapes, each training image is randomly sampled by one of the following options:
> * Use the entire original input image.
> * Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3, 0.5, 0.7, 0.9.
> * Randomly sample a patch. 
> The size of each sampled patch is $[0.1, 1]$ of the original image size, and the aspect ratio is between $1/2$ and $2$. We keep the overlapped part of the ground truth box if the center of it is in the sampled patch. After the aforementioned sampling step, each sampled patch is resized to fixed size and is horizontally flipped with probability of 0.5, in addition to applying some photo-metric distortions similar to those described in [14].

---
## 3. Experimental Results
### 3.1 PASCAL VOC2007
> We use $conv4\_3$, $conv7$ (fc7), $conv8\_2$, $conv9\_2$, $conv10\_2$, and $conv11\_2$ to predict both location and confidences. In this part, SSD300 and SSD512 outperform Faster-RCNN.
> **detection analysis tool from [21]**
> 
### 3.2 Model Analysis
> To understand SSD better, we carried out controlled experiments to examine how each component affects performance.
> **Data augmentation is crucial.** Fast and Faster R-CNN use the original image and the horizontal flip to train. We use a more extensive sampling strategy, similar to YOLO. **We do not know how much our sampling strategy will benefit Fast and Faster R-CNN, but they are likely to benefit less because they use a feature pooling step during classification that is relatively robust to object translation by design.**$???$
> **More default box shapes is better.** Using a variety of default box shapes seems to make the task of predicting boxes easier for the network.
> **Atrous is faster** $???$
> **Multiple output layers at different resolutions is better.** A major contribution of SSD is using default boxes of different scales on different output layers. To measure the advantage gained, we progressively remove layers and compare results. To measure the advantage gained, we progressively remove layers and compare results.
### 3.3 PASCAL VOC2012
### 3.4 COCO
### 3.5 Preliminary ILSVRC results
### 3.6 Data Augmentation for Small Object Accuracy
> Without a follow-up feature resampling step as in Faster R-CNN, the classification task for small objects is relatively hard for SSD. 
### 3.7 Inference time
> Considering the large number of boxes generated from our method, it is essential to perform non-maximum suppression($nms$) efficiently during inference. By using a confidence threshold of 0.01, we can filter out most boxes.

---
## 4. Related Work
> There are two established classes of methods for object detection in images, one based on **sliding windows** and the other based on **region proposal classification**. Before the advent of convolutional neural networks, the state-of-the-art for those two approaches - Deformable Part Model (DPM) and Selective Search - had comparable performance. However, after the dramatic improvement brought on by R-CNN, which combines selective search region proposals and convolutional network based post-classification, region proposal object detection methods became prevalent.
> The original R-CNN approach has been improved in a variety of ways.
> * The first set of approaches improve the quality and speed of post-classification, since it requires the classification of thousands of image crops, which is expensive and time-consuming. SPPnet speeds up the original R-CNN approach significantly.
> * The second set of approaches improve the quality of proposal generation using deep neural networks.
> * Another set of methods, which are directly related to our approach, skip the proposal step altogether and predict bounding boxes and confidences for multiple categories directly.

---
## 5. Conclusions

---
## Acknowledgment
