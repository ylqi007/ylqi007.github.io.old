---
layout: post
title: Faster R-CNN 阅读笔记
category: 深度学习
tags: "Deep-Learning"
---

# Faster R-CNN
参考：[Faster R-CNN](http://www.cnblogs.com/venus024/p/5717766.html)
> Advances like SPPnet [7] and Fast R-CNN [5] have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a **Region Proposal Network (RPN)** that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. **An RPN is a fully-convolutional network that simultaneously predicts object bounds and objectness scores at each position.** RPNs are trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. With a simple alternating optimization, RPN and Fast R-CNN can be trained to share convolutional features.
* region proposal computation 是计算的瓶颈；
* 引入了Region Proposal Network (RPN)，thus enabling nearly cost-free region proposals.
* RPN 网络同时预测每个position的 bounds 和 scores。
* RPNs are trained end-to-end，可以产生high-quality region proposals，然后在Fast R-CNN网络中用于detection。
* 通过简单的optimization，RPN 和 Fast R-CNN 可以公用convolutional features.

---
## 1. Introduction
> Recent advances in object detection are driven by the success of **region proposal methods** (e.g., [22]) and **region-based convolutional neural networks (R-CNNs)** [6]. Although region-based CNNs were computationally expensive as originally developed in [6], their cost has been drastically reduced thanks to sharing convolutions across proposals [7, 5].
> In this paper, we show that an algorithmic change—**computing proposals with a deep net**—leads to an elegant and effective solution, where proposal computation is nearly cost-free given the detection network’s computation. To this end, we introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks [7, 5]. By sharing convolutions at test-time, the marginal cost for computing proposals is small (e.g., 10ms per image).
> Our observation is that the convolutional (conv) feature maps used by region-based detectors, like Fast R-CNN, can also be used for generating region proposals. On top of these conv features, we construct RPNs by adding two additional conv layers: one that encodes each conv map position into a short (e.g., 256-d) feature vector and a second that, at each conv map position, outputs an objectness score and regressed bounds for k region proposals relative to various scales and aspect ratios at that location (k = 9 is a typical value).
* convolutional ($conv$) feature maps 可以在Fast R-CNN中被 region-based detectors 使用，**也可以用于generate region proposals**. 
* 在这些 conv feature maps 上，建立RPN网络。RPN包含两部分：$(1)$ one that encodes each conv map position into a short feature vector; $(2)$ the second outputs an $objectness score$ and $regressed bounds$ for $k$ region proposals relative to various scales and aspect ratios at that location ($k=9$ is typical value)

> Our RPNs are thus a kind of fully-convolutional network (FCN) [14] and they can be trained end-to-end specifically for the task for generating detection proposals. **To unify RPNs with Fast R-CNN [5] object detection networks, we propose a simple training scheme that alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed.** This scheme converges quickly and produces a unified network with conv features that are shared between both tasks.
* To unify RPNs with Fast R_CNN，文中提出了 a simple training scheme that alternates between **fine-tuning for region proposal task** and then **fine-tuning for object detection**, while keeping the proposals fixed.

---
## 2. Related Work
> Several recent papers have proposed ways of using deep networks for locating class-specific or classagnostic bounding boxes [21, 18, 3, 20]. In the **OverFeat method** [18], a fully-connected (fc) layer is trained to predict the box coordinates for the localization task that assumes a single object. The fc layer is then turned into a conv layer for detecting multiple class-specific objects. The **Multi-Box methods** [3, 20] generate region proposals from a network whose last fc layer simultaneously predicts multiple (e.g., 800) boxes, which are used for R-CNN [6] object detection.
> 
> **Shared computation of convolutions** [18, 7, 2, 5] has been attracting increasing attention for efficient, yet accurate, visual recognition.

---
## 3. Regiion Proposal Networks
> A **Region Proposal Network (RPN)** takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score. We model this process with **a fully convolutional network**.
> 
> Because our ultimate goal is to **share computation with a Fast R-CNN object detection network** [5], we assume that both nets share a common set of conv layers.
* 作者的实验中investigate ZF model 和 VGG model。

> To generate region proposals, we slide a small network over the conv feature map output by the last shared conv layer. This network is **fully connected** to an $n \times n$ spatial window of the input conv feature map. Each sliding window is mapped to **a lower-dimensional vector** (256-d for ZF and 512-d for VGG). This vector is fed into two sibling fully-connected layers—**a box-regression layer (reg)** and **a box-classification layer (cls)**.

> **Translation-Invariant Anchors**
> At each sliding-window location, we simultaneously predict $k$ region proposals, so the reg layer has $4k$ outputs encoding the coordinates of k boxes. The $cls$ layer outputs $2k$ scores that estimate probability of object/not-object for each proposal. The $k$ proposals are parameterized relative to $k$ reference boxes, called **anchors**.
> An important property of our approach is that is that it is $translation\ invariant$, both in terms of the anchors and the functions that compute proposals relative to the anchors.
* 在每一个sliding-window的位置上，同时predict $k$ 个region proposal，则$reg layer$会有 $4k$ 个输出，标记 $k$ 个位置；$cls\ layer$有 $2k$ 个scores输出，分别estimate 每一个proposal 的 probability of object/non-object。
* $anchors$ 的概念

> **A Loss Function for Learning Region Proposals**
> For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors:
> 1. the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box;
> 2. an anchor that has an IoU overlap higher than 0.7 with any ground-truth box.

> Note that a single ground-truth box may assign positive labels to multiple anchors. We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. Anchors that are neither positive nor negative do not contribute to the training objective.
> With these definitions, we minimize an objective function following the multi-task loss in Fast RCNN [5]. Our loss function for an image is defined as:
$$
\begin {equation} \label {}
L({p_{i}}, {t_{i}}) = \frac{1}{N_{cls}} \sum_{i}L_{cls}(p_{i}, p^{*}_{i})+\lambda \frac{1}{N_{reg}} \sum_{i}p^{*}_{i} L_{cls}(t_{i}, t^{*}_{i})
\end {equation}
$$
> Here, $i$ is the index of an anchor in a mini-batch and $p_{i}$ is the predicted probability of anchor $i$ being an object. The ground-truth label $p_{i}$ is $1$ if the anchor is positive, and is $0$ if the anchor is negative. ti is a vector representing the 4 parameterized coordinates of the predicted bounding box, and $t_{i}$ is that of the ground-truth box associated with a positive anchor.

> **Optimization**
> The RPN, which is naturally implemented as a fully-convolutional network [14], can be trained end-to-end by back-propagation and stochastic gradient descent (SGD)[12].
> 
> **Sharing Convolutional Features for Region Proposal and Object Detection**
> Both RPN and Fast R-CNN, trained independently, will modify their conv layers in different ways. We therefore need to develop a technique that allows for sharing conv layers between the two networks, rather than learning two separate networks. Note that this is not as easy as simply defining a single network that includes both RPN and Fast R-CNN, and then optimizing it jointly with backpropagation. The reason is that Fast R-CNN training depends on fixed object proposals and it is not clear a priori if learning Fast R-CNN while simultaneously changing the proposal mechanism will converge. While this joint optimizing is an interesting question for future work, we develop a pragmatic 4-step training algorithm to learn shared features via alternating optimization.
> > In the first step, we train the RPN as described above. This network is initialized with an ImageNet pre- trained model and fine-tuned end-to-end for the region proposal task.
* 训练RPN网络，用一个pre-trained的model初始化。

> > In the second step, we train a separate detection network by Fast R-CNN using the proposals generated by the step-1 RPN. This detection network is also initialized by the ImageNet-pre-trained model. At this point the two networks do not share conv layers.
* 用第一步中生成的proposals，通过Fast R-CNN训练一个单独的detection network。

> > In the third step, we use the detector network to initialize RPN training, but we fix the shared conv layers and only fine-tune the layers unique to RPN. Now the two networks share conv layers.
* 用第二步中的detector network初始化RPN训练，但是固定shared conv layers，only fine-tune the layers unique to RPN。此时两个网络公用conv layers. ***?????*** $??不解??$

> > Finally, keeping the shared $conv$ layers fixed, we fine-tune the $fc$ layers of the Fast R-CNN. As such, both networks share the same $conv$ layers and form a unified network.
* 保持conv layers固定，微调Fast R-CNN中的fc layer。如此，两个网络可以共享一个conv layers，并建立一个统一的network。
>
* RPN 和 Fast-RCNN 分别独立训练，并以不同的方式修改conv layers的参数。因此需要一种技术允许两个networks共享conv layers的参数，而不是从两个单独的网络中分别训练。但是要注意到，想要简单定义一个网络来同时包含RPN和Fast R-CNN，并通过BP算法同时优化的做法并不容易实现。这个因为Fast R-CNN训练依靠固定的object proposals。如果学习Fast R-CNN的同时改变proposal mechanism，这种做法并没有clear a priori. 然而这种同时优化的做法是个有趣的问题，值得未来进一步的工作。
> 
> **Implementation Details**


---
## 4. Experiments
**Ablation Experiments**

**Detection Accuracy and Running Time of VGG-16**

**Analysis of Recall-to-IoU**

**One-Stage Detection $vs.$ Two-Stage Proposal + Detection**

---
## 5. Conclusion
> We have presented Region Proposal Networks (RPNs) for efficient and accurate region proposal generation. By sharing convolutional features with the down-stream detection network, the region proposal step is nearly cost-free. Our method enables a unified, deep-learning-based object detection system to run at 5-17 fps. The learned RPN also improves region proposal quality and thus the overall object detection accuracy.

