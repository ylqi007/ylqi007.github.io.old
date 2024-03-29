---
layout: post
title: Monte Carlo Tree Search - beginers guide
category: 机器学习
tags: "Machine-Learning"
---

# [Monte Carlo Tree Search - Beginners Guide](https://int8.io/monte-carlo-tree-search-beginners-guide/)

# 1. Introduction
## 1.1 Alpha Go/Zero
> Alpha Go/Zero system is a mix of several methods assembled into one great engineering piece of work. The core components of the Alpha Go/Zero are:
> * **Monte Carlo Tree Search** (certain variant with PUCT function for tree traversal)
> * **Residual Convolutional Neural Networks** – policy and value network(s) used for game evaluation and move prior probability estimation
> * **Reinforcement learning** used for training the network(s) via self-plays

## 1.2 Monte Carlo Tree Seach
> From a helicopter view Monte Carlo Tree Search has one main purpose: given a **game state** to choose **the most promising next move**.

![Game Definition](/public/img/posts/MachineLearning/MCTS/equation1.png)

**game tree**, **state**, **node**, **move**, **branching factor**

> To limit a game tree size, only visited states are **expanded**, unexpanded nodes are marked gray.
> Game tree is a recursive data structure, therefore after choosing the best move you end up in a child node which is in fact a root node for its subtree.

## 1.3 Minimax and Alpha-beta pruning
> The ultimate goal is to find the most promising next move assuming given game state and the game tree implied.
> The **biggest weakness of minimax** algorithm is the necessity to expand the whole game tree. For games with high branching factor (like Go or Chess) it leads to enormous game trees and so certain failure. 
> There is a few remedy for this. 
> * One way to go is **expand our game tree only up to certain threshold depth d .**
> * Another way to overcome game tree size problem is to prune the game tree via **alpha-beta pruning algorithm**. Alpha-beta pruning is boosted minimax. It traverses the game tree in minimax-fashion avoiding expansion of some tree branches.

---
# 2. Monte Carlo Tree Search - basic concepts
> Monte Carlo Tree Search simulates the games many times and tries to predict the most promising move based on the simulation results.
> Node is considered visited **if a playout has been started in that node** – meaning it has been evaluated at least once. If all children nodes of a node are visited node is considered **fully expanded**, otherwise – well – it is not fully expanded and further expansion is possible.
> Please be aware that **nodes chosen** by rollout policy function **during simulation** are not **considered visited**. They remain unvisited even though a rollout passes through them, only the node where simulation starts is marked visited.

## 2.1 Backpropagation - propagating back the simulation result
> Once simulation for a freshly visisted node (sometimes called a **leaf**) is finished, its result is **ready to be propagated back up** to the current game tree root node. The node where simulation started is marked **visited**.

## 2.2 Nodes' statistics
> The motivation for back-propagating simulation result is to update the **total simulation reward** $Q(v)$ and **total number of visit** $N(v)$ for all nodes v on backpropagation path (including the node where the simulation started).
> * $Q(v)$ – **Total simulation reward** is an attribute of a node $v$ and in a simplest form is a sum of simulation results that passed through considered node.
> * $N(v)$ – **Total number of visits** is another attribute of a node $v$ representing a counter of how many times a node has been on the backpropagation path (and so how many times it contributed to the total simulation reward)
> Nodes with high reward are good candidates to follow (**exploitation**) but those with low amount of visits may be interesting too (because they are not **explored** well).

## 2.3 Game Tree Traversal

![MCTS Tree](/public/img/posts/MachineLearning/MCTS/mcts_tree.png)

> Our current node – marked blue – is fully expanded so it must have been visited and so stores its node statistics: total simulation reward and total number of visits, same applies to its children. These values are compounds of our last piece: **Upper Confidence Bound applied to trees or shortly UCT**

## 2.4 Upper Confidence Bound applied to trees
> UCT is a function that lets us choose the next node among visited nodes to traverse through – the core function of Monte Carlo Tree Search

![UCT function](/public/img/posts/MachineLearning/MCTS/uct.png)

> **Node maximizing UCT is the one to follow** during Monte Carlo Tree Search tree traversal. Let’s see what UCT function does:
> First of all our function is defined for a child node vi of a node v. It is a sum of two components – the first component of our function $\frac{Q(vi)}{N(vi)}$, also called **exploitation component**, can be read as a winning/losing rate – we have total simulation reward divided by total number of visits which etimates win ratio in the node vi. This already looks promising – at the end of the day we might want to traverse through the nodes that have high winning rate.
> Why can’t we use exploitation component only? Because we would very quickly end up **greedily exploring only those nodes that bring a single winning playout** very early at the beginning of the search.

> One important remark on UCT function: in competitive games its **exploitation component** $Q_{i}$ is always **computed relative to player who moves at node** $i$ – it means that while traversing the game tree **perspective changes depending on a node being traversed through:** for any two consecutive nodes this perspective is opposite.

## 2.5 Terminating Monte Carlo Tree Search
> We now know almost all the pieces needed to successfully implement Monte Carlo Tree Search, there are few questions we need to answer though. **First of all when do we actually end the MCTS procedure?** The answer to this one is: it depends on the context. If you build a game engine then your “thinking time” is probably limited, plus your computational capacity has its boundaries, too. Therefore the safest bet is to run MCTS routine as long as your resources let you.
Once the MCTS routine is finished, the best move is usually the one with the highest number of visits N(vi) since it’s value has been estimated best (the value estimation itself must have been high as it’s been explored most often, too)

