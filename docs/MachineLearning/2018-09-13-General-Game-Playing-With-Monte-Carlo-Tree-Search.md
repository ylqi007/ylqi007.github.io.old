---
layout: post
title: General Game-Playing With Monte Carlo Tree Search
category: 机器学习
tags: "Machine-Learning"
---

# [General Game-Playing With Monte Carlo Tree Search](https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238)

> Monte Carlo tree search (MCTS) is a general game-playing algorithm to find the best move from any given game state of any game.

> There are significant benefits to using MCTS over vanilla minimax:
* 1. To determine which moves are good, depth-limited minimax needs a function that gives the estimated strength of any given game state. This *heuristic* function may be difficult to obtain.
* 2. MCTS efficiently deals with games with a high branching factor. As it gains information, MCTS increasingly favors moves that are more likely to be good, making its search asymmetric.
* 3. Minimax needs to run to completion to give the best move, which makes its runtime (and run-space) non-flexible. For games with large state spaces like chess and Go, this exhaustive search may even be intractable. MCTS does not need to run to completion; it outputs stronger plays the longer it runs, but its search can be stopped at any point. Having this flexible property, we say that MCTS is *anytime*.

> Being *aheuristic*, *asymmetric*, and *anytime* makes MCTS an attractive option for complex general game-playing.


## MCTS in Brief
> Minimax algorithms exhaustively search every possible ways to choose the best moves, but these methods are not suitable for large games, like chess and Go. MCTS deals with large trees effectively by *sampling* many paths down the game tree. This means it repeatedly goes down many, but not all, of the paths. As it tries more paths, it gains better estimates for which paths are good. Each one of these sample trials is called a *Monte Carlo simulation*, and each one of these simulations plays a sample game to the end to see which player wins. A lots of simulations have to be played before MCTS get enough information to output a good move.
To store the statistical information gained from these simulations, MCTS builds its own *search tree* from scratch, node by node, during the simulations.

![Game Definition](/public/img/posts/MachineLearning/MCTS/MCTS_Algorithm_diagram_from_Chaslot_2006.jpeg)

> The above figure details the actual procedure. In phase (1), existing information is used to repeatedly choose successive child nodes down to the end of the search tree. Next, in phase (2), the search tree is expanded by adding a node. Then, in phase (3), a simulation is run to the end to determine the winner. Finally, in phase (4), all the nodes in the selected path are updated with new information gained from the simulated game. This 4-phase algorithm is run repeatedly until enough information is gathered to produce a good move.

> It's important to note that the search tree is identical in structure to the game tree, and that the search tree additionally contains **statistical information** gathered from the simulations. We do not actually have the entire game tree, because it would be too massive. Instread, the search tree is built from scratch and its structure corresponds to a subset of the game tree's structure, likely a very small subset.


## MCTS in Detail
### 1. Selection
> The path selection should achieve two goals: We should **explore** new paths to gain information, and we should use existing information to **exploit** paths known to be good. In order to help us achieve these two goals, we need to select child nodes using a **selection function** that balances explorations and exploitation.
1. Random selection certainly does explore well, but it does not exploit at all. ==> This is a bad way.
2. The other way is to use the average win rate of each node. This achieves good exploitation, but it scores poorly on exploration. ==> Equally bad with random selection.
Some very smart people have figured out a good selection function that balances exploration with exploitation well, called [UCB1 (Upper Confidence Bound 1)](http://ggp.stanford.edu/readings/uct.pdf). When applied to MCTS, the combined algorithm is named UCT (Upper Confidence Bound 1 applied to trees). So, **MCTS+UCB1=UCT**.
The UCB1 selection function is:
$\frac{w_{i}}{s_{i}} + c \sqrt{\frac{ln s_{p}}{s_{i}}}$
* $w_{i}$: this node's number of simulations that resulted in a win;
* $s_{i}$: this node's total number of simulations;
* $s_{p}$: parent node's total number of simulation;
* $c$: exploration parameter.

> * The left term $w_{i}/s_{i}$ is the *exploitation term*. It is simply the average win rate, going larger the better a  node has historically performed.
> * The right term $\sqrt{ln s_{p}/s_{i}}$ is the *exploration term*. It goes larger the less frequently a node is selected for simulation.
> * The exploration parameter $c$ is just a number we can choose that allows us to control how much the equation favors exploration over exploitation; the usual number chosen is $c=\sqrt{2}$

> The numbers inside the nodes in the tree diagram are statistics for that node, corresponding to **number of wins** $w_{i}$ and **total number of simulations** $s_{i}$. Each time we need to select between multiple child nodes, we use the UCB1 selection function to get a UCB1 value for each child node, and we select the child node with the maximum value.
> **Note:** The problem of having an agent simultaneously balance exploration and exploiration between several choices when the payout of each choice is unknown is called the [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit).

### 2. Expansion
> After selection stops, there will be at least one unvisited move in the search tree. And we randomly choose one unexpanded move and create the child node corresponding to that move. We add this node as a child to the last selected node in the selection phase, expanding the search tree. The statistics information in the node is initialized with 0 wins out of 0 simulations ($w_{i}=0$, $s_{i}=0$).
> Some implementations choose to expand the tree by multiple nodes per simulation, but the most memory-efficient implementation is to create just one node per simulation.

### 3. Simulation
> Continuing from the newly-created node in the expansion phase, moves are selected randomly and the game state is repeatedly advanced. This repeats until the game is finished and a winner emerges.

### 4. Backpropagation
> After the simulation phase, the statistics on all the visited nodes are updated. Each visited node has its simulation count $s_{i}$ incremented. Depending on which player wins, its win count $w_{i}$ may also incremented.
> In the selection phase, MCTS uses the UCB1 selection function to make a decision on which child node to select. The UCB1 function, in turn, uses the numbers of wins $w_{i}$ and simulations $s_{i}$ of the children nodes, and the number of simulations of the parent node $s_{p}$, to generate the UCB1 values for each child node.
> 
> The beauty of MCTS(UCT) is that, due to its asymmetrical nature, the tree selection and growth gradually converges to better moves. At the end, you get the child node with the highest number of simulation $s_{i}$ and that's the best move according to MCTS.


