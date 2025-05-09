# RRT* ALgorithm with region Constraint

### Overview

  The project focuses on improving sampling bassed path planning algotithms for robotic navigation in strutured environments. It implements two Variants : RRT* and Region Constraint RRT*,
  which leverages environmental features like obstacle corners and edges to optimize the exploration. the goal is to demonstrate how incorporating spatial awareness( for e.g Corners) can reduce the computation time and produce shorter paths compared to traditional RRT*


### Components 

  Algorithms : 
  RRT* : A standard Random Exploring Random Tree algorithm for optimal path planning in obstacle rich environments
  RC-RRT* : A version of RRT* that uses radial sampling and vector analysis to identify obstacle corners in the occupancy grid and generates point near detected edges to guide the tree growth towards target.

  Dynamic obstacle handling :
  Simulates real time obstacle detection and dhynamic replanning during Robots navigation.

  
  Performance Comparison and Visualization:
  Executes multiple trial to compare computational time and path lenght between RRT* and RC-RRT* and plots the occupancy grids, RRT* Trees, path and detected corners and dynamic obstacle for analysis.

### Outcomes : 

  RC-RRT* typically achieves faster convergance and shorter paths in structured environments specifically corridors or juctions where it had shorter path lenght as well as palnning tim, but struggles with more complex maps with lots of corners or edges.
  Overall the performance is comparable to RRT* in complex environments in terms of planning time.

### Applicationns :
  Autnomous robots navigating indoor/ outdoor environments with walls, corners and obstacles.

### Referances : 
  R. Yang, Z. Hou, X. Zhuang and H. Chen, "Trajectory Planning via Region-Constrained Searching and Path-Based Collision Avoidance," 2023 IEEE International Conference on Unmanned Systems (ICUS), Hefei, China, 2023, pp. 439-444, doi: 10.1109/ICUS58632.2023.10318313. keywords: {Trajectory planning;Optimization methods;Collision avoidance;Vehicle dynamics;Trajectory optimization;Matlab;unmanned aerial vehicle;trajectory planning;real-time capability},


  
