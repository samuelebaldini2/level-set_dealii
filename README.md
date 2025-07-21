# level-set_dealii
This repository contains a C++ implementation of a level set method based on the [deal.II](https://www.dealii.org) finite element library. The code extends the approach introduced in the second minicode of the [step-87 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_87.html).

The reinitialization of the level set function is performed via closest point projection onto the interface, followed by a tangential correction step to enhance accuracy near the zero level set. Finally, the signed distance function is computed using a direct algorithm that finds the closest computed point on the surface for each point of the grid. This search is performed using a KD-Tree, provided by the [nanoflann](https://github.com/jlblancoc/nanoflann.git) library.

## Features
- Closest point projection with tangential correction
- Direct signed distance computation using KD-Tree
- Based on deal.II [step-87 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_87.html)
  
## Dependencies
- [deal.II](https://www.dealii.org)
- [nanoflann](https://github.com/jlblancoc/nanoflann.git)




