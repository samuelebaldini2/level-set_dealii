# levelset_rcp
This repository contains a C++ implementation of a level set method based on the [deal.II](https://www.dealii.org) finite element library. The code extends the approach introduced in the second minicode of the [step-87 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_87.html). The main idea is to provide a Reinitialization by Closest Point (RCP) strategy for complex interface tracking within the Finite Element (FE) framework.

The zero level set is captured with markers projected onto the interface. Reinitialization is then performed directly by identifying, for each grid point, its closest marker via a KD-tree search, implemented with the [nanoflann](https://github.com/jlblancoc/nanoflann.git) library. For points lying within a narrow band around the interface, a tangential correction is applied.

## Features
- Closest point projection with tangential correction
- Direct signed distance computation using KD-Tree
- Based on deal.II [step-87 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_87.html)
  
## Dependencies
- [deal.II](https://www.dealii.org)
- [nanoflann](https://github.com/jlblancoc/nanoflann.git)




