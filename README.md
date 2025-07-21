# level-set_dealii
This repository contains a C++ implementation of a level set method based on the [deal.II](https://www.dealii.org) finite element library. The code extends the approach introduced in the second minicode of the [step-87 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_87.html).

The reinitialization of the level set function is performed via closest point projection onto the interface, followed by a tangential correction step to enhance accuracy near the zero level set. Finally, the signed distance function is computed using a direct algorithm that, for each grid point, finds the closest point on the surface. This search is efficiently performed using a KD-Tree, provided by the [nanoflann](https://github.com/jlblancoc/nanoflann.git) library.




