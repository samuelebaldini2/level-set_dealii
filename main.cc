#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <iostream>
#include <fstream>

#include "level_set.h"
#include "grid.h"


int main(int argc, char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::cout.precision(5);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  grid<2> grd(64);

  level_set<2,2> levelset(grd, pcout);
  levelset.init();
  levelset.reinit();
  grd.write_vtu(0.0,0);

  double dt = 0.01;
  
  for (int it=1; it<201; it++)
  { 
    pcout<<"Iteration: "<<it<<std::endl;
    levelset.adv_rk4(dt*it, dt);
    levelset.reinit();
    if (it % 1 == 0)
    {
      levelset.print(levelset.signed_distance,"solution");
      grd.write_vtu(dt*it, it);
    }   
  }
  
}

