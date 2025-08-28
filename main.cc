#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include "level_set.h"
#include "grid.h"


int main(int argc, char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  std::cout.precision(15);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  
  std::ofstream fout("error_norm.txt");
  fout.precision(5);
  ConditionalOStream pfout(fout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  
  int N = 64;
  double dt_array[3] = {0.005, 0.0025, 0.00125};
  int max_it_array[3] = {400, 800, 1600};
  int reinit_each[3] = {2,4,8};
  double narrow_band_width = 0.5/N;
  for (int level = 0; level < 3; level++)
  {
    const int dim = 2;
    grid<dim> grd(N);

    const int n = 2;
    level_set<dim,n> levelset("signed_solution", grd, pcout);

    pcout<<std::endl;
    pcout<< "====================================================" <<std::endl;
    pcout<< "====================================================" <<std::endl;
    pcout << "Grid size = "<< N << "x"<<N<<" -- FE order = "<< n <<std::endl;

    levelset.init();
    levelset.reinit_with_tangential_correction(narrow_band_width);
    levelset.print();
    levelset.print_marker(0);
    grd.write_vtu(0.0,0);

    double dt = dt_array[level];
    int max_it = max_it_array[level];
        
    for (int it = 1; it <= max_it; it++)
    { 
      pcout<< "====================================================" <<std::endl;
      pcout<< "Iteration: "<< it << " -- Time: " << dt*(it-1) << " -> " << dt*it << std::endl;
      pcout<<"===================================================="<<std::endl;

      levelset.adv_rk4(dt*(it-1), dt);
      if (it % reinit_each[level] == 0)
        levelset.reinit_with_tangential_correction(narrow_band_width);
      if (it % (max_it / 20) == 0)
      {
        levelset.print();
        levelset.print_marker(it);
        grd.write_vtu(dt*it, it);
      }   
      
    }

    levelset.init_test();
    std::vector<double> local_error = levelset.compute_error_l2(); 
    std::vector<double> global_error(2, 0.0);

    MPI_Reduce(local_error.data(), global_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    pfout << "====================================================" << std::endl;
    pfout << "Grid size = "<< N << "x" << N <<" -- FE order = "<< n <<std::endl;
    pfout << "Error of the signed function in L2 norm = " << sqrt(global_error[0]) << std::endl;
    pfout << "Error of the color function in L2 norm = " << sqrt(global_error[1]) << std::endl;

    pcout << "====================================================" << std::endl;
    pcout << "Grid size = "<< N << "x" << N <<" -- FE order = "<< n <<std::endl;
    pcout << "Error of the signed function in L2 norm = " << sqrt(global_error[0]) << std::endl;
    pcout << "Error of the color function in L2 norm = " << sqrt(global_error[1]) << std::endl;

    N *= 2;
  }

}

