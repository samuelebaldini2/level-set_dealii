#pragma once

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>


using namespace dealii;

#ifdef DEAL_II_WITH_P4EST
  template <int dim, int spacedim = dim>
  using DistributedTriangulation = typename std::conditional_t<
    dim == 1,
    parallel::shared::Triangulation<dim, spacedim>,
    parallel::distributed::Triangulation<dim, spacedim>>;
#else
  template <int dim, int spacedim = dim>
  using DistributedTriangulation =
    parallel::shared::Triangulation<dim, spacedim>;
#endif

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
}

template <int dim>
class grid
{
public:
    grid(int );
    void write_vtu(const double, const int );
    
    DistributedTriangulation<dim> triangulation;
    mutable DataOut<dim> data_out;
private:
    std::vector< std::pair< double, std::string > > times_and_names;
};

template <int dim>
grid<dim>::grid(int N): triangulation(MPI_COMM_WORLD)
{
    GridGenerator::subdivided_hyper_cube(triangulation, N);
}

template <int dim>
void grid<dim>::write_vtu(const double time, const int it)
{
//   data_out.write_vtu_in_parallel("../resu/level_set.vtu", MPI_COMM_WORLD);
//   data_out.write_vtu_with_pvtu_record("../resu/","level_set", it, MPI_COMM_WORLD);
  data_out.write_vtu_in_parallel("../resu/level_set_"+std::to_string(it)+".vtu", MPI_COMM_WORLD);
  
  times_and_names.emplace_back(time,"level_set_"+std::to_string(it)+".vtu");  
  std::ofstream pvd_output ("../resu/solution.pvd");
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

  data_out.clear();
}