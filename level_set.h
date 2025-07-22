#pragma once

// deal.II/lac
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/affine_constraints.h>

// deal.II/fe
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/mapping_q_cache.h>

// deal.II/dofs
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

// deal.II/numerics
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

// deal.II/base
#include <deal.II/base/tensor.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>

// deal.II/grid
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_vector.h>

// deal.II/matrix_free
#include <deal.II/matrix_free/fe_point_evaluation.h>

// deal.II/distributed
#include <deal.II/distributed/tria.h>

// std 
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>

// local
#include "grid.h"
#include "nanoflann.hpp"

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

template <int dim, int spacedim, typename T>
std::tuple<std::vector<Point<spacedim>>, std::vector<types::global_dof_index>>
collect_support_points_with_narrow_band(
  const Mapping<dim, spacedim>                &mapping,
  const DoFHandler<dim, spacedim>             &dof_handler_signed_distance,
  const LinearAlgebra::distributed::Vector<T> &signed_distance,
  const DoFHandler<dim, spacedim>             &dof_handler_support_points,
  const double                                 narrow_band_threshold);

template <int dim, int spacedim, typename T>
std::vector<Point<spacedim>>
collect_interface_points_linear(
  const Mapping<dim, spacedim>                &mapping,
  const DoFHandler<dim, spacedim>             &dof_handler_signed_distance,
  const LinearAlgebra::distributed::Vector<T> &signed_distance,
  const DoFHandler<dim, spacedim>             &dof_handler_support_points);

template <int dim>
std::vector<Point<dim>>
compute_initial_closest_points(
  const std::vector<Point<dim>>               &support_points,
  const std::vector<Point<dim>>               &interface_points);

template <int dim>
struct DealII_PointCloud
{
    std::vector<dealii::Point<dim>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t d) const {
        return pts[idx][d];  
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

template <typename T>
std::vector<T> gather_all_vectors(const std::vector<T> &local_data, MPI_Comm comm);

template <int dim>
std::vector<Point<dim>> gather_local_points(const std::vector<Point<dim>> &local_data);

template <int dim, int fe_degree>
class level_set
{
public:
  level_set(const grid<dim> &, const ConditionalOStream &);
  void init();
  void reinit();
  void adv(double);
  void adv_rk4(double, double);
  void print();

private:
  void compute_system_matrix_rk();
  void compute_system_rhs_rk(double, double, int);

  const grid<dim> *grid_ptr;
  const FE_Q<dim>       fe;
  const MappingQ1<dim>  mapping;
  DoFHandler<dim>       dof_handler;
  AffineConstraints<double> constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  LinearAlgebra::distributed::Vector<double> signed_distance;

  dealii::LinearAlgebraPETSc::MPI::SparseMatrix system_matrix;
  
  std::vector<LinearAlgebra::distributed::Vector<double>> tmp_solution_rk;
  
  dealii::LinearAlgebraPETSc::MPI::Vector rk_solution;
  dealii::LinearAlgebraPETSc::MPI::Vector system_rhs;

  const ConditionalOStream *pcout_ptr;
};

#include "level_set.tpp"