#pragma once

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/tensor.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_vector.h>

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
  void print(dealii::LinearAlgebraPETSc::MPI::Vector, std::string);
  void print(LinearAlgebra::distributed::Vector<double>, std::string);
  LinearAlgebra::distributed::Vector<double> solution_distance;
  LinearAlgebra::distributed::Vector<double> signed_distance;

private:
  void compute_system_matrix_rk();
  void compute_system_rhs_rk(double, double, int);

  const grid<dim> *grid_ptr;
  const FE_Q<dim>       fe;
  const MappingQ1<dim>  mapping;
  DoFHandler<dim>       dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  dealii::LinearAlgebraPETSc::MPI::SparseMatrix system_matrix;
  
  std::vector<LinearAlgebra::distributed::Vector<double>> tmp_solution_rk;
  
  dealii::LinearAlgebraPETSc::MPI::Vector rk_solution;
  dealii::LinearAlgebraPETSc::MPI::Vector system_rhs;

  const ConditionalOStream *pcout_ptr;
};

template <int dim>
Tensor<1, dim> beta(const Point<dim> &p, double t)
{
  Assert(dim >= 2, ExcNotImplemented());

  Tensor<1, dim> vel;
  vel[0] = (2*sin(M_PI*p[0])*sin(M_PI*p[0]))*(sin(M_PI*p[1])*cos(M_PI*p[1]))*cos((M_PI * t)/2);
  vel[1] = (-2*sin(M_PI*p[1])*sin(M_PI*p[1]))*(sin(M_PI*p[0])*cos(M_PI*p[0]))*cos((M_PI * t)/2);

  return vel;
}

  template <int dim, int spacedim, typename T>
  std::tuple<std::vector<Point<spacedim>>, std::vector<types::global_dof_index>>
  collect_support_points_with_narrow_band(
    const Mapping<dim, spacedim>                &mapping,
    const DoFHandler<dim, spacedim>             &dof_handler_signed_distance,
    const LinearAlgebra::distributed::Vector<T> &signed_distance,
    const DoFHandler<dim, spacedim>             &dof_handler_support_points,
    const double                                 narrow_band_threshold)
  {
    AssertThrow(narrow_band_threshold >= 0,
                ExcMessage("The narrow band threshold"
                           " must be larger than or equal to 0."));
    const auto &tria = dof_handler_signed_distance.get_triangulation();
    const Quadrature<dim> quad(dof_handler_support_points.get_fe()
                                 .base_element(0)
                                 .get_unit_support_points());

    FEValues<dim> distance_values(mapping,
                                  dof_handler_signed_distance.get_fe(),
                                  quad,
                                  update_values);

    FEValues<dim> req_values(mapping,
                             dof_handler_support_points.get_fe(),
                             quad,
                             update_quadrature_points);

    std::vector<T>                       temp_distance(quad.size());
    std::vector<types::global_dof_index> local_dof_indices(
      dof_handler_support_points.get_fe().n_dofs_per_cell());

    std::vector<Point<dim>>              support_points;
    std::vector<types::global_dof_index> support_points_idx;

    const bool has_ghost_elements = signed_distance.has_ghost_elements();

    const auto &locally_owned_dofs_req =
      dof_handler_support_points.locally_owned_dofs();
    std::vector<bool> flags(locally_owned_dofs_req.n_elements(), false);

    if (has_ghost_elements == false)
      signed_distance.update_ghost_values();

    for (const auto &cell :
         tria.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
      {
        const auto cell_distance =
          cell->as_dof_handler_iterator(dof_handler_signed_distance);
        distance_values.reinit(cell_distance);
        distance_values.get_function_values(signed_distance, temp_distance);

        const auto cell_req =
          cell->as_dof_handler_iterator(dof_handler_support_points);
        req_values.reinit(cell_req);
        cell_req->get_dof_indices(local_dof_indices);

        for (const auto q : req_values.quadrature_point_indices())
          if (std::abs(temp_distance[q]) < narrow_band_threshold)
            {
              const auto idx = local_dof_indices[q];

              if (locally_owned_dofs_req.is_element(idx) == false ||
                  flags[locally_owned_dofs_req.index_within_set(idx)])
                continue;

              flags[locally_owned_dofs_req.index_within_set(idx)] = true;

              support_points_idx.emplace_back(idx);
              support_points.emplace_back(req_values.quadrature_point(q));
            }
      }

    if (has_ghost_elements == false)
      signed_distance.zero_out_ghost_values();

    return {support_points, support_points_idx};
  }

    template <int dim, int spacedim, typename T>
  std::vector<Point<spacedim>>
  collect_interface_points_linear(
    const Mapping<dim, spacedim>                &mapping,
    const DoFHandler<dim, spacedim>             &dof_handler_signed_distance,
    const LinearAlgebra::distributed::Vector<T> &signed_distance,
    const DoFHandler<dim, spacedim>             &dof_handler_support_points)
  {
    const auto &tria = dof_handler_signed_distance.get_triangulation();
    const Quadrature<dim> quad(dof_handler_support_points.get_fe()
                                 .base_element(0)
                                 .get_unit_support_points());

    FEValues<dim> distance_values(mapping,
                                  dof_handler_signed_distance.get_fe(),
                                  quad,
                                  update_values);

    FEValues<dim> req_values(mapping,
                             dof_handler_support_points.get_fe(),
                             quad,
                             update_quadrature_points);

    std::vector<T>                       temp_distance(quad.size());
    std::vector<types::global_dof_index> local_dof_indices(
      dof_handler_support_points.get_fe().n_dofs_per_cell());

    std::vector<Point<dim>>              interface_points;

    const bool has_ghost_elements = signed_distance.has_ghost_elements();

    if (has_ghost_elements == false)
      signed_distance.update_ghost_values();

    for (const auto &cell :
         tria.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
      {
        const auto cell_distance =
          cell->as_dof_handler_iterator(dof_handler_signed_distance);
        distance_values.reinit(cell_distance);
        distance_values.get_function_values(signed_distance, temp_distance);

        const auto cell_req =
          cell->as_dof_handler_iterator(dof_handler_support_points);
        req_values.reinit(cell_req);
        cell_req->get_dof_indices(local_dof_indices);

        for (const auto q : req_values.quadrature_point_indices())
          if (temp_distance[q] < 0)
            for (const auto q2 : req_values.quadrature_point_indices())
              if (temp_distance[q] * temp_distance[q2] < 0 && q2 != q)
              {
                double t = - temp_distance[q2] / (temp_distance[q2]-temp_distance[q]);
                Point<dim> point = req_values.quadrature_point(q2) + t * (req_values.quadrature_point(q2)-req_values.quadrature_point(q));
                interface_points.push_back(point);
              }
      }

    // std::sort(interface_points.begin(), interface_points.end());
    // auto point_compare = [](const auto& a, const auto& b) {
    //     for (int i = 0; i < dim; ++i) {
    //         if (a[i] < b[i] && fabs(a[i] - b[i]) > 1.e-10) return true;
    //         if (a[i] > b[i] && fabs(a[i] - b[i]) > 1.e-10) return false;
    //     }
    //     return false;
    // };
    // auto last = std::unique(interface_points.begin(), interface_points.end());
    // interface_points.erase(last, interface_points.end());

    if (has_ghost_elements == false)
      signed_distance.zero_out_ghost_values();

    return interface_points;
  }

  template <int dim>
  std::vector<Point<dim>>
  compute_initial_closest_points(
    const std::vector<Point<dim>>               &support_points,
    const std::vector<Point<dim>>               &interface_points)
  {
    std::vector<Point<dim>> closest_points;

    for (const auto support_point : support_points)
    {
      double temp_distance = 1.e10;
      Point<dim> temp_point;
      for (const auto interface_point : interface_points)
        if (fabs(support_point.distance(interface_point)) < temp_distance)
        {
          temp_distance = fabs(support_point.distance(interface_point));
          temp_point = interface_point;
        }
      closest_points.emplace_back(temp_point);
    }

    return closest_points;
  }

template <int dim, int fe_degree>
level_set<dim, fe_degree>::level_set(const grid<dim> &grd, const ConditionalOStream &pcout):
  grid_ptr(&grd),  
  fe(fe_degree),
  dof_handler(grid_ptr->triangulation),
  pcout_ptr(&pcout)  
{}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::init()
{
  *pcout_ptr << "Initialization of the level-set function" << std::endl;

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_active_dofs(dof_handler);

  signed_distance.reinit(locally_owned_dofs,
                  locally_relevant_dofs,
                  MPI_COMM_WORLD);
  VectorTools::interpolate(mapping,
                            dof_handler,
                            Functions::SignedDistance::Sphere<dim>(
                              (dim == 1) ? Point<dim>(0.5) :
                              (dim == 2) ? Point<dim>(0.5, 0.75) :
                                          Point<dim>(0.5, 0.5, 0.5),
                              0.15),
                            signed_distance);
  signed_distance.update_ghost_values();

  print(signed_distance,"solution");
}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::reinit()
{
  *pcout_ptr << "  - determine interface points" << std::endl;

  const auto interface_points =
    collect_interface_points_linear(mapping,
                                    dof_handler,
                                    signed_distance,
                                    dof_handler);
  
  const auto global_interface_points = gather_local_points(interface_points);

  *pcout_ptr << "  - determine narrow band" << std::endl;

  const auto [support_points, support_points_idx] =
    collect_support_points_with_narrow_band(mapping,
                                            dof_handler,
                                            signed_distance,
                                            dof_handler,
                                            0.1 /*narrow_band_threshold*/);

  auto closest_points =
    compute_initial_closest_points(support_points, global_interface_points);

  *pcout_ptr << "  - determine closest point iteratively" << std::endl;
  constexpr int    max_iter     = 30;
  constexpr double tol_distance = 1e-6;

  // std::vector<Point<dim>> closest_points = support_points; // initial guess

  std::vector<unsigned int> unmatched_points_idx(closest_points.size());
  std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);

  std::vector<bool> flags(unmatched_points_idx.size(), false);

  int n_unmatched_points =
    Utilities::MPI::sum(unmatched_points_idx.size(), MPI_COMM_WORLD);

  Utilities::MPI::RemotePointEvaluation<dim, dim> rpe;

  for (int it = 0; it < max_iter && n_unmatched_points > 0; ++it)
    {
      *pcout_ptr << "    - iteration " << it << ": " << n_unmatched_points;

      std::vector<Point<dim>> unmatched_points(unmatched_points_idx.size());
      for (unsigned int i = 0; i < unmatched_points_idx.size(); ++i)
        unmatched_points[i] = closest_points[unmatched_points_idx[i]];

      rpe.reinit(unmatched_points, grid_ptr->triangulation, mapping);

      AssertThrow(rpe.all_points_found(),
                  ExcMessage("Processed point is outside domain."));

      const auto eval_values =
        VectorTools::point_values<1>(rpe, dof_handler, signed_distance);

      const auto eval_gradient =
        VectorTools::point_gradients<1>(rpe, dof_handler, signed_distance);

      std::vector<unsigned int> unmatched_points_idx_next;

      for (unsigned int i = 0; i < unmatched_points_idx.size(); ++i)
        if (std::abs(eval_values[i]) > tol_distance)
        {
          closest_points[unmatched_points_idx[i]] -=
            eval_values[i] * eval_gradient[i];
          
          unmatched_points_idx_next.emplace_back(unmatched_points_idx[i]);
        }
        else if (!flags[unmatched_points_idx[i]])
        {
          flags[unmatched_points_idx[i]] = true;

          // u is the vector from the actual closest point to the support point
          // v is the gradient at the closest point
          // w is the vector normal to v that pass from the support point (we need the direction only)
          auto u = support_points[unmatched_points_idx[i]] - closest_points[unmatched_points_idx[i]];
          if (u.norm() < 0.02 || true)
          {  
            auto v = eval_gradient[i];
            auto w = v - u;

            double w_dot_v = 0;
            double u_dot_v = 0;
            for (int idim = 0; idim < dim; idim ++)
            {
              w_dot_v += w[idim]*v[idim];
              u_dot_v += u[idim]*v[idim];
            }
            w -= w_dot_v / eval_gradient[i].norm_square() * eval_gradient[i];
            w /= w.norm() + 1.e-10;

            double sin_theta = sqrt(std::max(0.0 , u.norm_square()*v.norm_square() - u_dot_v*u_dot_v)) / (u.norm()*v.norm());
            auto tangential_correction = - u.norm() * sin_theta * w;
            
            closest_points[unmatched_points_idx[i]] += tangential_correction;

            unmatched_points_idx_next.emplace_back(unmatched_points_idx[i]);
          }
        }

      unmatched_points_idx.swap(unmatched_points_idx_next);

      n_unmatched_points =
        Utilities::MPI::sum(unmatched_points_idx.size(), MPI_COMM_WORLD);

      *pcout_ptr << " -> " << n_unmatched_points << std::endl;
    }


  if (n_unmatched_points > 0)
    *pcout_ptr << "WARNING: The tolerance of " << n_unmatched_points
          << " points is not yet attained." << std::endl;

  *pcout_ptr << "  - determine distance in narrow band" << std::endl;

  auto support_points_total = DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  // std::vector<double> closest_point_vector;
  // for (const auto &point : closest_points)
  // {
  //   if (point[0]<1.e4)
  //     for (int idim=0; idim<dim; idim++)
  //       closest_point_vector.push_back(point[idim]);
  // }
  
  // std::vector<double> closest_point_total_vector = gather_all_vectors<double>(closest_point_vector, MPI_COMM_WORLD);

  // std::vector<Point<dim>> closest_point_total;
  // int cnt = 0;
  // for (unsigned int i=0; i<closest_point_total_vector.size()/dim; i++)
  // {
  //   Point<dim> point;
  //   for (unsigned int idim=0; idim<dim; idim++)
  //     point[idim] = closest_point_total_vector[cnt*dim + idim];
      
  //   closest_point_total.push_back(point);
  //   cnt++;
  // }

  const auto global_closest_point = gather_local_points(closest_points);

  DealII_PointCloud<dim> cloud;

  cloud.pts = global_closest_point;

  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, DealII_PointCloud<dim>>,
      DealII_PointCloud<dim>,
      dim>;

  KDTree index(dim, cloud, {10 /* max leaf */});
  index.buildIndex();
  
  for (auto i : locally_owned_dofs)
  {
    int sign;
    (signed_distance[i]>0) ? sign = 1 : sign = -1;
    
    size_t ret_index;
    double out_dist_sqr;

    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &out_dist_sqr);
    index.findNeighbors(resultSet, &support_points_total[i][0], nanoflann::SearchParameters());

    signed_distance[i] = sqrt(out_dist_sqr)*sign;
  }

  // for (unsigned int i = 0; i < closest_points.size(); ++i)
  // {
  //   double sign = signed_distance[support_points_idx[i]] / (fabs(signed_distance[support_points_idx[i]]) + 1.e-10);
  //   signed_distance[support_points_idx[i]] =
  //     support_points[i].distance(closest_points[i]) * sign;
  // }
  

  signed_distance.update_ghost_values();
}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::adv(double t)
{
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              locally_owned_dofs,
                                              MPI_COMM_WORLD,
                                              locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        MPI_COMM_WORLD);

  // =====
  // begin assembly
  // =====

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const double dt = 0.01;

  solution_distance.update_ghost_values();

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_matrix = 0.;
        cell_rhs    = 0.;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          { 
            const auto &x_q = fe_values.quadrature_point(q_point);
            auto beta_q = beta(x_q, t);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                        fe_values.shape_value(j, q_point) *
                                        fe_values.JxW(q_point);

                    cell_matrix(i, j) -= dt * beta_q *
                                        fe_values.shape_grad(i, q_point) *
                                        fe_values.shape_value(j, q_point) *
                                        fe_values.JxW(q_point);

                    cell_rhs(i) += solution_distance[local_dof_indices[j]] *
                                  fe_values.shape_value(j, q_point) *
                                  fe_values.shape_value(i, q_point) *
                                  fe_values.JxW(q_point);
                  }
              }
          }

        constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
      }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                      MPI_COMM_WORLD);

  SolverControl solver_control(dof_handler.n_dofs(),
                                1e-6 * system_rhs.l2_norm());

  LA::MPI::PreconditionAMG::AdditionalData data;
  #ifdef USE_PETSC_LA
    data.symmetric_operator = true;
  #else
    /* Trilinos defaults are good */
  #endif

  SolverGMRES<LA::MPI::Vector>::AdditionalData additional_data;
  additional_data.max_basis_size = 100;
  SolverGMRES<LA::MPI::Vector> solver(solver_control, additional_data);
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, data);

  solver.solve(system_matrix,
                completely_distributed_solution,
                system_rhs,
                preconditioner);

  *pcout_ptr << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  constraints.distribute(completely_distributed_solution);

  for (const auto i : completely_distributed_solution.locally_owned_elements())
    signed_distance[i] = completely_distributed_solution[i];
  
  signed_distance.update_ghost_values();

  // for (const auto i : completely_distributed_solution.ghost_elements())
  //   signed_distance[i] = completely_distributed_solution[i];

  // print(signed_distance, "solution1");
}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::adv_rk4(double t, double dt)
{
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              locally_owned_dofs,
                                              MPI_COMM_WORLD,
                                              locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        MPI_COMM_WORLD);

  for (unsigned int i = 0; i < 4; i++)
    {
      LinearAlgebra::distributed::Vector<double> rk_solution;
      rk_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            MPI_COMM_WORLD);

      tmp_solution_rk.push_back(rk_solution);
    }
  double alpha_rk [4] = {0. , dt/2 , dt/2 , dt};

  // =====
  // begin assembly
  // =====

  solution_distance.update_ghost_values();

  compute_system_matrix_rk(); 
  compute_system_rhs_rk(t, alpha_rk[0], 0);

  SolverControl solver_control(dof_handler.n_dofs(),
                                1e-6 * system_rhs.l2_norm());

  LA::MPI::PreconditionAMG::AdditionalData data;
  #ifdef USE_PETSC_LA
    data.symmetric_operator = true;
  #else
    /* Trilinos defaults are good */
  #endif

  SolverGMRES<LA::MPI::Vector>::AdditionalData additional_data;
  additional_data.max_basis_size = 100;
  SolverGMRES<LA::MPI::Vector> solver(solver_control, additional_data);
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, data);

  for (int i_rk_step = 0; i_rk_step < 4; i_rk_step ++)
  {
    *pcout_ptr << "Runge-Kutta step: " << i_rk_step  << std::endl;
    if (i_rk_step > 0)
    {
      system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
      compute_system_rhs_rk(t + alpha_rk[i_rk_step]*dt , alpha_rk[i_rk_step], i_rk_step);
    }
      
    rk_solution.reinit(locally_owned_dofs,
                        MPI_COMM_WORLD);

    solver.solve(system_matrix,
                  rk_solution,
                  system_rhs,
                  preconditioner);
   
    for (const auto i : rk_solution.locally_owned_elements())
      tmp_solution_rk[i_rk_step][i] = rk_solution[i];
  
    tmp_solution_rk[i_rk_step].update_ghost_values();                      

    *pcout_ptr << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;
  }

  constraints.distribute(rk_solution);

  double alpha_rk2 [4] = {1,2,2,1};

  for (const auto i : rk_solution.locally_owned_elements())
    for (unsigned int i_rk_step = 0; i_rk_step < 4; i_rk_step++)
      signed_distance[i] += dt/6 * alpha_rk2[i_rk_step] * tmp_solution_rk[i_rk_step][i];
  
  signed_distance.update_ghost_values();

}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::compute_system_matrix_rk()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_matrix = 0.;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          { 
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                        fe_values.shape_value(j, q_point) *
                                        fe_values.JxW(q_point);
                  }
              }
          }

        constraints.distribute_local_to_global(cell_matrix,
                                                local_dof_indices,
                                                system_matrix);
      }

  system_matrix.compress(VectorOperation::add);

}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::compute_system_rhs_rk(double t ,double alpha, int rk_step)
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_rhs    = 0.;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          { 
            const auto &x_q = fe_values.quadrature_point(q_point);
            auto beta_q = beta(x_q, t);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_rhs(i) -= beta_q * ( signed_distance[local_dof_indices[j]] ) *
                                  fe_values.shape_grad(j, q_point) *
                                  fe_values.shape_value(i, q_point) *
                                  fe_values.JxW(q_point);

                    if (rk_step > 0)
                      cell_rhs(i) -= beta_q * ( alpha*tmp_solution_rk[rk_step-1][local_dof_indices[j]] ) *
                                    fe_values.shape_grad(j, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);
                  }
              }
          }

        constraints.distribute_local_to_global(cell_rhs,
                                                local_dof_indices,
                                                system_rhs);
      }

  system_rhs.compress(VectorOperation::add);

}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::print(LA::MPI::Vector solution, std::string file_name)
{
  grid_ptr->data_out.add_data_vector(dof_handler, solution, file_name);
  grid_ptr->data_out.build_patches(mapping);
}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::print(LinearAlgebra::distributed::Vector<double> solution, std::string file_name)
{
  grid_ptr->data_out.add_data_vector(dof_handler, solution, file_name);
  grid_ptr->data_out.build_patches(mapping);
}

template <typename T>
std::vector<T> gather_all_vectors(const std::vector<T> &local_data, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int local_size = static_cast<int>(local_data.size());
  std::vector<int> recv_counts(size);
  MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

  // Calcolo dei displacements
  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i)
    displs[i] = displs[i - 1] + recv_counts[i - 1];

  int total_size = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
  std::vector<T> global_data(total_size);

  MPI_Datatype mpi_type;
  if constexpr (std::is_same<T, double>::value)
    mpi_type = MPI_DOUBLE;
  else if constexpr (std::is_same<T, int>::value)
    mpi_type = MPI_INT;
  else
    static_assert(sizeof(T) == 0, "Unsupported type for MPI_Allgatherv");

  MPI_Allgatherv(local_data.data(), local_size, mpi_type,
                 global_data.data(), recv_counts.data(), displs.data(), mpi_type,
                 comm);

  return global_data;
}

template <int dim>
std::vector<Point<dim>> gather_local_points(const std::vector<Point<dim>> &local_data)
{
  std::vector<double> point_vector;
  for (const auto &point : local_data)
  {
    if (point[0]<1.e4)
      for (int idim=0; idim<dim; idim++)
        point_vector.push_back(point[idim]);
  }
  
  std::vector<double> point_total_vector = gather_all_vectors<double>(point_vector, MPI_COMM_WORLD);

  std::vector<Point<dim>> point_total;
  int cnt0 = 0;
  for (unsigned int i=0; i<point_total_vector.size()/dim; i++)
  {
    Point<dim> point;
    for (unsigned int idim=0; idim<dim; idim++)
      point[idim] = point_total_vector[cnt0*dim + idim];
      
    point_total.push_back(point);
    cnt0++;
  }

  return point_total;
}