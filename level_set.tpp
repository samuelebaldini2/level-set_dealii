template <int dim>
Tensor<1, dim> rayleigh_kothe_vortex(const Point<dim> &p, double t)
{
  Assert(dim >= 2, ExcNotImplemented());

  Tensor<1, dim> vel;
  if (dim == 2)
  {
    vel[0] = 2 * (sin(M_PI*p[0])*sin(M_PI*p[0])) * (sin(M_PI*p[1])*cos(M_PI*p[1])) * cos((M_PI * t)/2);
    vel[1] = -2 * (sin(M_PI*p[1])*sin(M_PI*p[1])) * (sin(M_PI*p[0])*cos(M_PI*p[0])) * cos((M_PI * t)/2);
  }
  else
  {
    vel[0] = 2 * (sin(M_PI*p[0])*sin(M_PI*p[0])) * sin(M_PI*p[1]) * sin(M_PI*p[2]) * cos((M_PI * t)/2);
    vel[1] = 2 * (sin(M_PI*p[1])*sin(M_PI*p[1])) * sin(M_PI*p[0]) * sin(M_PI*p[2]) * cos((M_PI * t)/2);
    vel[2] = 2 * (sin(M_PI*p[2])*sin(M_PI*p[2])) * sin(M_PI*p[0]) * sin(M_PI*p[1]) * cos((M_PI * t)/2);
  }

  return vel;
}

template <int dim>
Tensor<1, dim> rigid_rotation(const Point<dim> &p, double t)
{
  Assert(dim >= 2, ExcNotImplemented());

  Tensor<1, dim> vel;
  if (dim == 2)
  {
    vel[0] = 2*M_PI * (p[1] - 0.5) * cutoff(p[0]);
    vel[1] = - 2*M_PI * (p[0] - 0.5) * cutoff(p[0]);
  }
  else
  {
    abort();
  }

  return vel;
}

template <int dim, int spacedim, typename T>
std::tuple<std::vector<Point<spacedim>>, std::map<int,std::vector<std::vector<int>>> , std::vector<std::vector<int>> , std::vector<int> >
collect_interface_points(
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
                                update_values | update_quadrature_points);

  FEPointEvaluation<1, dim> fe_point_eval(mapping, dof_handler_support_points.get_fe(), update_values | update_gradients);

  std::vector<T>                       temp_distance(quad.size());
  std::vector<types::global_dof_index> local_dof_indices(dof_handler_support_points.get_fe().n_dofs_per_cell());

  std::vector<Point<dim>>              interface_points;
  std::map<int,std::vector<std::vector<int>>> interface_points_clustered;
  std::vector<std::vector<int>>        interface_segments;
  std::vector<int>                     interface_elements;

  const bool has_ghost_elements = signed_distance.has_ghost_elements();

  if (has_ghost_elements == false)
    signed_distance.update_ghost_values();

  int cnt_interface_node = 0;

  for (const auto &cell :
        tria.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    {
      const auto cell_distance =
        cell->as_dof_handler_iterator(dof_handler_signed_distance);
      distance_values.reinit(cell_distance);
      distance_values.get_function_values(signed_distance, temp_distance);
      cell_distance->get_dof_indices(local_dof_indices);

      std::vector<Point<dim>> segment;
      std::vector<int> tmp_numbering = {0,1,2,3};
      
      compute_intersection_points<dim>(temp_distance, tmp_numbering, distance_values, fe_point_eval, segment, mapping, cell_distance);
      if (segment.size() != 0)
        interface_elements.push_back(cell->active_cell_index());

      int n_segments = 9;
      if (segment.size() == 2 && false)
      {
        std::vector<int> cell_interface_points;
        auto increment = segment[1] - segment[0];
        increment /= n_segments;
        for (int i = 0; i < n_segments; i++)
        {
          interface_points.push_back(segment[0] + i*increment);
          cell_interface_points.push_back(cnt_interface_node);
          interface_segments.push_back({cnt_interface_node,cnt_interface_node+1});
          cnt_interface_node++;
        }
        interface_points.push_back(segment[1]);
        cell_interface_points.push_back(cnt_interface_node);
        cnt_interface_node++;

        interface_points_clustered[cell->active_cell_index()].push_back(cell_interface_points);
      }
      else //if (segment.size() == 4)
      {
        std::vector<int> cell_interface_points;

        std::vector<std::vector<int>> subcells(4);
        subcells[0] = {0, 6, 4, 8};
        subcells[1] = {6, 1, 8, 5};
        subcells[2] = {8, 5, 7, 3};
        subcells[3] = {4, 8, 2, 7};

        for (unsigned int isubcell = 0; isubcell < subcells.size(); isubcell++)
        {
          std::vector<Point<dim>> sub_segment;
          compute_intersection_points<dim>(temp_distance, subcells[isubcell], distance_values, fe_point_eval, sub_segment, mapping, cell_distance);

          if (sub_segment.size() == 2)
          {
            auto increment = sub_segment[1] - sub_segment[0];
            increment /= n_segments;
            for (int i = 0; i < n_segments; i++)
            {
              interface_points.push_back(sub_segment[0] + i*increment);
              cell_interface_points.push_back(cnt_interface_node);
              interface_segments.push_back({cnt_interface_node,cnt_interface_node+1});
              cnt_interface_node++;
            }
            interface_points.push_back(sub_segment[1]);
            cell_interface_points.push_back(cnt_interface_node);
            cnt_interface_node++;

            interface_points_clustered[cell->active_cell_index()].push_back(cell_interface_points);
          }
          else
          {
            AssertThrow(sub_segment.size() == 0, ExcMessage("MORE THAN TWO INTERSECTION IN SUBCELL"));
          }
        }
      }
    }


  if (has_ghost_elements == false)
    signed_distance.zero_out_ghost_values();

  return {interface_points, interface_points_clustered, interface_segments, interface_elements};
}

template <int dim>
void compute_intersection_points(std::vector<double> temp_distance, std::vector<int> node_number, FEValues<dim> & req_values, 
  FEPointEvaluation<1, dim> & fe_point_eval, std::vector<Point<dim>> & segment, const Mapping<dim, dim> & mapping, 
  const typename DoFHandler<dim,dim>::cell_iterator  & cell_distance)
{
  for (unsigned int i_edge = 0; i_edge < GeometryInfo<dim>::lines_per_cell; ++i_edge)
  {
    int q0 = GeometryInfo<dim>::line_to_cell_vertices(i_edge,0);
    int q1 = GeometryInfo<dim>::line_to_cell_vertices(i_edge,1);

    double phi0 = temp_distance[node_number[q0]];
    double phi1 = temp_distance[node_number[q1]];

    if (phi0 * phi1 < 0)
    {
      Point<dim> p0 = req_values.quadrature_point(node_number[q0]);
      Point<dim> p1 = req_values.quadrature_point(node_number[q1]);

      Tensor<1, dim> dir = p1 - p0;
      double length = dir.norm();
      dir /= length; 

      double s = phi1 / (phi1 - phi0); 
      Point<dim> x = p1 + s * (p0 - p1);

      p0 = p1;
      p1 = x;

      for (unsigned int iter = 0; iter < 10; ++iter)
      {
          std::vector<Point<dim>> points = {mapping.transform_real_to_unit_cell(cell_distance, x)};
          fe_point_eval.reinit(cell_distance, ArrayView<const Point<dim>>(points));
          fe_point_eval.evaluate(temp_distance, EvaluationFlags::values | EvaluationFlags::gradients);

          double phi = fe_point_eval.get_value(0);
          Tensor<1, dim, double> grad = fe_point_eval.get_gradient(0);

          double dphi_ds = grad * dir; 

          if (std::abs(dphi_ds) < 1e-12)
            break;

          s = phi / dphi_ds;

          x = x - s * dir;

          if (std::abs(phi) < 1e-10)
            break;
          else if (iter == 9)
          {
              std::cout << "Warning: Newton method did not converge"<< std::endl;
          }
      }
      segment.push_back(x); 
    }
  }
}

template <int dim, int fe_degree>
level_set<dim, fe_degree>::level_set(const std::string name, const grid<dim> &grd, const ConditionalOStream &pcout):
  grid_ptr(&grd),
  fe(fe_degree),
  dof_handler(grid_ptr->triangulation),
  pcout_ptr(&pcout),
  field_name(name)
{}

namespace SignedDistance
{
  template <int dim>
  class MaxOfTwoSpheres : public Function<dim>
  {
  public:
    MaxOfTwoSpheres(const Point<dim> &center1, double radius1,
                    const Point<dim> &center2, double radius2)
      : Function<dim>(1),
        c1(center1), r1(radius1),
        c2(center2), r2(radius2) {}

    virtual double value(const Point<dim> &p,
                          const unsigned int /*component*/ = 0) const override
    {
      double sdf1 = p.distance(c1) - r1;
      double sdf2 = p.distance(c2) - r2;
      return std::min(sdf1, sdf2); 
    }

  private:
    Point<dim> c1, c2;
    double r1, r2;
  };
}


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

  // VectorTools::interpolate(mapping,
  //                          dof_handler,
  //                          SignedDistance::MaxOfTwoSpheres<dim>(
  //                            (dim == 2) ? Point<dim>(0.35, 0.5) :
  //                                         Point<dim>(0.5, 0.5, 0.5),
  //                            0.25,
  //                            (dim == 2) ? Point<dim>(0.65, 0.5) :
  //                                         Point<dim>(0.8, 0.5, 0.5),
  //                            0.25),
  //                          signed_distance); 
  // VectorTools::interpolate(mapping,
  //                           dof_handler,
  //                           Functions::SignedDistance::ZalesakDisk<dim>(
  //                             (dim == 1) ? Point<dim>(0.5) :
  //                             (dim == 2) ? Point<dim>(0.5, 0.75) :
  //                                         Point<dim>(0.5, 0.5, 0.5),
  //                             0.15, 0.05, 0.25),
  //                           signed_distance);

  signed_distance.update_ghost_values();
}


template <int dim, int fe_degree>
void level_set<dim, fe_degree>::init_test()
{
  signed_distance_test.reinit(locally_owned_dofs,
                  locally_relevant_dofs,
                  MPI_COMM_WORLD);

  VectorTools::interpolate(mapping,
                            dof_handler,
                            Functions::SignedDistance::Sphere<dim>(
                              (dim == 1) ? Point<dim>(0.5) :
                              (dim == 2) ? Point<dim>(0.5, 0.75) :
                                          Point<dim>(0.5, 0.5, 0.5),
                              0.15),
                            signed_distance_test);
  // VectorTools::interpolate(mapping,
  //                          dof_handler,
  //                          SignedDistance::MaxOfTwoSpheres<dim>(
  //                            (dim == 2) ? Point<dim>(0.35, 0.5) :
  //                                         Point<dim>(0.5, 0.5, 0.5),
  //                            0.25,
  //                            (dim == 2) ? Point<dim>(0.65, 0.5) :
  //                                         Point<dim>(0.8, 0.5, 0.5),
  //                            0.25),
  //                          signed_distance_test);
  // VectorTools::interpolate(mapping,
  //                         dof_handler,
  //                         Functions::SignedDistance::ZalesakDisk<dim>(
  //                           (dim == 1) ? Point<dim>(0.5) :
  //                           (dim == 2) ? Point<dim>(0.5, 0.75) :
  //                                       Point<dim>(0.5, 0.5, 0.5),
  //                           0.15, 0.05, 0.25) ,
  //                         signed_distance_test);


  signed_distance_test.update_ghost_values();
}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::reinit_with_tangential_correction(double narrow_band_threshold)
{
  *pcout_ptr << "  - determine interface points" << std::endl;
  const auto [interface_points, interface_points_clustered, interface_segments_idx, interface_elements] =
    collect_interface_points(mapping,
                                    dof_handler,
                                    signed_distance,
                                    dof_handler);                            

  *pcout_ptr << "  - point projection on the interface" << std::endl;

  std::vector<Point<dim>> closest_points = interface_points;

  constexpr int    max_iter     = 30;
  constexpr double tol_distance = 1e-10;

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
            eval_values[i] * eval_gradient[i] / eval_gradient[i].norm_square();

          unmatched_points_idx_next.emplace_back(unmatched_points_idx[i]);
        }

      unmatched_points_idx.swap(unmatched_points_idx_next);

      n_unmatched_points =
        Utilities::MPI::sum(unmatched_points_idx.size(), MPI_COMM_WORLD);

      *pcout_ptr << " -> " << n_unmatched_points << std::endl;
    }


  if (n_unmatched_points > 0)
    *pcout_ptr << "WARNING: The tolerance of " << n_unmatched_points
          << " points is not yet attained." << std::endl;

  *pcout_ptr << "  - determine closest point" << std::endl;

  auto support_points_total = DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  const auto global_closest_point = gather_local_points(closest_points);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    global_interface_point = global_closest_point;

  // int quadric_idx = 0;
  // std::vector<int> point_to_quadric(closest_points.size());
  // std::map<int,quadric<dim>> interface_quadrics;
  // for (const auto cell_idx : interface_elements)
  // {
  //   const auto cell_interface_points = interface_points_clustered.at(cell_idx);
  //   for (const auto points_idx : cell_interface_points)
  //   {
  //     quadric<dim> quad(points_idx, closest_points);
  //     interface_quadrics[quadric_idx] = quad;
      
  //     for (const auto point_idx : points_idx)
  //       point_to_quadric[point_idx] = quadric_idx;

  //     quadric_idx ++;
  //   }
  // }

  std::map<int,Point<dim>> point_projection;
  
  DealII_PointCloud<dim> cloud;
  cloud.pts = global_closest_point;
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, DealII_PointCloud<dim>>,
      DealII_PointCloud<dim>,
      dim>;

  KDTree index(dim, cloud, {10 /* max leaf */});
  index.buildIndex();

  std::vector<Point<dim>> support_points;
  std::vector<int> support_points_idx;

  for (auto i : locally_owned_dofs)
  {
    int sign;
    (signed_distance[i]>0) ? sign = 1 : sign = -1;

    size_t ret_index;
    double out_dist_sqr;

    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &out_dist_sqr);
    index.findNeighbors(resultSet, &support_points_total[i][0], nanoflann::SearchParameters());

    point_projection[i] = global_closest_point[ret_index];

    if (support_points_total[i].distance(point_projection[i]) < narrow_band_threshold && !(ret_index % 10 == 0 || (ret_index+1) % 10 == 0))
    {
      support_points.push_back(support_points_total[i]);
      support_points_idx.push_back(i);
    }

  }

  auto point_projection_copy(point_projection);

  *pcout_ptr << "  - tangential correction step" << std::endl;

  std::vector<unsigned int> unmatched_points_idx2(support_points.size());
  std::iota(unmatched_points_idx2.begin(), unmatched_points_idx2.end(), 0);

  int n_unmatched_points2 =
    Utilities::MPI::sum(unmatched_points_idx2.size(), MPI_COMM_WORLD);

  Utilities::MPI::RemotePointEvaluation<dim, dim> rpe2;

  int max_iter2 = 30;
  for (int it = 0; it < max_iter2 && n_unmatched_points2 > 0; ++it)
    {
      *pcout_ptr << "    - iteration " << it << ": " << n_unmatched_points2;

      std::vector<Point<dim>> unmatched_points(unmatched_points_idx2.size());
      for (unsigned int i = 0; i < unmatched_points_idx2.size(); ++i)
        unmatched_points[i] = point_projection[support_points_idx[unmatched_points_idx2[i]]];

      rpe2.reinit(unmatched_points, grid_ptr->triangulation, mapping);

      AssertThrow(rpe2.all_points_found(),
                  ExcMessage("Processed point is outside domain."));

      const auto eval_values =
        VectorTools::point_values<1>(rpe2, dof_handler, signed_distance);

      const auto eval_gradient =
        VectorTools::point_gradients<1>(rpe2, dof_handler, signed_distance);

      std::vector<unsigned int> unmatched_points_idx_next;

      for (unsigned int i = 0; i < unmatched_points_idx2.size(); ++i)
      {
        const unsigned int idx = support_points_idx[unmatched_points_idx2[i]];
        Tensor<1,dim> v = support_points[unmatched_points_idx2[i]] - point_projection[idx];

        const Tensor<1,dim> &grad = eval_gradient[i];
        const double grad_norm2 = grad.norm_square();
        const double v_norm2    = v.norm_square();

        const double v_dot_grad = v * grad;
        Tensor<1,dim> v_tg = v - (v_dot_grad / grad_norm2) * grad;

        double rho = 1;
        double new_distance = (point_projection[idx] + rho*v_tg).distance(support_points[unmatched_points_idx2[i]]);
        double old_distance = point_projection[idx].distance(support_points[unmatched_points_idx2[i]]);
        while ( new_distance > old_distance && rho > 1.e-6)
        {
          rho /= 2;
          new_distance = (point_projection[idx] + rho*v_tg).distance(support_points[unmatched_points_idx2[i]]);
        }

        bool updated = false;
        Tensor<1,dim> delta;

        if (std::abs(eval_values[i]) > tol_distance)
        {
          delta -= eval_values[i] * grad / grad_norm2;
          updated = true;
        }
        if (v_tg.norm_square() > 1.e-3 * v_norm2 && rho > 2.e-6)
        {
          delta += rho*v_tg;
          updated = true;
        }

        if (updated)
        {
          point_projection[idx] += delta;
          unmatched_points_idx_next.emplace_back(unmatched_points_idx2[i]);
        }
      }

      unmatched_points_idx2.swap(unmatched_points_idx_next);
      n_unmatched_points2 = Utilities::MPI::sum(unmatched_points_idx2.size(), MPI_COMM_WORLD);
      *pcout_ptr << " -> " << n_unmatched_points2 << std::endl;
    }


  if (n_unmatched_points2 > 0)
  {
    *pcout_ptr << "WARNING: The tolerance of " << n_unmatched_points2
        << " points is not yet attained." << std::endl;
    not_converged_points.clear();
    for (unsigned int i = 0; i < unmatched_points_idx2.size(); i++)
    {
      not_converged_points.push_back(support_points[unmatched_points_idx2[i]]);
      point_projection[support_points_idx[unmatched_points_idx2[i]]] = point_projection_copy[support_points_idx[unmatched_points_idx2[i]]];
    }

    for (const auto point : not_converged_points)
      std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<": "<<point<<std::endl;
    std::cout<<std::endl;

    for (const auto i : unmatched_points_idx2)
      std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<": "<<point_projection[support_points_idx[i]]<<std::endl;
    std::cout<<std::endl;
  }

  for (auto i : locally_owned_dofs)
  {
    int sign;
    (signed_distance[i]>0) ? sign = 1 : sign = -1;

    signed_distance[i] = support_points_total[i].distance(point_projection[i])*sign;
  }

  
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

  signed_distance.update_ghost_values();

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
            auto vel_q = rigid_rotation(x_q, t);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                        fe_values.shape_value(j, q_point) *
                                        fe_values.JxW(q_point);

                    cell_matrix(i, j) -= dt * vel_q *
                                        fe_values.shape_grad(i, q_point) *
                                        fe_values.shape_value(j, q_point) *
                                        fe_values.JxW(q_point);

                    cell_rhs(i) += signed_distance[local_dof_indices[j]] *
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
  additional_data.max_basis_size = 50;
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
  double alpha_rk [4] = {1. , 1./2 , 1./2 , 1.};

  // =====
  // begin assembly
  // =====

  compute_system_matrix_rk();
  compute_system_rhs_rk(t, alpha_rk[0]*dt, 0);

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

  for (int rk_step = 0; rk_step < 4; rk_step ++)
  {
    *pcout_ptr << "Runge-Kutta step: " << rk_step  << std::endl;
    if (rk_step > 0)
    {
      system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
      compute_system_rhs_rk(t + alpha_rk[rk_step]*dt, alpha_rk[rk_step]*dt, rk_step);
    }

    rk_solution.reinit(locally_owned_dofs,
                        MPI_COMM_WORLD);

    solver.solve(system_matrix,
                  rk_solution,
                  system_rhs,
                  preconditioner);

    for (const auto i : rk_solution.locally_owned_elements())
      tmp_solution_rk[rk_step][i] = rk_solution[i];

    tmp_solution_rk[rk_step].update_ghost_values();

    *pcout_ptr << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;
  }

  constraints.distribute(rk_solution);

  for (const auto i : rk_solution.locally_owned_elements())
    for (unsigned int rk_step = 0; rk_step < 4; rk_step++)
      signed_distance[i] += dt / (6 * alpha_rk[rk_step]) * tmp_solution_rk[rk_step][i];

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
            auto vel_q = rayleigh_kothe_vortex(x_q, t);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_rhs(i) -= vel_q * ( signed_distance[local_dof_indices[j]] ) *
                                  fe_values.shape_grad(j, q_point) *
                                  fe_values.shape_value(i, q_point) *
                                  fe_values.JxW(q_point);

                    if (rk_step > 0)
                      cell_rhs(i) -= vel_q * ( alpha*tmp_solution_rk[rk_step-1][local_dof_indices[j]] ) *
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
std::vector<double> level_set<dim, fe_degree>::compute_error_l2()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const Quadrature<dim> quad(dof_handler.get_fe()
                                .base_element(0)
                                .get_unit_support_points());

  FEValues<dim> distance_values(mapping,
                                dof_handler.get_fe(),
                                quad,
                                update_values | update_quadrature_points);

  // FEValues<dim> req_values(mapping,
  //                           dof_handler.get_fe(),
  //                           quad,
  //                           update_quadrature_points);

  FEPointEvaluation<1, dim> fe_point_eval(mapping, dof_handler.get_fe(), update_values | update_gradients);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  signed_distance.update_ghost_values();
  signed_distance_test.update_ghost_values();
  
  double local_integral = 0.;
  double local_color_integral = 0.;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        
        double error    = 0.;
        double area = 0;
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                error += (signed_distance[local_dof_indices[j]] - signed_distance_test[local_dof_indices[j]]) *
                                (signed_distance[local_dof_indices[j]] - signed_distance_test[local_dof_indices[j]]) *
                                fe_values.shape_value(j, q_point) *
                                fe_values.JxW(q_point);

                area += fe_values.shape_value(j, q_point) *
                        fe_values.JxW(q_point);
              }

          }
        
        local_integral += error;

        double phase_area = 0;
        double phase_area_test = 0;

        const auto cell_req = cell->as_dof_handler_iterator(dof_handler);
        // req_values.reinit(cell_req);
        distance_values.reinit(cell_req);

        std::vector<double> temp_distance(dofs_per_cell);
        std::vector<double> temp_distance_test(dofs_per_cell);
        
        distance_values.get_function_values(signed_distance, temp_distance);
        distance_values.get_function_values(signed_distance_test, temp_distance_test);

        std::vector<Point<dim>> polygon_points;
        std::vector<Point<dim>> polygon_points_test;

        std::vector<int> tmp_numbering = {0,1,2,3};
        compute_intersection_points<dim>(temp_distance, tmp_numbering, distance_values, fe_point_eval, polygon_points, mapping, cell_req);
        compute_intersection_points<dim>(temp_distance_test, tmp_numbering, distance_values, fe_point_eval, polygon_points_test, mapping, cell_req);

        if (polygon_points.size()>0)
        {  
          for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
          {
            if (temp_distance[vertex_index] < 0)
              polygon_points.push_back(distance_values.quadrature_point(vertex_index));
          }               
          phase_area = shoelace<dim>(polygon_points);
        }
        else
        {
          double sum = 0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            sum += temp_distance[i];
          if (sum < 0)
            phase_area = area;
        }

        if (polygon_points_test.size()>0)
        {  
          for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
          {
            if (temp_distance_test[vertex_index] < 0)
              polygon_points_test.push_back(distance_values.quadrature_point(vertex_index));
          }      
          phase_area_test = shoelace<dim>(polygon_points_test);         
        }
       else
        {
          double sum = 0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            sum += temp_distance_test[i];
          if (sum < 0)
            phase_area_test = area;
        }

        
        local_color_integral += ( phase_area - phase_area_test ) * ( phase_area - phase_area_test );
        
      }
  
  return {local_integral , local_color_integral};
}

template <int dim>
double shoelace(std::vector<Point<dim>> pts)
{
    static_assert(dim == 2, "Shoelace è definita solo per dim=2");

    if (pts.size() < 3)
        return 0.0;

    double cx = 0.0, cy = 0.0;
    for (const auto &p : pts)
    {
        cx += p[0];
        cy += p[1];
    }
    cx /= pts.size();
    cy /= pts.size();

    std::sort(pts.begin(), pts.end(),
              [cx, cy](const Point<2> &a, const Point<2> &b) {
                  double ang_a = std::atan2(a[1] - cy, a[0] - cx);
                  double ang_b = std::atan2(b[1] - cy, b[0] - cx);
                  return ang_a < ang_b;
              });

    double sum = 0.0;
    for (unsigned int i = 0; i < pts.size(); ++i)
    {
        const auto &p1 = pts[i];
        const auto &p2 = pts[(i + 1) % pts.size()];
        sum += p1[0] * p2[1] - p2[0] * p1[1];
    }

    return std::abs(sum) * 0.5;
}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::print()
{
  grid_ptr->data_out.add_data_vector(dof_handler, signed_distance, field_name);
  grid_ptr->data_out.build_patches(mapping);
}

template <int dim,  int fe_degree> 
void level_set<dim, fe_degree>::print_marker(const int it)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::pair<double,std::string>> times_and_names;
    std::string fname = "../resu_"+std::to_string(grid_ptr->N_ele)+"/markers/marker_" + std::to_string(it) + ".csv";

    std::ofstream f(fname);
    f << "X,Y,value\n";
    for (unsigned int i=0; i<global_interface_point.size(); ++i)
        f << std::setprecision(8)
          << global_interface_point[i][0] << ","
          << global_interface_point[i][1] << ","
          << i << "\n";
  }
}

template <int dim, int fe_degree>
LinearAlgebra::distributed::Vector<double>* level_set<dim, fe_degree>::get_field_ptr()
{
  return &signed_distance;
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


