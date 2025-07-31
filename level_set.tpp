// Funzione che serializza un vettore di segmenti in un buffer di double
template <int dim>
std::vector<double> serialize_segments(const std::vector<Segmento<dim>> &segments) {
    std::vector<double> buffer(segments.size() * 2 * dim);
    for (size_t i = 0; i < segments.size(); ++i) {
        for (int d = 0; d < dim; ++d) {
            buffer[i * 2 * dim + d] = segments[i].p1[d];
            buffer[i * 2 * dim + dim + d] = segments[i].p2[d];
        }
    }
    return buffer;
}

// Funzione che deserializza il buffer in un vettore di segmenti
template <int dim>
std::vector<Segmento<dim>> deserialize_segments(const std::vector<double> &buffer) {
    const size_t n = buffer.size() / (2 * dim);
    std::vector<Segmento<dim>> segments(n);
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            segments[i].p1[d] = buffer[i * 2 * dim + d];
            segments[i].p2[d] = buffer[i * 2 * dim + dim + d];
        }
    }
    return segments;
}

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

double cutoff(double s)
{
  if (s < 0.075)
    return std::pow((s / 0.1), 2); // quadratica: va da 0 a 1
  else if (s > 0.925)
    return std::pow(((1.0 - s) / 0.1), 2);
  else
    return 1.0;
}

template <int dim>
Tensor<1, dim> rigid_rotation(const Point<dim> &p, double t)
{
  Assert(dim >= 2, ExcNotImplemented());

  Tensor<1, dim> vel;
  if (dim == 2)
  {
    vel[0] = 2*M_PI * (p[1] - 0.5) * cutoff(p[0])* cutoff(p[1]);
    vel[1] = - 2*M_PI * (p[0] - 0.5) * cutoff(p[0])* cutoff(p[1]);
  }
  else
  {
    abort();
  }

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
std::tuple<std::vector<Point<spacedim>> , std::vector<std::vector<int>> >
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
  std::vector<std::vector<int>> interface_segments;

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

      const auto cell_req =
        cell->as_dof_handler_iterator(dof_handler_support_points);
      req_values.reinit(cell_req);
      cell_req->get_dof_indices(local_dof_indices);

      std::vector<Point<dim>> segment;
      std::vector<Point<dim>> nodes_at_zero;
      for (unsigned int i_edge = 0; i_edge < GeometryInfo< dim >::lines_per_cell; i_edge ++)
      {
        int q = GeometryInfo<2>::line_to_cell_vertices(i_edge,0);
        int q2 = GeometryInfo<2>::line_to_cell_vertices(i_edge,1);

        if (temp_distance[q] * temp_distance[q2] < 0 )
        {
          double t = - temp_distance[q2] / (temp_distance[q2] - temp_distance[q]);
          Point<dim> point = req_values.quadrature_point(q2) + t * (req_values.quadrature_point(q2)-req_values.quadrature_point(q));
          segment.push_back(point);
        }
      }

      if (segment.size() == 2)
      {
        auto increment = segment[1] - segment[0];
        int n_segments = 100;
        increment /= n_segments;
        for (int i = 0; i < n_segments; i++)
        {
          interface_points.push_back(segment[0] + i*increment);
          interface_segments.push_back({cnt_interface_node,cnt_interface_node+1});
          cnt_interface_node++;
        }
        interface_points.push_back(segment[0] + n_segments*increment);
        cnt_interface_node++;
      }
      else if (segment.size() == 4)
      {
        for (int i = 0; i < 4; i++)
        {
          if (temp_distance[i] > 0)
          {
            std::vector<Point<dim>> segment2;
            for (unsigned int i_edge = 0; i_edge < GeometryInfo< dim >::lines_per_cell; i_edge ++)
            {
              int q = GeometryInfo<2>::line_to_cell_vertices(i_edge,0);
              int q2 = GeometryInfo<2>::line_to_cell_vertices(i_edge,1);

              if (q == i || q2 == i)
              {
                if (temp_distance[q] * temp_distance[q2] < 0 )
                {
                  double t = - temp_distance[q2] / (temp_distance[q2] - temp_distance[q]);
                  Point<dim> point = req_values.quadrature_point(q2) + t * (req_values.quadrature_point(q2)-req_values.quadrature_point(q));
                  segment2.push_back(point);
                }
              }
            }
            if (segment2.size() == 2)
            {
              auto increment = segment2[1] - segment2[0];
              int n_segments = 100;
              increment /= n_segments;
              for (int i = 0; i < n_segments; i++)
              {
                interface_points.push_back(segment2[0] + i*increment);
                interface_segments.push_back({cnt_interface_node,cnt_interface_node+1});
                cnt_interface_node++;
              }
              interface_points.push_back(segment2[0] + n_segments*increment);
              cnt_interface_node++;
            }
          }
        }
      }
      else if (segment.size() != 0)
      {
        std::cout<<"MORE THAN TWO INTERSECTION POINT"<<std::endl;
        abort();
      }
        
    }

  if (has_ghost_elements == false)
    signed_distance.zero_out_ghost_values();

  return {interface_points, interface_segments};
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
level_set<dim, fe_degree>::level_set(const std::string name, const grid<dim> &grd, const ConditionalOStream &pcout):
  grid_ptr(&grd),
  fe(fe_degree),
  dof_handler(grid_ptr->triangulation),
  pcout_ptr(&pcout),
  field_name(name)
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
void level_set<dim, fe_degree>::reinit_with_tangential_correction(double narrow_band_threshold, int it)
{
  *pcout_ptr << "  - determine interface points" << std::endl;
  const auto [interface_points, interface_segments_idx] =
    collect_interface_points_linear(mapping,
                                    dof_handler,
                                    signed_distance,
                                    dof_handler);                            

  *pcout_ptr << "  - point projection on the interface" << std::endl;

  std::vector<Point<dim>> closest_points = interface_points;

  constexpr int    max_iter     = 30;
  constexpr double tol_distance = 1e-8;

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

  auto global_interface_segments = gather_local_segments(interface_segments_idx, closest_points);

  *pcout_ptr << "  - determine closest point" << std::endl;

  auto support_points_total = DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  const auto global_closest_point = gather_local_points(closest_points);

  // std::ofstream out("points"+std::to_string(it)+".txt");
  // for (const auto &p : global_closest_point)
  // {
  //     for (unsigned int d = 0; d < dim; ++d)
  //         out << p[d] << (d < dim - 1 ? " " : "\n");
  // }
  // out.close();

  std::map<int,Point<dim>> point_projection;
  
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

    point_projection[i] = global_closest_point[ret_index];

  }

  auto point_projection_copy(point_projection);

  signed_distance.update_ghost_values();

  *pcout_ptr << "  - determine narrow band" << std::endl;
  const auto [support_points, support_points_idx] =
    collect_support_points_with_narrow_band(mapping,
                                            dof_handler,
                                            signed_distance,
                                            dof_handler,
                                            narrow_band_threshold);

  *pcout_ptr << "  - tangential correction step" << std::endl;

  std::vector<unsigned int> unmatched_points_idx2(support_points.size());
  std::iota(unmatched_points_idx2.begin(), unmatched_points_idx2.end(), 0);

  int n_unmatched_points2 =
    Utilities::MPI::sum(unmatched_points_idx2.size(), MPI_COMM_WORLD);

  Utilities::MPI::RemotePointEvaluation<dim, dim> rpe2;

  int max_iter2 = 10;
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
        auto v = support_points[unmatched_points_idx2[i]] - point_projection[support_points_idx[unmatched_points_idx2[i]]];
        double v_dot_grad = 0;
        for (int idim = 0; idim < dim; idim ++)
          v_dot_grad += v[idim] * eval_gradient[i][idim];

        double sin_theta = sqrt(v.norm_square()*eval_gradient[i].norm_square() - v_dot_grad*v_dot_grad) / (v.norm()*eval_gradient[i].norm());
        auto v_tg = v - (v_dot_grad / eval_gradient[i].norm_square()) * eval_gradient[i];

        if (std::abs(eval_values[i]) > tol_distance)
          {
            point_projection[support_points_idx[unmatched_points_idx2[i]]] -=
              eval_values[i] * eval_gradient[i];

            unmatched_points_idx_next.emplace_back(unmatched_points_idx2[i]);
          }
        else if (v_tg.norm() > 1.e-1*v.norm())
        {
          point_projection[support_points_idx[unmatched_points_idx2[i]]] +=  v_tg;
          unmatched_points_idx_next.emplace_back(unmatched_points_idx2[i]);
        }
      }

      unmatched_points_idx2.swap(unmatched_points_idx_next);

      n_unmatched_points2 =
        Utilities::MPI::sum(unmatched_points_idx2.size(), MPI_COMM_WORLD);

      *pcout_ptr << " -> " << n_unmatched_points2 << std::endl;
    }


  if (n_unmatched_points2 > 0)
  {
    *pcout_ptr << "WARNING: The tolerance of " << n_unmatched_points2
        << " points is not yet attained." << std::endl;
    for (unsigned int i = 0; i < unmatched_points_idx2.size(); i++)
      point_projection[support_points_idx[unmatched_points_idx2[i]]] = point_projection_copy[support_points_idx[unmatched_points_idx2[i]]];
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
void level_set<dim, fe_degree>::reinit_with_planes(int it)
{
  *pcout_ptr << "  - determine interface points" << std::endl;
  const auto [interface_points, interface_segments_idx] =
    collect_interface_points_linear(mapping,
                                    dof_handler,
                                    signed_distance,
                                    dof_handler);                            

  *pcout_ptr << "  - point projection on the interface" << std::endl;

  std::vector<Point<dim>> closest_points = interface_points;

  constexpr int    max_iter     = 30;
  constexpr double tol_distance = 1e-8;

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

  auto global_interface_segments = gather_local_segments(interface_segments_idx, closest_points);

  *pcout_ptr << "  - determine closest point" << std::endl;

  auto support_points_total = DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  const auto global_closest_point = gather_local_points(closest_points);

  // std::ofstream out("points"+std::to_string(it)+".txt");
  // for (const auto &p : global_closest_point)
  // {
  //     for (unsigned int d = 0; d < dim; ++d)
  //         out << p[d] << (d < dim - 1 ? " " : "\n");
  // }
  // out.close();

  std::map<int,Point<dim>> point_projection;

  DealII_SegmentCloud<dim> cloud;
  cloud.pts = global_interface_segments;
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, DealII_SegmentCloud<dim>>,
        DealII_SegmentCloud<dim>,
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

    double distance = global_interface_segments[ret_index].distance_to(support_points_total[i]);
    signed_distance[i] = distance*sign;

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
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  signed_distance.update_ghost_values();
  signed_distance_test.update_ghost_values();
  
  double local_integral = 0.;
  double local_color_error = 0.;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        double error    = 0.;
        double cell_error = 0;
        double cell_color = 0.;
        double test_cell_color = 0.;
        double area = 0;

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                int color = signed_distance[local_dof_indices[j]] / fabs(signed_distance[local_dof_indices[j]]) <= 0 ?
                            1 : 0;
                int test_color = signed_distance_test[local_dof_indices[j]] / fabs(signed_distance_test[local_dof_indices[j]]) <= 0 ?
                            1 : 0;
                error += (signed_distance[local_dof_indices[j]] - signed_distance_test[local_dof_indices[j]]) *
                                (signed_distance[local_dof_indices[j]] - signed_distance_test[local_dof_indices[j]]) *
                                fe_values.shape_value(j, q_point) *
                                fe_values.JxW(q_point);

                cell_error += (color - test_color) * (color - test_color)*
                                fe_values.shape_value(j, q_point) *
                                fe_values.JxW(q_point);
                
                cell_color += color *
                                fe_values.shape_value(j, q_point) *
                                fe_values.JxW(q_point);

                test_cell_color += test_color *
                                fe_values.shape_value(j, q_point) *
                                fe_values.JxW(q_point);

                area += fe_values.shape_value(j, q_point) *
                        fe_values.JxW(q_point);
              }

          }

        local_color_error += cell_error;
        
        local_integral += error;
      }
  
  return {local_integral , local_color_error};

}

template <int dim, int fe_degree>
void level_set<dim, fe_degree>::print()
{
  grid_ptr->data_out.add_data_vector(dof_handler, signed_distance, field_name);
  grid_ptr->data_out.build_patches(mapping);
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

template <int dim> 
std::vector<Segmento<dim>> gather_local_segments(std::vector<std::vector<int>> interface_segments_idx, std::vector<Point<dim>> closest_points)
{
  std::vector<Segmento<dim>> interface_segments;
  for (unsigned int i_segment = 0; i_segment < interface_segments_idx.size(); i_segment++)
  {
    Segmento<dim> i_segmento;
    i_segmento.p1 = closest_points[interface_segments_idx[i_segment][0]];
    i_segmento.p2 = closest_points[interface_segments_idx[i_segment][1]];
    interface_segments.push_back(i_segmento);
  }
  

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> local_buffer = serialize_segments<dim>(interface_segments);
  const int local_count = local_buffer.size();

  std::vector<int> recvcounts(size), displs(size);
  int total_count = 0;

  MPI_Gather(&local_count, 1, MPI_INT,
            recvcounts.data(), 1, MPI_INT,
            0, MPI_COMM_WORLD);

  if (rank == 0) {
      displs[0] = 0;
      for (int i = 1; i < size; ++i)
          displs[i] = displs[i - 1] + recvcounts[i - 1];
      total_count = displs[size - 1] + recvcounts[size - 1];
  }

  MPI_Bcast(&total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> global_buffer(total_count);

  MPI_Gatherv(local_buffer.data(), local_count, MPI_DOUBLE,
              global_buffer.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  MPI_Bcast(global_buffer.data(), total_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto gathered_segments = deserialize_segments<dim>(global_buffer);

  return gathered_segments;
}


