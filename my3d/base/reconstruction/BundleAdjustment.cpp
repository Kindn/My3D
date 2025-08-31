/*
 * filename: BundleAdjustment.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "base/reconstruction/BundleAdjustment.h"

namespace gopt {

VertexCamera::VertexCamera() {}

VertexCamera::VertexCamera(
    const typename my3d::base::BundleAdjustment::Camera &camera) {
  this->setEstimate(camera);
}

VertexCamera::VertexCamera(
    const size_t &id,
    const typename my3d::base::BundleAdjustment::Camera &camera,
    const bool &fixed, const bool &fix_intrinsics, const bool &fix_translation,
    const bool &fix_rotation)
    : fix_intrinsics_{false}, fix_translation_{fix_translation},
      fix_rotation_{fix_rotation} {
  this->setId(id);
  this->setEstimate(camera);
  this->setFixed(fixed);
}

void VertexCamera::setToOrigin() {
  estimate_.rotation.setIdentity();
  estimate_.translation.setZero();
  estimate_.camera.fx_ = 0.0;
  estimate_.camera.fy_ = 0.0;
  // estimate_.camera.cx_ = 0.0;
  // estimate_.camera.cy_ = 0.0;
  // estimate_.camera.alpha_ = 0.0;
  estimate_.camera.k1_ = 0.0;
  estimate_.camera.k2_ = 0.0;
}

void VertexCamera::plus(const Eigen::VectorXd &update) {
  if (!fix_rotation_) {
    estimate_.rotation = (estimate_.rotation *
                          Eigen::Quaterniond(1.0, update(0) / 2.0,
                                             update(1) / 2.0, update(2) / 2.0))
                             .normalized();
  }
  if (!fix_translation_) {
    estimate_.translation += Eigen::Vector3d(update(3), update(4), update(5));
  }
  if (!fix_intrinsics_) {
    estimate_.camera.fx_ += update(6);
    estimate_.camera.fy_ += update(7);
    // estimate_.camera.cx_ += update(8);
    // estimate_.camera.cy_ += update(9);
    // estimate_.camera.alpha_ += update(8);
    estimate_.camera.k1_ += update(8);
    estimate_.camera.k2_ += update(9);
  }
}

VertexCameraPose::VertexCameraPose() {}

VertexCameraPose::VertexCameraPose(
    const typename my3d::base::BundleAdjustment::Camera &camera) {
  this->setEstimate(camera);
}

VertexCameraPose::VertexCameraPose(
    const size_t &id,
    const typename my3d::base::BundleAdjustment::Camera &camera,
    const bool &fixed) {
  this->setId(id);
  this->setEstimate(camera);
  this->setFixed(fixed);
}

void VertexCameraPose::setToOrigin() {
  estimate_.rotation.setIdentity();
  estimate_.translation.setZero();
}

void VertexCameraPose::plus(const Eigen::VectorXd &update) {
  estimate_.rotation = (estimate_.rotation *
                        Eigen::Quaterniond(1.0, update(0) / 2.0,
                                           update(1) / 2.0, update(2) / 2.0))
                           .normalized();
  estimate_.translation += Eigen::Vector3d(update(3), update(4), update(5));
}

VertexCameraTranslation::VertexCameraTranslation() {}

VertexCameraTranslation::VertexCameraTranslation(
    const typename my3d::base::BundleAdjustment::Camera &camera) {
  this->setEstimate(camera);
}

VertexCameraTranslation::VertexCameraTranslation(
    const size_t &id,
    const typename my3d::base::BundleAdjustment::Camera &camera,
    const bool &fixed) {
  this->setId(id);
  this->setEstimate(camera);
  this->setFixed(fixed);
}

void VertexCameraTranslation::setToOrigin() { estimate_.translation.setZero(); }

void VertexCameraTranslation::plus(const Eigen::VectorXd &update) {
  estimate_.translation += Eigen::Vector3d(update(0), update(1), update(2));
}

VertexCameraRotation::VertexCameraRotation() {}

VertexCameraRotation::VertexCameraRotation(
    const typename my3d::base::BundleAdjustment::Camera &camera) {
  this->setEstimate(camera);
}

VertexCameraRotation::VertexCameraRotation(
    const size_t &id,
    const typename my3d::base::BundleAdjustment::Camera &camera,
    const bool &fixed) {
  this->setId(id);
  this->setEstimate(camera);
  this->setFixed(fixed);
}

void VertexCameraRotation::setToOrigin() { estimate_.rotation.setIdentity(); }

void VertexCameraRotation::plus(const Eigen::VectorXd &update) {
  estimate_.rotation = (estimate_.rotation *
                        Eigen::Quaterniond(1.0, update(0) / 2.0,
                                           update(1) / 2.0, update(2) / 2.0))
                           .normalized();
}

VertexCameraIntrinsics::VertexCameraIntrinsics() {}

VertexCameraIntrinsics::VertexCameraIntrinsics(
    const typename my3d::base::BundleAdjustment::Camera &camera) {
  this->setEstimate(camera);
}

VertexCameraIntrinsics::VertexCameraIntrinsics(
    const size_t &id,
    const typename my3d::base::BundleAdjustment::Camera &camera,
    const bool &fixed) {
  this->setId(id);
  this->setEstimate(camera);
  this->setFixed(fixed);
}

void VertexCameraIntrinsics::setToOrigin() {
  estimate_.camera.fx_ = 0.0;
  estimate_.camera.fy_ = 0.0;
  estimate_.camera.cx_ = 0.0;
  estimate_.camera.cy_ = 0.0;
  // estimate_->camera.alpha_ = 0.0;
  estimate_.camera.k1_ = 0.0;
  estimate_.camera.k2_ = 0.0;
}

void VertexCameraIntrinsics::plus(const Eigen::VectorXd &update) {
  estimate_.camera.fx_ += update(0);
  estimate_.camera.fy_ += update(1);
  estimate_.camera.cx_ += update(2);
  estimate_.camera.cy_ += update(3);
  // estimate_->camera.alpha_ += update(0);
  estimate_.camera.k1_ += update(4);
  estimate_.camera.k2_ += update(5);
}

VertexPoint3D::VertexPoint3D() {}

VertexPoint3D::VertexPoint3D(const Eigen::Vector3d &point_3D) {
  estimate_ = point_3D;
}

void VertexPoint3D::setToOrigin() { estimate_.setZero(); }

void VertexPoint3D::plus(const Eigen::VectorXd &update) {
  estimate_.x() += update(0);
  estimate_.y() += update(1);
  estimate_.z() += update(2);
}

EdgeProjectionFull::EdgeProjectionFull() {}

void EdgeProjectionFull::computeResidual() {
  auto v0 = std::dynamic_pointer_cast<VertexCamera>(vertices_[0]);
  auto v1 = std::dynamic_pointer_cast<VertexPoint3D>(vertices_[1]);
  const typename my3d::base::BundleAdjustment::Camera camera =
      v0->getEstimate();
  const Eigen::Vector3d &point_3D = v1->getEstimate();
  const Eigen::Vector3d pc =
      camera.rotation.inverse() * (point_3D - camera.translation);
  const Eigen::Vector2d projection = camera.camera.project(pc);
  residual_ = projection - measurement_;
  // const Eigen::Matrix3x4d pose =
  //     my3d::base::composePose(camera.rotation.inverse().toRotationMatrix(),
  //     -(camera.rotation.inverse() * camera.translation));
  // residual_ = my3d::base::computeReprojectionResidual(point_3D, measurement_,
  // pose, &camera.camera);
}

EdgeProjectionOnlyPose::EdgeProjectionOnlyPose() {}

void EdgeProjectionOnlyPose::computeResidual() {
  auto v0 = std::dynamic_pointer_cast<VertexCameraPose>(vertices_[0]);
  auto v1 = std::dynamic_pointer_cast<VertexPoint3D>(vertices_[1]);
  const typename my3d::base::BundleAdjustment::Camera camera =
      v0->getEstimate();
  const Eigen::Vector3d &point_3D = v1->getEstimate();
  const Eigen::Vector3d pc =
      camera.rotation.inverse() * (point_3D - camera.translation);
  const Eigen::Vector2d projection = camera.camera.project(pc);
  residual_ = projection - measurement_;
}

EdgeProjectionSeparated::EdgeProjectionSeparated() {}

void EdgeProjectionSeparated::computeResidual() {
  auto v_in = std::dynamic_pointer_cast<VertexCameraIntrinsics>(vertices_[0]);
  auto v_t = std::dynamic_pointer_cast<VertexCameraTranslation>(vertices_[1]);
  auto v_r = std::dynamic_pointer_cast<VertexCameraRotation>(vertices_[2]);
  auto v_p = std::dynamic_pointer_cast<VertexPoint3D>(vertices_[3]);
  const typename my3d::base::BundleAdjustment::Camera camera =
      v_in->getEstimate();
  const Eigen::Quaterniond rotation = v_r->getEstimate().rotation;
  const Eigen::Vector3d translation = v_t->getEstimate().translation;
  const Eigen::Vector3d &point_3D = v_p->getEstimate();
  const Eigen::Vector3d pc = rotation.inverse() * (point_3D - translation);
  const Eigen::Vector2d projection = camera.camera.project(pc);
  residual_ = projection - measurement_;
}

} // namespace gopt

namespace my3d {
namespace base {

bool BundleAdjustment::Camera::hasBogusParams(
    const double &min_focal_length_ratio,
    const double &max_focal_length_ratio) {
  const size_t size = std::max(width, height);
  const double fx_ratio = camera.fx_ / size;
  const double fy_ratio = camera.fy_ / size;
  bool has_bogus_focal_lengths =
      fx_ratio < min_focal_length_ratio || fx_ratio > max_focal_length_ratio ||
      fy_ratio < min_focal_length_ratio || fy_ratio > max_focal_length_ratio;
  bool has_bogus_principal_points = camera.cx_ < 0.0 || camera.cx_ > width ||
                                    camera.cy_ < 0.0 || camera.cy_ > height;

  return has_bogus_focal_lengths || has_bogus_principal_points;
}

BundleAdjustment::BundleAdjustment(const Config &config) : config_{config} {}

BundleAdjustment::~BundleAdjustment() {
  for (auto &camera : cameras_) {
    if (camera.second != nullptr) {
      delete camera.second;
    }
  }
  for (auto &point_3D : points_3D_) {
    if (point_3D.second != nullptr) {
      delete point_3D.second;
    }
  }
}

bool BundleAdjustment::addCamera(Camera *camera) {
  if (cameras_.find(camera->id) != cameras_.end()) {
    return false;
  } else if (camera == nullptr) {
    return false;
  }

  cameras_.insert(std::make_pair(camera->id, camera));

  return true;
}

bool BundleAdjustment::addPoint3D(Point3D *point_3D) {
  if (points_3D_.find(point_3D->id) != points_3D_.end()) {
    return false;
  } else if (point_3D == nullptr) {
    return false;
  }

  points_3D_.insert(std::make_pair(point_3D->id, point_3D));

  return true;
}

bool BundleAdjustment::addObservation(size_t id_point_3D, size_t id_camera,
                                      const Eigen::Vector2d &observation) {
  if (cameras_.find(id_camera) == cameras_.end() ||
      points_3D_.find(id_point_3D) == points_3D_.end()) {
    return false;
  }
  Camera *camera = cameras_.at(id_camera);
  Point3D *point_3D = points_3D_.at(id_point_3D);

  camera->points_2D.push_back(observation);
  point_3D->observations.insert(
      std::make_pair(id_camera, camera->points_2D.size() - 1UL));

  return true;
}

bool BundleAdjustment::addObservation(size_t id_point_3D, size_t id_camera,
                                      size_t observation_idx) {
  if (cameras_.find(id_camera) == cameras_.end() ||
      points_3D_.find(id_point_3D) == points_3D_.end()) {
    return false;
  }
  Camera *camera = cameras_.at(id_camera);
  Point3D *point_3D = points_3D_.at(id_point_3D);

  if (observation_idx >= camera->points_2D.size()) {
    return false;
  }
  point_3D->observations.insert(std::make_pair(id_camera, observation_idx));

  return true;
}

void BundleAdjustment::removeCamera(size_t id) {
  auto it = cameras_.find(id);
  if (it != cameras_.end()) {
    delete it->second;
    cameras_.erase(it);
  }
}

void BundleAdjustment::removePoint3D(size_t id) {
  auto it = points_3D_.find(id);
  if (it != points_3D_.end()) {
    delete it->second;
    points_3D_.erase(it);
  }
}

void BundleAdjustment::removeObservation(size_t id_point_3D, size_t id_camera) {
  auto it_point_3D = points_3D_.find(id_point_3D);
  auto it_camera = cameras_.find(id_camera);
  if (it_point_3D != points_3D_.end() && it_camera != cameras_.end()) {
    Point3D *point_3D = it_point_3D->second;
    point_3D->observations.erase(id_camera);
  }
}

void BundleAdjustment::config(const Config &config) { config_ = config; }

BundleAdjustment::Report BundleAdjustment::optimize() {
  Report report;

  gopt::OptSolverBase *solver;
  switch (config_.optimization_algorithm) {
  case OptAlgorithm::ALGORITHM_GAUSS_NEWTON_DENSE:
    solver = new gopt::GaussNewtonSolver;

  case OptAlgorithm::ALGORITHM_GAUSS_NEWTON_SHUR:
    solver = new gopt::GaussNewtonShurSolver;

  case OptAlgorithm::ALGORITHM_GAUSS_NEWTON_SPARSE_SHUR:
    solver = new gopt::GaussNewtonSparseShurSolver;

  case OptAlgorithm::ALGORITHM_LEVENBERG_MARQUART_SPARSE_SHUR:
    solver = new gopt::LevenbergMarquartSparseShurSolver;

  default:
    solver = new gopt::LevenbergMarquartSparseShurSolver;
  }

  gopt::FactorGraph graph;
  graph.setOptSolver(solver);

  gopt::optimization_config_t gopt_config;
  gopt_config.epsilon = 1e-3;
  gopt_config.verbose = config_.verbose;
  gopt_config.max_iteration_num = config_.max_num_iters;
  graph.setOptConfig(gopt_config);

  /* Add the vertices */
  size_t id_vertex = 0;
  gopt_vertices_camera_.clear();
  for (auto &camera : cameras_) {

    switch (config_.cam_opt_mode) {
    case CameraOptimizationMode::FULL: {
      gopt::FactorGraph::VertexPtr v;
      v = std::make_shared<gopt::VertexCamera>(
          id_vertex, *(camera.second), camera.second->fixed,
          camera.second->fix_intrinsics, camera.second->fix_translation,
          camera.second->fix_rotation);
      if (!graph.addVertex(v)) {
        report.success = false;
        report.error_info =
            "Failed to add vertex for camera " + std::to_string(camera.first);

        return report;
      }
      gopt_vertices_camera_.insert(std::make_pair(camera.first, v));
      id_vertex++;
      break;
    }

    case CameraOptimizationMode::ONLY_POSE: {
      gopt::FactorGraph::VertexPtr v;
      v = std::make_shared<gopt::VertexCameraPose>(id_vertex, *(camera.second),
                                                   camera.second->fixed);
      if (!graph.addVertex(v)) {
        report.success = false;
        report.error_info =
            "Failed to add vertex for camera " + std::to_string(camera.first);

        return report;
      }
      gopt_vertices_camera_.insert(std::make_pair(camera.first, v));
      id_vertex++;
      break;
    }

    case CameraOptimizationMode::FULL_AND_SEPARATED: {
      gopt::FactorGraph::VertexPtr v_in =
          std::make_shared<gopt::VertexCameraIntrinsics>(
              id_vertex, *(camera.second), camera.second->fix_intrinsics);
      gopt::FactorGraph::VertexPtr v_t =
          std::make_shared<gopt::VertexCameraTranslation>(
              id_vertex, *(camera.second), camera.second->fix_translation);
      gopt::FactorGraph::VertexPtr v_r =
          std::make_shared<gopt::VertexCameraRotation>(
              id_vertex, *(camera.second), camera.second->fix_rotation);
      if (!(graph.addVertex(v_in) && graph.addVertex(v_t) &&
            graph.addVertex(v_r))) {
        report.success = false;
        report.error_info =
            "Failed to add vertex for camera " + std::to_string(camera.first);

        return report;
      }
      gopt_vertices_camera_.insert(std::make_pair(camera.first, v_in));
      gopt_vertices_camera_intrinsics_.insert(
          std::make_pair(camera.first, v_in));
      gopt_vertices_camera_translation_.insert(
          std::make_pair(camera.first, v_t));
      gopt_vertices_camera_rotation_.insert(std::make_pair(camera.first, v_r));
      id_vertex++;
      break;
    }

    default: {
      gopt::FactorGraph::VertexPtr v;
      v = std::make_shared<gopt::VertexCameraPose>(id_vertex, *(camera.second),
                                                   camera.second->fixed);
      if (!graph.addVertex(v)) {
        report.success = false;
        report.error_info =
            "Failed to add vertex for camera " + std::to_string(camera.first);

        return report;
      }
      gopt_vertices_camera_.insert(std::make_pair(camera.first, v));
      id_vertex++;
      break;
    }
    }
  }
  gopt_vertices_point_3D_.clear();
  for (auto &point_3D : points_3D_) {
    auto v = std::make_shared<gopt::VertexPoint3D>();
    v->setId(id_vertex);
    v->setEstimate(point_3D.second->position);
    v->setFixed(point_3D.second->fixed);
    v->setMarginalized(true);
    if (!graph.addVertex(v)) {
      report.success = false;
      report.error_info =
          "Failed to add vertex for 3D point " + std::to_string(point_3D.first);

      return report;
    }
    gopt_vertices_point_3D_.insert(std::make_pair(point_3D.first, v));
    id_vertex++;
  }

  /* Add the edges */
  size_t id_edge = 0;
  gopt_edges_projection_.clear();
  for (auto &point_3D : points_3D_) {
    const size_t id_point_3D = point_3D.first;
    if (gopt_vertices_point_3D_.find(id_point_3D) ==
        gopt_vertices_point_3D_.end()) {
      continue;
    }

    for (auto &observation : point_3D.second->observations) {
      const size_t id_camera = observation.first;
      const size_t idx_point_2D = observation.second;
      if (gopt_vertices_camera_.find(id_camera) !=
          gopt_vertices_camera_.end()) {
        if (idx_point_2D >= cameras_.at(id_camera)->points_2D.size()) {
          continue;
        }

        const Eigen::Vector2d measurement =
            cameras_.at(id_camera)->points_2D[idx_point_2D];
        // auto edge = std::make_shared<gopt::EdgeProjection>();
        // edge->setId(id_edge);
        // edge->setVertex(0, gopt_vertices_camera_.at(id_camera));
        // edge->setVertex(1, gopt_vertices_point_3D_.at(id_point_3D));
        // edge->setMeasurement(measurement);
        // edge->setInformation(Eigen::Matrix2d::Identity());
        // edge->setLossFunction(std::make_shared<gopt::HuberLoss>());
        gopt::FactorGraph::EdgePtr edge;
        // const std::shared_ptr<gopt::LossFunctionBase> loss = nullptr;
        const std::shared_ptr<gopt::LossFunctionBase> loss =
            std::make_shared<gopt::HuberLoss>();
        switch (config_.cam_opt_mode) {
        case CameraOptimizationMode::FULL:
          edge = std::make_shared<gopt::EdgeProjectionFull>(
              id_edge, gopt_vertices_camera_.at(id_camera),
              gopt_vertices_point_3D_.at(id_point_3D), measurement,
              Eigen::Matrix2d::Identity(), loss);
          break;

        case CameraOptimizationMode::ONLY_POSE:
          edge = std::make_shared<gopt::EdgeProjectionOnlyPose>(
              id_edge, gopt_vertices_camera_.at(id_camera),
              gopt_vertices_point_3D_.at(id_point_3D), measurement,
              Eigen::Matrix2d::Identity(), loss);
          break;

        case CameraOptimizationMode::FULL_AND_SEPARATED:
          edge = std::make_shared<gopt::EdgeProjectionSeparated>(
              id_edge, gopt_vertices_camera_intrinsics_.at(id_camera),
              gopt_vertices_camera_translation_.at(id_camera),
              gopt_vertices_camera_rotation_.at(id_camera),
              gopt_vertices_point_3D_.at(id_point_3D), measurement,
              Eigen::Matrix2d::Identity(), loss);
          break;

        default:
          edge = std::make_shared<gopt::EdgeProjectionFull>(
              id_edge, gopt_vertices_camera_.at(id_camera),
              gopt_vertices_point_3D_.at(id_point_3D), measurement,
              Eigen::Matrix2d::Identity(), loss);
          break;
        }
        if (!graph.addEdge(edge)) {
          report.success = false;
          report.error_info = "Failed to add edge for camera " +
                              std::to_string(id_camera) + " and 3D point " +
                              std::to_string(id_point_3D);

          return report;
        }
        gopt_edges_projection_.push_back(
            EdgeInfo(id_camera, id_point_3D, edge));
        id_edge++;
      }
    }
  }

  std::cout << "[BundleAdjustmentINFO] "
            << "Processing " << gopt_vertices_camera_.size() << " cameras, "
            << gopt_vertices_point_3D_.size() << " 3D points, "
            << gopt_edges_projection_.size() << " observations. " << std::endl;
  int ret = graph.optimize();
  report.success = (ret >= 0) ? true : false;
  report.ret_val = ret;
  report.final_cost = graph.getCost();
  report.num_cameras = gopt_vertices_camera_.size();
  report.num_points_3D = gopt_vertices_point_3D_.size();
  report.num_observations = gopt_edges_projection_.size();

  double rmse = 0;
  for (auto &edge_info : gopt_edges_projection_) {
    rmse += edge_info.edge->computeError2();
  }
  rmse /= static_cast<double>(gopt_edges_projection_.size());
  report.final_squared_reprojection_error = rmse;

  update();

  return report;
}

void BundleAdjustment::update() {
  for (auto &v : gopt_vertices_camera_) {
    Camera *camera = cameras_.at(v.first);
    switch (config_.cam_opt_mode) {
    case CameraOptimizationMode::FULL:
      *camera = std::dynamic_pointer_cast<gopt::VertexCamera>(v.second)
                    ->getEstimate();
      break;

    case CameraOptimizationMode::ONLY_POSE:
      *camera = std::dynamic_pointer_cast<gopt::VertexCameraPose>(v.second)
                    ->getEstimate();
      break;

    case CameraOptimizationMode::FULL_AND_SEPARATED:
      *camera =
          std::dynamic_pointer_cast<gopt::VertexCameraIntrinsics>(v.second)
              ->getEstimate();
      camera->id = v.first;
      camera->rotation =
          std::dynamic_pointer_cast<gopt::VertexCameraRotation>(v.second)
              ->getEstimate()
              .rotation;
      camera->translation =
          std::dynamic_pointer_cast<gopt::VertexCameraTranslation>(v.second)
              ->getEstimate()
              .translation;
      break;

    default:
      *camera = std::dynamic_pointer_cast<gopt::VertexCamera>(v.second)
                    ->getEstimate();
      break;
    }
    // *camera = v.second->getEstimate();
  }
  for (auto &v : gopt_vertices_point_3D_) {
    Point3D *point_3D = points_3D_.at(v.first);
    point_3D->position = v.second->getEstimate();
  }
}

} // namespace base
} // namespace my3d