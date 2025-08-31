/*
 * filename: BundleAdjustment.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_BASE_RECONSTRUCTION_BUNDLE_ADJUSTMENT_H_
#define _MY3D_BASE_RECONSTRUCTION_BUNDLE_ADJUSTMENT_H_

#include <iostream>
#include <unordered_map>
#include <vector>

#include "base/camera/RadialPinHoleCamera.h"
#include "base/camera/projection.h"
#include "base/pose.h"
#include "utils/eigen_types.h"

#include "gopt/graph/BaseBinaryEdge.h"
#include "gopt/graph/BaseVertex.h"
#include "gopt/graph/FactorGraph.h"
#include "gopt/loss/HuberLoss.h"
#include "gopt/solver/GaussNewtonShurSolver.h"
#include "gopt/solver/GaussNewtonSolver.h"
#include "gopt/solver/GaussNewtonSparseShurSolver.h"
#include "gopt/solver/LevenbergMarquartSparseShurSolver.h"

namespace gopt {
struct VertexCamera;
struct VertexPoint3D;
struct EdgeProjection;
} // namespace gopt

namespace my3d {
namespace base {

class BundleAdjustment {
public:
  enum OptAlgorithm {
    ALGORITHM_LEVENBERG_MARQUART_SPARSE_SHUR = 0,
    ALGORITHM_GAUSS_NEWTON_DENSE = 1,
    ALGORITHM_GAUSS_NEWTON_SHUR = 2,
    ALGORITHM_GAUSS_NEWTON_SPARSE_SHUR = 3
  };

  enum CameraOptimizationMode {
    FULL = 0,
    FULL_AND_SEPARATED = 1,
    ONLY_POSE = 2
  };

  struct Config;
  struct Report;
  struct Camera;
  struct Point3D;
  struct EdgePair;
  struct EdgeInfo;

  struct Config {
    size_t max_num_iters{20};
    OptAlgorithm optimization_algorithm{
        OptAlgorithm::ALGORITHM_LEVENBERG_MARQUART_SPARSE_SHUR};
    bool verbose{true};
    CameraOptimizationMode cam_opt_mode{CameraOptimizationMode::FULL};
  };

  struct Report {
    int num_iter{-1};
    double final_cost{-1};
    double final_squared_reprojection_error{-1};
    bool success{false};
    int ret_val;
    std::string error_info{""};
    size_t num_cameras{0};
    size_t num_points_3D{0};
    size_t num_observations{0};
    std::unordered_map<size_t, Camera *> cameras;
    std::unordered_map<size_t, Point3D *> points_3D;
  };

  struct Camera {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t id;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d translation;
    EigenVec<Eigen::Vector2d> points_2D;
    size_t width;
    size_t height;
    RadialPinHoleCamera camera;
    bool fixed{false};
    bool fix_intrinsics{false};  // Used for FULL(_AND_SEPARATED) mode
    bool fix_translation{false}; // Used for FULL(_AND_SEPARATED) mode
    bool fix_rotation{false};    // Used for FULL(_AND_SEPARATED) mode

    bool hasBogusParams(const double &min_focal_length_ratio,
                        const double &max_focal_length_ratio);
  };

  struct Point3D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t id;
    Eigen::Vector3d position;
    /* Observations represented by (camera_id, points_2D_idx) */
    std::unordered_map<size_t, size_t> observations;
    bool fixed{false};
  };

  struct EdgePair {
    EdgePair() {}
    EdgePair(size_t idc, size_t idp) : id_camera(idc), id_point_3D(idp) {}

    size_t id_camera;
    size_t id_point_3D;
  };

  struct EdgeInfo {
    EdgeInfo() {}
    EdgeInfo(size_t idc, size_t idp,
             const std::shared_ptr<gopt::FactorGraph::Edge> e)
        : id_camera{idc}, id_point_3D{idp}, edge{e} {}

    size_t id_camera;
    size_t id_point_3D;
    std::shared_ptr<gopt::FactorGraph::Edge> edge;
  };

public:
  BundleAdjustment() = delete;
  explicit BundleAdjustment(const Config &config);
  ~BundleAdjustment();

  /**
   * @brief Add an camera into the BA problem. The function will do nothing and
   * return false if 1) there already exists a camera with the same ID 2) camera
   * == nullptr
   */
  bool addCamera(Camera *camera);

  /**
   * @brief Add an camera into the BA problem. The function will do nothing and
   * return false if 1) there already exists a camera with the same ID
   */
  bool addCamera(
      const size_t &id, const size_t &width, const size_t &height,
      const bool &fixed = false,
      const Eigen::Quaterniond &rotation = Eigen::Quaterniond::Identity(),
      const Eigen::Vector3d &translation = Eigen::Vector3d::Zero(),
      const EigenVec<Eigen::Vector2d> &points_2D = EigenVec<Eigen::Vector2d>(),
      const base::RadialPinHoleCamera camera_model =
          base::RadialPinHoleCamera()) {
    Camera *camera = new Camera;
    camera->id = id;
    camera->rotation = rotation.normalized();
    camera->translation = translation;
    camera->points_2D = points_2D;
    camera->camera = camera_model;
    camera->width = width;
    camera->height = height;
    camera->fixed = fixed;
    camera->fix_intrinsics = fixed;
    camera->fix_translation = fixed;
    camera->fix_rotation = fixed;

    return addCamera(camera);
  }

  /**
   * @brief Add an camera into the BA problem. The function will do nothing and
   * return false if 1) there already exists a camera with the same ID
   */
  bool addCamera(
      const size_t &id, const size_t &width, const size_t &height,
      const bool &fix_intrinsics, const bool &fix_translation,
      const bool &fix_rotation,
      const Eigen::Quaterniond &rotation = Eigen::Quaterniond::Identity(),
      const Eigen::Vector3d &translation = Eigen::Vector3d::Zero(),
      const EigenVec<Eigen::Vector2d> &points_2D = EigenVec<Eigen::Vector2d>(),
      const base::RadialPinHoleCamera camera_model =
          base::RadialPinHoleCamera()) {
    Camera *camera = new Camera;
    camera->id = id;
    camera->rotation = rotation;
    camera->translation = translation;
    camera->points_2D = points_2D;
    camera->camera = camera_model;
    camera->width = width;
    camera->height = height;
    camera->fixed = fix_intrinsics && fix_translation && fix_rotation;
    camera->fix_intrinsics = fix_intrinsics;
    camera->fix_translation = fix_translation;
    camera->fix_rotation = fix_rotation;

    return addCamera(camera);
  }

  /**
   * @brief Add an 3D point into the BA problem. The function will do nothing
   * and return false if there already exists a point with the same ID.
   */
  bool addPoint3D(Point3D *point_3D);

  /**
   * @brief Add an 3D point into the BA problem. The function will do nothing
   * and return false if there already exists a point with the same ID.
   */
  bool addPoint3D(size_t id,
                  const Eigen::Vector3d &position = Eigen::Vector3d::Zero(),
                  const std::unordered_map<size_t, size_t> &observations =
                      std::unordered_map<size_t, size_t>(),
                  const bool &fixed = false) {
    Point3D *point_3D = new Point3D;
    point_3D->id = id;
    point_3D->position = position;
    point_3D->observations = observations;
    point_3D->fixed = fixed;

    return addPoint3D(point_3D);
  }

  /**
   * @brief Add an observation into the BA problem. The function will do nothing
   * and return false if the specified 3D point or camera does not exist. Note
   * that this function will not check the repetition of observations in the
   * specified camera.
   */
  bool addObservation(size_t id_point_3D, size_t id_camera,
                      const Eigen::Vector2d &observation);

  /**
   * @brief Add an observation into the BA problem. The function will do nothing
   * and return false if the specified 3D point or camera does not exist, or
   * observation_idx exceeds the limit
   */
  bool addObservation(size_t id_point_3D, size_t id_camera,
                      size_t observation_idx);

  /**
   * @brief Remove the node with ID id. If the node with the given ID does not
   * exist, this function will do nothing. Note that this function will not
   * remove the observations corresponding to the removed camera in 3D points.
   */
  void removeCamera(size_t id);

  /**
   * @brief Remove the node with ID id. If the node with the given ID does not
   * exist, this function will do nothing.
   */
  void removePoint3D(size_t id);

  /**
   * @brief Remove an observation into the BA problem. The function will do
   * nothing  if the specified 3D point or camera does not exist.
   */
  void removeObservation(size_t id_point_3D, size_t id_camera);

  /**
   * @brief Set bundle adjustment configuration.
   */
  void config(const Config &config);

  /**
   * @brief Solve the BA problem constructed by the cameras, 3D points, and
   * corresponding observations you add.
   */
  Report optimize();

  /**
   * @brief Get number of existing cameras in the BA problem.
   */
  size_t getNumCameras() const { return cameras_.size(); }

  /**
   * @brief Get number of existing 3D points in the BA problem.
   */
  size_t getNumPoints3D() const { return points_3D_.size(); }

  /**
   * @brief Update camera parameters and 3D point coordinates to cameras_ and
   * points_3D_ after optimization
   */
  void update();

  /**
   * @brief Get the cameras
   */
  std::unordered_map<size_t, Camera *> &getCameras() { return cameras_; }

  /**
   * @brief Get the 3D points
   */
  std::unordered_map<size_t, Point3D *> &getPoints3D() { return points_3D_; }

  /**
   * @brief Get the cameras (const version)
   */
  const std::unordered_map<size_t, Camera *> &getCameras() const {
    return cameras_;
  }

  /**
   * @brief Get the 3D points (const version)
   */
  const std::unordered_map<size_t, Point3D *> &getPoints3D() const {
    return points_3D_;
  }

  /**
   * @brief Only should be called if camera optimization mode is
   * FULL(_AND_SEPARATED)
   */
  void fixCameraIntrinsics(const size_t &camera_id) {
    if (config_.cam_opt_mode == CameraOptimizationMode::FULL_AND_SEPARATED &&
        gopt_vertices_camera_intrinsics_.find(camera_id) !=
            gopt_vertices_camera_intrinsics_.end()) {
      gopt_vertices_camera_intrinsics_.at(camera_id)->setFixed(true);
    }
  }

  /**
   * @brief Only should be called if camera optimization mode is
   * FULL(_AND_SEPARATED)
   */
  void fixCameraTranslation(const size_t &camera_id) {
    if (config_.cam_opt_mode == CameraOptimizationMode::FULL_AND_SEPARATED &&
        gopt_vertices_camera_translation_.find(camera_id) !=
            gopt_vertices_camera_translation_.end()) {
      gopt_vertices_camera_translation_.at(camera_id)->setFixed(true);
    }
  }

  /**
   * @brief Only should be called if camera optimization mode is
   * FULL(_AND_SEPARATED)
   */
  void fixCameraRotation(const size_t &camera_id) {
    if (config_.cam_opt_mode == CameraOptimizationMode::FULL_AND_SEPARATED &&
        gopt_vertices_camera_rotation_.find(camera_id) !=
            gopt_vertices_camera_rotation_.end()) {
      gopt_vertices_camera_rotation_.at(camera_id)->setFixed(true);
    }
  }

private:
  Config config_;
  std::unordered_map<size_t, Camera *> cameras_;
  std::unordered_map<size_t, Point3D *> points_3D_;
  std::unordered_map<size_t, std::shared_ptr<gopt::FactorGraph::Vertex>>
      gopt_vertices_camera_;
  std::unordered_map<size_t, std::shared_ptr<gopt::FactorGraph::Vertex>>
      gopt_vertices_camera_intrinsics_;
  std::unordered_map<size_t, std::shared_ptr<gopt::FactorGraph::Vertex>>
      gopt_vertices_camera_translation_;
  std::unordered_map<size_t, std::shared_ptr<gopt::FactorGraph::Vertex>>
      gopt_vertices_camera_rotation_;
  std::unordered_map<size_t, std::shared_ptr<gopt::VertexPoint3D>>
      gopt_vertices_point_3D_;
  std::vector<EdgeInfo> gopt_edges_projection_;
};

} // namespace base
} // namespace my3d

namespace gopt {

/**
 * @brief Vertex representing a camera. This class is not responsible for
 * managing the memory to which the Camera pointer points.
 *
 */
template <size_t Dim>
struct VertexCameraBase
    : public gopt::BaseVertex<Dim,
                              typename my3d::base::BundleAdjustment::Camera> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCameraBase() {}
  virtual ~VertexCameraBase() {}
};

struct VertexCamera : public VertexCameraBase<10> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCamera();
  VertexCamera(const typename my3d::base::BundleAdjustment::Camera &camera);
  VertexCamera(const size_t &id,
               const typename my3d::base::BundleAdjustment::Camera &camera,
               const bool &fixed, const bool &fix_intrinsics = false,
               const bool &fix_translation = false,
               const bool &fix_rotation = false);

  virtual void setToOrigin() override;

  virtual void plus(const Eigen::VectorXd &update) override;

  bool fix_intrinsics_{false};
  bool fix_translation_{false};
  bool fix_rotation_{false};
};

struct VertexCameraPose : public VertexCameraBase<6> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCameraPose();
  VertexCameraPose(const typename my3d::base::BundleAdjustment::Camera &camera);
  VertexCameraPose(const size_t &id,
                   const typename my3d::base::BundleAdjustment::Camera &camera,
                   const bool &fixed);

  virtual void setToOrigin() override;

  virtual void plus(const Eigen::VectorXd &update) override;
};

struct VertexCameraTranslation : public VertexCameraBase<3> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCameraTranslation();
  VertexCameraTranslation(
      const typename my3d::base::BundleAdjustment::Camera &camera);
  VertexCameraTranslation(
      const size_t &id,
      const typename my3d::base::BundleAdjustment::Camera &camera,
      const bool &fixed);

  virtual void setToOrigin() override;

  virtual void plus(const Eigen::VectorXd &update) override;
};

struct VertexCameraRotation : public VertexCameraBase<3> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCameraRotation();
  VertexCameraRotation(
      const typename my3d::base::BundleAdjustment::Camera &camera);
  VertexCameraRotation(
      const size_t &id,
      const typename my3d::base::BundleAdjustment::Camera &camera,
      const bool &fixed);

  virtual void setToOrigin() override;

  virtual void plus(const Eigen::VectorXd &update) override;
};

struct VertexCameraIntrinsics : public VertexCameraBase<6> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCameraIntrinsics();
  VertexCameraIntrinsics(
      const typename my3d::base::BundleAdjustment::Camera &camera);
  VertexCameraIntrinsics(
      const size_t &id,
      const typename my3d::base::BundleAdjustment::Camera &camera,
      const bool &fixed);

  virtual void setToOrigin() override;

  virtual void plus(const Eigen::VectorXd &update) override;
};

/**
 * @brief Vertex representing a 3D point.
 */
struct VertexPoint3D : public gopt::BaseVertex<3, Eigen::Vector3d> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexPoint3D();
  VertexPoint3D(const Eigen::Vector3d &point_3D);

  virtual void setToOrigin() override;

  virtual void plus(const Eigen::VectorXd &update) override;
};

/**
 * @brief Edge representing the re-projection residual.
 */
struct EdgeProjectionFull
    : public gopt::BaseBinaryEdge<2, Eigen::Vector2d, VertexCamera,
                                  VertexPoint3D> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectionFull();
  EdgeProjectionFull(const size_t &id,
                     const gopt::FactorGraph::VertexPtr vertex_camera,
                     const gopt::FactorGraph::VertexPtr vertex_point_3D,
                     const Eigen::Vector2d &measurement,
                     const Eigen::Matrix2d &information,
                     const std::shared_ptr<gopt::LossFunctionBase> &loss) {
    this->setId(id);
    this->setVertex(0, vertex_camera);
    this->setVertex(1, vertex_point_3D);
    this->setMeasurement(measurement);
    this->setInformation(information);
    this->setLossFunction(loss);
  }

  virtual void computeResidual() override;
};

struct EdgeProjectionOnlyPose
    : public gopt::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraPose,
                                  VertexPoint3D> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectionOnlyPose();
  EdgeProjectionOnlyPose(const size_t &id,
                         const gopt::FactorGraph::VertexPtr vertex_camera,
                         const gopt::FactorGraph::VertexPtr vertex_point_3D,
                         const Eigen::Vector2d &measurement,
                         const Eigen::Matrix2d &information,
                         const std::shared_ptr<gopt::LossFunctionBase> &loss) {
    this->setId(id);
    this->setVertex(0, vertex_camera);
    this->setVertex(1, vertex_point_3D);
    this->setMeasurement(measurement);
    this->setInformation(information);
    this->setLossFunction(loss);
  }

  virtual void computeResidual() override;
};

struct EdgeProjectionSeparated : public gopt::BaseEdge<2, Eigen::Vector2d> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectionSeparated();
  // The camera pointer, i.e. the member estimator_ of the 3 camera vertices
  // should be the same
  EdgeProjectionSeparated(
      const size_t &id,
      const gopt::FactorGraph::VertexPtr vertex_camera_intrinsics,
      const gopt::FactorGraph::VertexPtr vertex_camera_translation,
      const gopt::FactorGraph::VertexPtr vertex_camera_rotation,
      const gopt::FactorGraph::VertexPtr vertex_point_3D,
      const Eigen::Vector2d &measurement, const Eigen::Matrix2d &information,
      const std::shared_ptr<gopt::LossFunctionBase> &loss) {
    this->setId(id);
    this->vertices_.resize(4);
    this->setVertex(0, vertex_camera_intrinsics);
    this->setVertex(1, vertex_camera_translation);
    this->setVertex(2, vertex_camera_rotation);
    this->setVertex(3, vertex_point_3D);
    this->setMeasurement(measurement);
    this->setInformation(information);
    this->setLossFunction(loss);
  }

  virtual void computeResidual() override;
};

} // namespace gopt

#endif // _MY3D_BASE_RECONSTRUCTION_BUNDLE_ADJUSTMENT_H_