/*
 * filename: Estimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    estimator for SimpleVO
 */

#ifndef _MY3D_SLAM_SIMPLE_VO_ESTIMATOR_H_
#define _MY3D_SLAM_SIMPLE_VO_ESTIMATOR_H_

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "FeatureTracker.h"
#include "estimators/AP3PPoseEstimator.h"
#include "estimators/EPnPPoseEstimator.h"
#include "estimators/EssentialMatrixEstimator.h"
#include "estimators/FundamentalMatrixEstimator.h"
#include "estimators/HomographyEstimator.h"
#include "estimators/LORANSAC.h"
#include "gopt/graph/BaseBinaryEdge.h"
#include "gopt/graph/BaseVertex.h"
#include "gopt/graph/FactorGraph.h"
#include "gopt/loss/HuberLoss.h"
#include "gopt/solver/GaussNewtonSchurSolver.h"
#include "gopt/solver/GaussNewtonSolver.h"
#include "gopt/solver/GaussNewtonSparseSchurSolver.h"
#include "gopt/solver/LevenbergMarquartSparseSchurSolver.h"
#include "sfm/SceneBuilder.h"
#include "utils/math.h"

namespace my3d {
namespace slam {
namespace simple_vo {

//* Forward declaration
class EstimatorVisualizer;

struct Observation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector2d pixel_coord{0.0, 0.0};
  Eigen::Vector3d sphere_coord{1.0, 0.0, 0.0};
  Observation() = default;
  Observation(Eigen::Vector2d const &_pixel_coord,
              Eigen::Vector3d const &_sphere_coord)
      : pixel_coord{_pixel_coord}, sphere_coord{_sphere_coord} {}
};

struct Track {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  size_t id{0UL};
  size_t start_idx{0UL};
  double depth{1.0};
  Eigen::Vector3d position{};
  EigenVec<Observation> observations{};
  bool is_triangulated{false};
  bool valid{false};
};

struct Frame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp{0.0};

  size_t id{0UL};

  Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};

  /* (track_id, observation) */
  EigenUMap<size_t, Observation> features{};

  Eigen::Matrix3x4d getPose() const {
    return base::composePose(rotation.toRotationMatrix(), translation);
  }

  Eigen::Matrix3x4d getProjectionMatrix() const {
    return base::composePose(rotation.inverse().toRotationMatrix(),
                             -rotation.inverse().toRotationMatrix() *
                                 translation);
  }
};

class Estimator {
  friend class EstimatorVisualizer;

public:
  typedef estimator::LORANSAC<estimator::AP3PPoseEstimator,
                              estimator::EPnPPoseEstimator>
      AbsolutePoseEstimator;

  enum MarginType {
    NONE = -1,
    MARGIN_OLDEST = 0,
    MARGIN_SECOND_NEWEST = 1,
    MARGIN_NEWEST = 2
  };

  struct Config {
    double min_avg_parallax_init_deg{0.6};
    estimator::RANSACConfig init_h_ransac_config{};
    estimator::RANSACConfig init_e_ransac_config{};
    double min_init_two_view_geom_inlier_ratio{0.2};
    double max_h_inlier_ratio{0.9};
    size_t min_num_init_tracks{20UL};
    double min_abs_pose_inlier_ratio{0.5};
    double min_num_frames_opt{2UL};
    /* max absolute pose estimation inlier reprojection error in pixel */
    double max_abs_pose_reproj_error{12.0};
    double max_valid_track_avg_proj_error_deg{0.5};
    double min_frame_rel_trans{3.0};
    double min_avg_parallax_keyframe_deg{0.3};
    size_t max_num_covis_keyframe{30UL};
    double dist_init_pair{1.0};

    size_t window_size{10UL};
    FeatureTracker::Config feature_tracker_config{};

    bool Check() const {
      return min_avg_parallax_init_deg > 0.0 && window_size > 2UL &&
             feature_tracker_config.Check();
    }
  };

  struct AbsPoseEstReport {
    bool success{false};
    Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
    std::vector<bool> inlier_mask{};
    size_t num_inliers{0UL};
  };

  struct RegistReport {
    bool success{false};
    size_t frame_id{0UL};
    Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
    bool is_keyframe{false};
  };

  struct LastFrameInfo {
    std::shared_ptr<base::Image> last_image{};
    double last_timestamp{0.0};
    EigenUMap<size_t, FeatureTracker::InputPoint> last_pts{};
    Eigen::Quaterniond last_tracking_rotation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d last_tracking_translation{Eigen::Vector3d::Zero()};

    Eigen::Matrix3x4d getPose() const {
      return base::composePose(last_tracking_rotation.toRotationMatrix(),
                               last_tracking_translation);
    }

    Eigen::Matrix3x4d getProjectionMatrix() const {
      return base::composePose(
          last_tracking_rotation.inverse().toRotationMatrix(),
          -last_tracking_rotation.inverse().toRotationMatrix() *
              last_tracking_translation);
    }
  };

  explicit Estimator(Config const &config) noexcept;

  Estimator(Estimator const &other) = delete;
  Estimator &operator=(Estimator const &other) = delete;

  void reset() noexcept;

  void feedFrame(std::shared_ptr<base::Image> const &image,
                 double const timestamp) noexcept;

  bool isInitialized() const { return is_initialized_; }

  inline void
  getCameraPoses(EigenVec<Eigen::Quaterniond> &rotations,
                 EigenVec<Eigen::Vector3d> &translations) const noexcept {
    rotations.clear();
    translations.clear();
    rotations.reserve(sliding_window_.size());
    translations.reserve(sliding_window_.size());
    for (size_t i{0UL}; i < sliding_window_.size(); ++i) {
      rotations.emplace_back(sliding_window_[i].rotation);
      translations.emplace_back(sliding_window_[i].translation);
    }
  }

  inline bool getTrackedPose(Eigen::Quaterniond &rotation,
                             Eigen::Vector3d &translation) const {
    if (!regist_report_.success && is_initialized_) {
      return false;
    } else {
      rotation = tracking_rotation_;
      translation = tracking_translation_;
    }

    return true;
  }

  Eigen::Vector3d getTrackPosition(size_t const track_id) const noexcept {
    assert(tracks_.find(track_id) != tracks_.end());
    auto const &track{tracks_.at(track_id)};
    assert(track->valid);
    auto const &start_frame{sliding_window_[track->start_idx]};
    return start_frame.rotation *
               (track->depth * track->observations.front().sphere_coord) +
           start_frame.translation;
  }

  void drawTrackingResult(cv::Mat &image) const noexcept;

private:
  struct InitInfo {};

  //! Must be called after optimize() and filterTracks() and before
  //! slideWindow()
  void updateFrameInfo(std::shared_ptr<base::Image> const &image,
                       double const timestamp,
                       bool const init_flag = false) noexcept;

  void createInitialTracks(
      EigenVec<Eigen::Vector2d> const &initial_corners) noexcept;

  FeatureTracker::Result
  trackFeatures(std::shared_ptr<base::Image> const &image,
                double const timestamp) noexcept;

  bool initialize(FeatureTracker::Result const &ft_result,
                  double const timestamp) noexcept;

  void optimize() noexcept;

  void normalizeFrameTranslations(Eigen::Vector3d &trans,
                                  double &scale) const noexcept;

  void filterTracks() noexcept;

  void marginalize(MarginType const &margin_type) noexcept;

  void slideWindow(MarginType const margin_type) noexcept;

  RegistReport registerFrame(FeatureTracker::Result const &tracked_pts,
                             double const timestamp) noexcept;

  /**
   * @brief Check whether the newest frame is a keyframe
   */
  bool isKeyframe() const;

  //! Make sure the track_id exsists, or coredump will occur
  double computeTrackAvgProjAngleError(size_t const track_id) const noexcept;

  AbsPoseEstReport
  estimateAbsPoseRANSACPnP(EigenVec<Eigen::Vector3d> const &pts_3d,
                           EigenVec<Eigen::Vector2d> const &pts_2d) const;

  AbsPoseEstReport
  estimateAbsPoseIterativePnP(EigenVec<Eigen::Vector3d> const &pts_3d,
                              EigenVec<Eigen::Vector2d> const &pts_2d,
                              Eigen::Quaterniond const &init_rot,
                              Eigen::Vector3d const &init_trans) const;

  Config config_{};

  std::shared_ptr<FeatureTracker> feature_tracker_{nullptr};
  std::vector<Frame> sliding_window_{};
  Eigen::Quaterniond tracking_rotation_{};
  Eigen::Vector3d tracking_translation_{};
  RegistReport regist_report_{};
  std::unordered_map<size_t, std::shared_ptr<Track>> tracks_{};
  std::unordered_map<size_t, std::shared_ptr<Track>> removed_tracks_{};
  std::unordered_map<size_t, std::shared_ptr<Track>> stored_tracks_{};
  std::unordered_map<size_t, std::shared_ptr<Track>> initial_tracks_{};
  /* The total number of tracks created so far (including those that have been
   * deleted) */
  size_t track_cnt_{0UL};
  size_t frame_cnt_{0UL};
  bool is_initialized_{false};
  bool is_ref_frame_selected_{false};
  FeatureTracker::Result tracked_pts_{};
  LastFrameInfo last_frame_info_{};
};

} // namespace simple_vo
} // namespace slam
} // namespace my3d

namespace gopt {

struct CameraPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};

  CameraPose() = default;
  CameraPose(Eigen::Quaterniond const &rot, Eigen::Vector3d const &trans)
      : rotation{rot}, translation{trans} {}
};

struct VertexCameraPose : public BaseVertex<6UL, CameraPose> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCameraPose(){};
  VertexCameraPose(CameraPose const &camera_pose) {
    this->setEstimate(camera_pose);
  }

  void setToOrigin() override;

  void plus(Eigen::VectorXd const &update) override;

  size_t frame_idx{0UL};
  bool fix_rot{false};
  bool fix_trans{false};
};

struct VertexTrackInvDepth : public BaseVertex<1UL, double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexTrackInvDepth(){};
  VertexTrackInvDepth(double const inv_depth) { this->setEstimate(inv_depth); }

  void setToOrigin() override;

  void plus(Eigen::VectorXd const &update) override;

  bool is_fixed{false};
};

struct EdgeProjectionMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d sphere0{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d sphere1{Eigen::Vector3d::UnitX()};
};

struct EdgeProjection : public BaseEdge<2UL, EdgeProjectionMeasurement> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjection() { tangent_base.setIdentity(); }

  EdgeProjection(size_t const id,
                 FactorGraph::VertexPtr const &vertex_camera_pose_0,
                 FactorGraph::VertexPtr const &vertex_camera_pose_1,
                 FactorGraph::VertexPtr const &vertex_inv_depth,
                 EdgeProjectionMeasurement const &measurement,
                 Eigen::Matrix2d const &information,
                 std::shared_ptr<gopt::LossFunctionBase> const &loss) {
    vertices_.resize(3UL);
    jacobians_.resize(3UL);
    residual_.resize(2L);
    this->setId(id);
    this->setVertex(0UL, vertex_camera_pose_0);
    this->setVertex(1UL, vertex_camera_pose_1);
    this->setVertex(2UL, vertex_inv_depth);
    this->setMeasurement(measurement);
    this->setInformation(information);
    this->setLossFunction(loss);
  }

  void setMeasurement(EdgeProjectionMeasurement const &measurement) override {
    BaseEdge<2UL, EdgeProjectionMeasurement>::setMeasurement(measurement);
    setTangentBase(measurement);
  }

  void computeResidual() override;

  // void computeJacobians() override;

  void setTangentBase(EdgeProjectionMeasurement const &measurement);

  bool enable_feg_{false};

protected:
  Eigen::Matrix<double, 3, 2> tangent_base{};
};

// TODO Implement marginalization prior edge

} // namespace gopt

#endif // _MY3D_SLAM_SIMPLE_VO_ESTIMATOR_H_