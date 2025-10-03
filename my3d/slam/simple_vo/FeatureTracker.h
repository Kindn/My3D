/*
 * filename: FeatureTracker.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    feature tracker for SimpleVO
 */

#ifndef _MY3D_SLAM_SIMPLE_VO_FEATURE_TRACKER_H_
#define _MY3D_SLAM_SIMPLE_VO_FEATURE_TRACKER_H_

#include <assert.h>
#include <cstdint>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "base/camera/CameraBase.h"
#include "base/camera/RadialPinHoleCamera.h"
#include "base/image/Image.h"
#include "feature/feature_utils.h"

namespace my3d {
namespace slam {
namespace simple_vo {

class FeatureTracker {
public:
  struct Config {
    int32_t max_num_corners{300};
    double quality_level{0.01};
    int32_t min_dist{30};
    std::shared_ptr<base::CameraBase> camera{nullptr};
    bool enable_two_way_check{true};
    double max_two_way_check_error{0.5};
    bool enable_two_view_geometry_outlier_rejection{false};

    bool Check() const {
      return max_num_corners >= 0 && quality_level >= 0.0 && min_dist > 0 &&
             camera != nullptr;
    }
  };

  struct InputPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t id{0UL};
    Eigen::Vector2d pos{};
    size_t track_len{0UL};
  };

  struct TrackedPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t point_id{0UL};
    Eigen::Vector2d pixel_coord_src{0.0, 0.0};
    Eigen::Vector2d pixel_coord{0.0, 0.0};
    Eigen::Vector3d sphere_coord{1.0, 0.0, 0.0};
    Eigen::Vector2d pixel_vel{0.0, 0.0};
    bool valid{false};
    double lk_error{0.0};
  };

  struct Result {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool success{false};
    EigenUMap<size_t, TrackedPoint> tracked_pts{};
    std::vector<TrackedPoint> tracked_add_pts{};
  };

  explicit FeatureTracker(Config const &config);

  inline EigenVec<Eigen::Vector2d>
  extractInitialCorners(std::shared_ptr<base::Image> const &src) {
    return extractAdditionalCorners(src, {});
  }

  /**
   * @brief
   *
   * @param src    the source image
   * @param tgt    the target image
   * @param pts    the TrackedPoints to be tracked in src
   * @param dt     timestamp of tgt - timestamp of src
   */
  Result track(std::shared_ptr<base::Image> const &src,
               std::shared_ptr<base::Image> const &tgt,
               EigenUMap<size_t, InputPoint> const &pts, double const dt);

  void reset();

  EigenVec<Eigen::Vector2d>
  extractAdditionalCorners(std::shared_ptr<base::Image> const &src,
                           EigenUMap<size_t, InputPoint> const &existing_pts);

private:
  void trackLK(std::shared_ptr<base::Image> const &src,
               std::shared_ptr<base::Image> const &tgt,
               EigenVec<Eigen::Vector2d> const &pts,
               EigenVec<Eigen::Vector2d> &tracked_pts,
               std::vector<uint8_t> &status,
               std::vector<float> &lk_errors) const;

  Config config_{};
  size_t add_pts_cnt_{0UL};
};

} // namespace simple_vo
} // namespace slam
} // namespace my3d

#endif // _MY3D_SLAM_SIMPLE_VO_FEATURE_TRACKER_H_