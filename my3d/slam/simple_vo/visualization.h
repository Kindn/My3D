/*
 * filename: visualization.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_SLAM_SIMPLE_VO_VISUALIZATION_H_
#define _MY3D_SLAM_SIMPLE_VO_VISUALIZATION_H_

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/viz.hpp>

#include "Estimator.h"
#include "base/reconstruction/Reconstruction.h"
#include "utils/threading.h"

namespace my3d {
namespace slam {
namespace simple_vo {

struct VizData {
  struct VizCamera {
    size_t id;
    size_t frame_idx;
    cv::Mat R;
    cv::Mat t;
    cv::Vec3b color;
  };

  struct VizPoint3D {
    size_t id;
    cv::Point3f position;
    cv::Vec3b color;
  };

  std::vector<VizPoint3D> point_cloud;
  std::vector<VizCamera> sliding_window;
  VizCamera tracked_frame;

  void clear() {
    point_cloud.clear();
    sliding_window.clear();
  }
};

class EstimatorVisualizer {
public:
  cv::Vec3b color_camera = cv::Vec3b{50, 50, 255};
  cv::Vec3b color_tracked_camera = cv::Vec3b{50, 255, 50};
  cv::Vec3b color_ref_camera_0 = cv::Vec3b{255, 50, 50};
  cv::Vec3b color_ref_camera_1 = cv::Vec3b{255, 255, 50};
  cv::Vec3b color_modified_point = cv::Vec3b{100, 180, 180};
  cv::Vec3b color_new_point = cv::Vec3b{50, 255, 50};

  EstimatorVisualizer(const std::shared_ptr<Estimator const> &estimator,
                      const std::string &window_name)
      : estimator_{estimator} {
    window_ = cv::viz::Viz3d(window_name);
  }

  void run();

  void threadVisualization();

  void join();

  void close();

  void update() {
    /* Get data from reconstruction_ */
    getVizData();
    updated = true;
  }

private:
  void getVizData();

  cv::Vec3b getCameraColor(const size_t &id, const size_t &idx) const;

  cv::Vec3b getPoint3DColor(const size_t &track_id) const;

private:
  cv::viz::Viz3d window_;
  std::thread thread_viz_;
  std::unique_ptr<util::ThreadGuard> thread_guard_viz_;
  VizData viz_data_;
  std::shared_ptr<Estimator const> estimator_;
  size_t num_current_cameras_{0};
  size_t num_current_point_cloud_{0};
  bool updated{false};
  bool has_tracked_frame_widget{false};
};

} // namespace simple_vo
} // namespace slam
} // namespace my3d

#endif // _MY3D_SLAM_SIMPLE_VO_VISUALIZATION_H_