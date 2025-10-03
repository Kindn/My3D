/*
 * filename: visualization.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "visualization.h"

namespace my3d {
namespace slam {
namespace simple_vo {

void EstimatorVisualizer::run() {
  thread_viz_ = std::thread(&EstimatorVisualizer::threadVisualization, this);
  // thread_guard_viz_ = std::make_unique<util::ThreadGuard>(thread_viz_);
}

void EstimatorVisualizer::threadVisualization() {
  window_.setBackgroundColor(cv::viz::Color::black());
  window_.showWidget("World Frame", cv::viz::WCoordinateSystem());

  while (!window_.wasStopped()) {
    if (!updated) {
      window_.spinOnce(1, true);
      continue;
    }
    /* Show cameras */
    char camera_name[1000];
    for (size_t i = 0; i < num_current_cameras_; ++i) {
      sprintf(camera_name, "keyframe_%ld", i);
      window_.removeWidget(camera_name);
    }
    num_current_cameras_ = 0;
    for (size_t i = 0; i < viz_data_.sliding_window.size(); ++i) {
      const cv::Vec2f fov(1.0f, 1.0f);
      sprintf(camera_name, "keyframe_%ld", i);
      const cv::Affine3f pose(viz_data_.sliding_window[i].R,
                              viz_data_.sliding_window[i].t);
      const cv::viz::WCameraPosition w_camera_postition{
          fov, 0.8,
          cv::viz::Color(viz_data_.sliding_window[i].color[0],
                         viz_data_.sliding_window[i].color[1],
                         viz_data_.sliding_window[i].color[2])};
      window_.showWidget(camera_name, w_camera_postition);
      window_.setWidgetPose(camera_name, pose);
      ++num_current_cameras_;
    }

    {
      const cv::Vec2f fov(1.0f, 1.0f);
      sprintf(camera_name, "traj_frame_%ld", viz_data_.tracked_frame.id);
      const cv::Affine3f pose(viz_data_.tracked_frame.R,
                              viz_data_.tracked_frame.t);
      const cv::viz::WCameraPosition w_camera_postition{
          fov, 0.5,
          cv::viz::Color(viz_data_.tracked_frame.color[0],
                         viz_data_.tracked_frame.color[1],
                         viz_data_.tracked_frame.color[2])};
      window_.showWidget(camera_name, w_camera_postition);
      window_.setWidgetPose(camera_name, pose);

      if (estimator_->regist_report_.success) {
        if (has_tracked_frame_widget) {
          window_.removeWidget("tracked_frame");
        }
        const cv::viz::WCameraPosition w_camera_postition1{
            fov, 0.8,
            cv::viz::Color(viz_data_.tracked_frame.color[0],
                           viz_data_.tracked_frame.color[1],
                           viz_data_.tracked_frame.color[2])};
        window_.showWidget("tracked_frame", w_camera_postition1);
        window_.setWidgetPose("tracked_frame", pose);
        has_tracked_frame_widget = true;
      }
    }

    /* Show point cloud */
    const std::string point_cloud_name = "point_cloud";
    if (num_current_point_cloud_ != 0) {
      window_.removeWidget(point_cloud_name);
      num_current_point_cloud_ = 0;
    }
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> point_colors;
    for (const auto &viz_point_3D : viz_data_.point_cloud) {
      points.push_back(viz_point_3D.position);
      point_colors.push_back(viz_point_3D.color);
    }
    if (!points.empty() && !point_colors.empty()) {
      const cv::viz::WCloud w_cloud(points, point_colors);
      window_.showWidget(point_cloud_name, w_cloud);
      ++num_current_point_cloud_;
    }

    updated = false;

    window_.spinOnce(1, true);
  }

  close();
}

void EstimatorVisualizer::join() { thread_viz_.join(); }

void EstimatorVisualizer::close() { window_.close(); }

void EstimatorVisualizer::getVizData() {
  viz_data_.clear();
  /* Get camera info */
  auto const &sliding_window{estimator_->sliding_window_};
  size_t const window_size{sliding_window.size()};
  for (size_t i{0UL}; i < window_size; ++i) {
    auto const &frame{sliding_window[i]};
    const Eigen::Matrix3d rotation{
        base::quaternion2RotationMatrix(frame.rotation)};
    const Eigen::Vector3d translation{frame.translation};
    cv::Mat R(3, 3, CV_32FC1, cv::Scalar(0));
    cv::Mat t(3, 1, CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R.at<float>(i, j) = static_cast<float>(rotation(i, j));
      }
      t.at<float>(i, 0) = static_cast<float>(translation(i));
    }
    VizData::VizCamera viz_camera;
    viz_camera.id = frame.id;
    viz_camera.frame_idx = i;
    viz_camera.R = R.clone();
    viz_camera.t = t.clone();
    viz_camera.color = getCameraColor(frame.id, i);
    viz_data_.sliding_window.push_back(viz_camera);
  }

  {
    cv::Mat R(3, 3, CV_32FC1, cv::Scalar(0));
    cv::Mat t(3, 1, CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R.at<float>(i, j) = static_cast<float>(
            estimator_->tracking_rotation_.toRotationMatrix().operator()(
                i, j));
      }
      t.at<float>(i, 0) =
          static_cast<float>(estimator_->tracking_translation_(i));
    }
    viz_data_.tracked_frame.id = estimator_->regist_report_.frame_id;
    viz_data_.tracked_frame.R = R.clone();
    viz_data_.tracked_frame.t = t.clone();
    viz_data_.tracked_frame.color = color_tracked_camera;
  }

  /* Get point cloud info */
  for (auto const &[track_id, track] : estimator_->tracks_) {
    const Eigen::Vector3d position{estimator_->getTrackPosition(track_id)};
    VizData::VizPoint3D viz_point_3D;
    viz_point_3D.id = track_id;
    viz_point_3D.position = cv::Point3f{static_cast<float>(position.x()),
                                        static_cast<float>(position.y()),
                                        static_cast<float>(position.z())};
    viz_point_3D.color = getPoint3DColor(track_id);
    viz_data_.point_cloud.push_back(viz_point_3D);
  }
}

cv::Vec3b EstimatorVisualizer::getCameraColor(const size_t &id,
                                              const size_t &idx) const {
  if (id == 0UL) {
    return color_ref_camera_0;
  } else if (id == 1UL) {
    return color_ref_camera_1;
  } else {
    return color_camera;
  }
}

cv::Vec3b EstimatorVisualizer::getPoint3DColor(const size_t &track_id) const {
  if (estimator_->tracks_.find(track_id) != estimator_->tracks_.end()) {
    auto const &track{estimator_->tracks_.at(track_id)};
    if (!track->valid) {
      return {255, 0, 0};
    } else {
      double const a{static_cast<double>(track->observations.size()) /
                     (estimator_->config_.window_size + 1UL)};
      return {0, util::clamp<uint8_t>(a * 255, 0, 255),
              util::clamp<uint8_t>((1.0 - a) * 255, 0, 255)};
    }
  } else if (estimator_->removed_tracks_.find(track_id) !=
             estimator_->removed_tracks_.end()) {
    return {128, 128, 128};
  } else if (estimator_->stored_tracks_.find(track_id) !=
             estimator_->stored_tracks_.end()) {
    return {100, 150, 180};
  }

  return {0, 0, 0};
}

} // namespace simple_vo
} // namespace slam
} // namespace my3d