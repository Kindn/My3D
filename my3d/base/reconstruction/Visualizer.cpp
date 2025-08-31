/*
 * filename: Visualizer.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "base/reconstruction/Visualizer.h"

namespace my3d {
namespace base {

void Visualizer::run() {
  thread_viz_ = std::thread(&Visualizer::threadVisualization, this);
  // thread_guard_viz_ = std::make_unique<util::ThreadGuard>(thread_viz_);
}

void Visualizer::threadVisualization() {
  window_.setBackgroundColor(cv::viz::Color::black());
  window_.showWidget("World Frame", cv::viz::WCoordinateSystem());

  while (!window_.wasStopped()) {
    /* Get data from reconstruction_ */
    getVizData();

    /* Show cameras */
    char camera_name[1000];
    for (size_t i = 0; i < num_current_cameras_; ++i) {
      sprintf(camera_name, "camera_%ld", i);
      window_.removeWidget(camera_name);
    }
    num_current_cameras_ = 0;
    for (size_t i = 0; i < viz_data_.cameras.size(); ++i) {
      const cv::Vec2f fov(1.0f, 1.0f);
      sprintf(camera_name, "camera_%ld", i);
      const cv::Affine3f pose(viz_data_.cameras[i].R, viz_data_.cameras[i].t);
      const cv::viz::WCameraPosition w_camera_postition{
          fov, 0.1,
          cv::viz::Color(viz_data_.cameras[i].color[0],
                         viz_data_.cameras[i].color[1],
                         viz_data_.cameras[i].color[2])};
      window_.showWidget(camera_name, w_camera_postition);
      window_.setWidgetPose(camera_name, pose);
      ++num_current_cameras_;
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

    window_.spinOnce(1000, true);
  }

  close();
}

void Visualizer::join() { thread_viz_.join(); }

void Visualizer::close() { window_.close(); }

void Visualizer::getVizData() {
  viz_data_.clear();
  /* Get camera info */
  const auto camera_ids = reconstruction_->getRegisteredImageIDs();
  const Scene *scene = reconstruction_->getScene();
  for (const size_t &camera_id : camera_ids) {
    if (!scene->existsNode(camera_id)) {
      continue;
    }
    if (reconstruction_->isImageFiltered(camera_id)) {
      continue;
    }

    const Eigen::Matrix3d rotation =
        base::quaternion2RotationMatrix(scene->getNodeRotation(camera_id));
    const Eigen::Vector3d translation = scene->getNodeTranslation(camera_id);
    cv::Mat R(3, 3, CV_32FC1, cv::Scalar(0));
    cv::Mat t(3, 1, CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R.at<float>(i, j) = static_cast<float>(rotation(i, j));
      }
      t.at<float>(i, 0) = static_cast<float>(translation(i));
    }
    VizData::VizCamera viz_camera;
    viz_camera.id = camera_id;
    viz_camera.R = R.clone();
    viz_camera.t = t.clone();
    viz_camera.color = getCameraColor(camera_id);
    viz_data_.cameras.push_back(viz_camera);
  }

  /* Get point cloud info */
  EigenVec<Track> tracks;
  for (const auto id_track : reconstruction_->getTracks()) {
    const Track *track = reconstruction_->getTrack(id_track.first);
    if (track != nullptr) {
      tracks.push_back(*track);
    }
  }
  for (const auto track : tracks) {
    const size_t track_id = track.getId();
    const Eigen::Vector3d position = track.get3DPosition();
    VizData::VizPoint3D viz_point_3D;
    viz_point_3D.id = track_id;
    viz_point_3D.position = cv::Point3f{static_cast<float>(position.x()),
                                        static_cast<float>(position.y()),
                                        static_cast<float>(position.z())};
    viz_point_3D.color = getPoint3DColor(track_id);
    viz_data_.point_cloud.push_back(viz_point_3D);
  }
}

cv::Vec3b Visualizer::getCameraColor(const size_t &camera_id) const {
  if (!reconstruction_->existsImage(camera_id)) {
    return cv::Vec3b{0, 0, 0};
  } else if (reconstruction_->isImageNew(camera_id)) {
    return color_new_camera;
  } else if (camera_id == reconstruction_->getRefImageID0()) {
    return color_ref_camera_0;
  } else if (camera_id == reconstruction_->getRefImageID1()) {
    return color_ref_camera_1;
  } else {
    return color_camera;
  }
}

cv::Vec3b Visualizer::getPoint3DColor(const size_t &point_id) const {
  const Track *track = reconstruction_->getTrack(point_id);
  if (track == nullptr) {
    return cv::Vec3b{0, 0, 0};
  }

  if (reconstruction_->isTrackModified(point_id)) {
    return color_modified_point;
  } else if (reconstruction_->isTrackNew(point_id)) {
    return color_new_point;
  } else {
    const Eigen::Matrix<uint8_t, 3, 1> color =
        reconstruction_->getScene()->computeTrackColor(*track);
    return cv::Vec3b{color[2], color[1], color[0]};
  }
}

} // namespace base
} // namespace my3d
