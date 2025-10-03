/*
 * filename: Estimator.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    estimator for SimpleVO
 */

#include "Estimator.h"

namespace my3d {
namespace slam {
namespace simple_vo {

Estimator::Estimator(Config const &config) noexcept
    : config_{config}, is_initialized_{false}, is_ref_frame_selected_{false} {
  assert(config_.Check());

  feature_tracker_ =
      std::make_shared<FeatureTracker>(config_.feature_tracker_config);
  sliding_window_.reserve(config_.window_size + 1UL);
}

void Estimator::reset() noexcept {
  tracks_.clear();
  initial_tracks_.clear();
  removed_tracks_.clear();
  stored_tracks_.clear();
  sliding_window_.clear();
  sliding_window_.reserve(config_.window_size + 1UL);
  if (feature_tracker_ == nullptr) {
    feature_tracker_ =
        std::make_unique<FeatureTracker>(config_.feature_tracker_config);
  } else {
    feature_tracker_->reset();
  }
  is_initialized_ = false;
  is_ref_frame_selected_ = false;
}

void Estimator::feedFrame(std::shared_ptr<base::Image> const &image,
                          double const timestamp) noexcept {
  //* Initialization: create reference frame
  if (!is_initialized_ && !is_ref_frame_selected_) {
    size_t const kMinNumCornersInRefFrame{static_cast<size_t>(
        config_.feature_tracker_config.max_num_corners * 0.7)};
    auto const initial_corners{
        feature_tracker_->extractAdditionalCorners(image, {})};
    size_t const num_initial_corners{initial_corners.size()};
    if (num_initial_corners < kMinNumCornersInRefFrame) {
      std::cout << "[WARNING] Too few corners. Skipped candidate. num_corners: "
                << num_initial_corners << std::endl;
      feature_tracker_->reset();

      return;
    } else {
      // createReferenceFrame(initial_corners, timestamp);
      //* Create initial tracks
      createInitialTracks(initial_corners);
      std::cout << "[INFO] Created " << initial_tracks_.size()
                << " initial tracks. " << std::endl;
      is_ref_frame_selected_ = true;
      updateFrameInfo(image, timestamp, false);

      return;
    }
  }

  tracked_pts_ = trackFeatures(image, timestamp);
  std::cout << "[INFO] Got "
            << tracked_pts_.tracked_pts.size() +
                   tracked_pts_.tracked_add_pts.size()
            << " tracked points. "
            << "existing: " << tracked_pts_.tracked_pts.size() << ", "
            << "additional: " << tracked_pts_.tracked_add_pts.size()
            << std::endl;
  if (!tracked_pts_.success) {
    std::cout << "[ERROR] Feature tracking failed. Skip this frame. "
              << std::endl;

    return;
  }

  //* Initialization: select the 2nd keyframe and do two-view-geometry
  //* initialization
  if (!is_initialized_ && is_ref_frame_selected_) {
    // Track features
    bool const init_succ{initialize(tracked_pts_, timestamp)};
    if (init_succ) {
      //* Initial bundle adjustment. Fix pose of reference frame and translation
      //* of the 2nd frame
      is_initialized_ = true;
      optimize();
      filterTracks();
      updateFrameInfo(image, timestamp, true);
      std::cout << "[INFO] System initialized. " << std::endl;
      return;
    } else {
      std::cout << "[ERROR] Initialization failed. Waiting for next frame to "
                   "try again. "
                << std::endl;
      feature_tracker_->reset();
      return;
    }
  }

  //* Compute initial pose for the incoming frame and add it to sliding window
  regist_report_ = registerFrame(tracked_pts_, timestamp);
  if (!regist_report_.success) {
    std::cout << "[ERROR] Frame registration failed. Skip this frame. "
              << std::endl;
    feature_tracker_->reset();
    return;
  }
  std::cout << "[INFO] is_keyframe: " << regist_report_.is_keyframe
            << std::endl;

  //* Sliding window optimization
  // if (regist_report_.is_keyframe) {
  optimize();
  filterTracks();
  updateFrameInfo(image, timestamp, false);
  // }

  //* If the sliding window is full, do marginalization
  MarginType margin_type{MarginType::NONE};
  if (!regist_report_.is_keyframe &&
      sliding_window_.size() <= config_.window_size) {
    // margin_type = MarginType::MARGIN_SECOND_NEWEST;
    // std::cout << "[ERROR] margin_type: "
    //           << "MARGIN_SECOND_NEWEST" << std::endl;
    margin_type = MarginType::MARGIN_NEWEST;
    std::cout << "[ERROR] margin_type: "
              << "MARGIN_NEWEST" << std::endl;
  } else if (!regist_report_.is_keyframe &&
             sliding_window_.size() > config_.window_size) {
    // margin_type = MarginType::MARGIN_SECOND_NEWEST;
    // std::cout << "[ERROR] margin_type: "
    //           << "MARGIN_SECOND_NEWEST" << std::endl;
    margin_type = MarginType::MARGIN_NEWEST;
    std::cout << "[ERROR] margin_type: "
              << "MARGIN_NEWEST" << std::endl;
  } else if (sliding_window_.size() > config_.window_size) {
    margin_type = MarginType::MARGIN_OLDEST;
    std::cout << "[ERROR] margin_type: "
              << "MARGIN_OLDEST" << std::endl;
  } else {
    std::cout << "[ERROR] margin_type: "
              << "NONE" << std::endl;
  }

  marginalize(margin_type);

  //* Slide window
  slideWindow(margin_type);
}

void Estimator::drawTrackingResult(cv::Mat &image) const noexcept {
  for (auto const &[track_id, tracked_pt] : tracked_pts_.tracked_pts) {
    if (!tracked_pt.valid) {
      continue;
    }

    cv::Scalar color(0, 0, 255);
    if (is_initialized_) {
      if (tracks_.find(track_id) != tracks_.end()) {
        auto const &track{tracks_.at(track_id)};
        double const a{static_cast<double>(track->observations.size()) /
                       (config_.window_size + 1UL)};
        color = cv::Scalar(0, util::clamp<int32_t>(a * 255, 0, 255),
                           util::clamp<int32_t>((1.0 - a) * 255, 0, 255));
      }
    } else {
      if (initial_tracks_.find(track_id) != initial_tracks_.end()) {
        auto const &track{initial_tracks_.at(track_id)};
        double const a{static_cast<double>(track->observations.size()) /
                       (config_.window_size + 1UL)};
        color = cv::Scalar(0, util::clamp<int32_t>(a * 255, 0, 255),
                           util::clamp<int32_t>((1.0 - a) * 255, 0, 255));
      }
    }
    cv::circle(image,
               cv::Point(static_cast<int32_t>(tracked_pt.pixel_coord.x()),
                         static_cast<int32_t>(tracked_pt.pixel_coord.y())),
               2, color, -1);
    cv::line(image,
             cv::Point(static_cast<int32_t>(tracked_pt.pixel_coord_src.x()),
                       static_cast<int32_t>(tracked_pt.pixel_coord_src.y())),
             cv::Point(static_cast<int32_t>(tracked_pt.pixel_coord.x()),
                       static_cast<int32_t>(tracked_pt.pixel_coord.y())),
             color, 1);
  }

  for (auto const &tracked_add_pt : tracked_pts_.tracked_add_pts) {
    cv::Scalar const color(0, 0, 255);
    cv::circle(image,
               cv::Point(static_cast<int32_t>(tracked_add_pt.pixel_coord.x()),
                         static_cast<int32_t>(tracked_add_pt.pixel_coord.y())),
               2, color, -1);
    cv::line(
        image,
        cv::Point(static_cast<int32_t>(tracked_add_pt.pixel_coord_src.x()),
                  static_cast<int32_t>(tracked_add_pt.pixel_coord_src.y())),
        cv::Point(static_cast<int32_t>(tracked_add_pt.pixel_coord.x()),
                  static_cast<int32_t>(tracked_add_pt.pixel_coord.y())),
        color, 1);
  }
}

void Estimator::updateFrameInfo(std::shared_ptr<base::Image> const &image,
                                double const timestamp,
                                bool const init_flag) noexcept {
  if (!is_ref_frame_selected_) {
    return;
  } else if (!is_initialized_) {
    tracking_rotation_ = Eigen::Quaterniond::Identity();
    tracking_translation_ = Eigen::Vector3d::Zero();
    last_frame_info_.last_tracking_rotation = tracking_rotation_;
    last_frame_info_.last_tracking_translation = tracking_translation_;
    last_frame_info_.last_image = image;
    last_frame_info_.last_timestamp = timestamp;
    last_frame_info_.last_pts.clear();
    last_frame_info_.last_pts.reserve(initial_tracks_.size());
    for (auto const &[track_id, track] : initial_tracks_) {
      FeatureTracker::InputPoint ft_in{};
      ft_in.id = track_id;
      ft_in.pos = track->observations[0].pixel_coord;
      ft_in.track_len = 1UL;
      last_frame_info_.last_pts.emplace(track_id, ft_in);
    }
  } else if (init_flag) {
    tracking_rotation_ = sliding_window_.back().rotation;
    tracking_translation_ = sliding_window_.back().translation;
    last_frame_info_.last_tracking_rotation = tracking_rotation_;
    last_frame_info_.last_tracking_translation = tracking_translation_;
    last_frame_info_.last_image = image;
    last_frame_info_.last_timestamp = timestamp;
    last_frame_info_.last_pts.clear();
    last_frame_info_.last_pts.reserve(tracks_.size());
    for (auto const &[track_id, track] : tracks_) {
      FeatureTracker::InputPoint ft_in{};
      ft_in.id = track_id;
      ft_in.pos = track->observations[1].pixel_coord;
      ft_in.track_len = 2UL;
      last_frame_info_.last_pts.emplace(track_id, ft_in);
    }
  } else {
    tracking_rotation_ = sliding_window_.back().rotation;
    tracking_translation_ = sliding_window_.back().translation;
    last_frame_info_.last_tracking_rotation = tracking_rotation_;
    last_frame_info_.last_tracking_translation = tracking_translation_;
    last_frame_info_.last_image = image;
    last_frame_info_.last_timestamp = timestamp;
    EigenUMap<size_t, FeatureTracker::InputPoint> tmp_last_pts{};
    for (auto const &[track_id, obs] : sliding_window_.back().features) {
      if (tracks_.find(track_id) == tracks_.end()) {
        continue;
      }
      FeatureTracker::InputPoint ft_in{};
      ft_in.id = track_id;
      ft_in.pos = obs.pixel_coord;
      if (last_frame_info_.last_pts.find(track_id) !=
          last_frame_info_.last_pts.end()) {
        //* existing track
        ft_in.track_len =
            last_frame_info_.last_pts.at(track_id).track_len + 1UL;
      } else {
        //* newly added track
        ft_in.track_len = 2UL;
      }
      tmp_last_pts.emplace(track_id, ft_in);
    }
    last_frame_info_.last_pts = tmp_last_pts;
  }
}

void Estimator::createInitialTracks(
    EigenVec<Eigen::Vector2d> const &initial_corners) noexcept {
  size_t const num_initial_corners{initial_corners.size()};
  initial_tracks_.clear();
  initial_tracks_.reserve(num_initial_corners);
  auto const &camera{config_.feature_tracker_config.camera};
  last_frame_info_.last_pts.clear();
  for (size_t i{0UL}; i < num_initial_corners; ++i) {
    Track track{};
    track.id = i;
    track.start_idx = 0UL;
    track.depth = 1.0;
    track.observations.emplace_back(initial_corners[i],
                                    camera->pix2Sphere(initial_corners[i]));
    track.is_triangulated = false;
    track.valid = false;
    initial_tracks_.emplace(i, std::make_shared<Track>(track));
  }
}

FeatureTracker::Result
Estimator::trackFeatures(std::shared_ptr<base::Image> const &image,
                         double const timestamp) noexcept {
  if (!is_ref_frame_selected_) {
    std::cout << "[WARNING] Reference frame has not been selected. No feature "
                 "will be tracked. "
              << std::endl;

    return {};
  }

  double const dt{timestamp - last_frame_info_.last_timestamp};
  // if (!is_initialized_) {
  //   pts.reserve(initial_tracks_.size());
  //   for (auto const &[track_id, track] : initial_tracks_) {
  //     FeatureTracker::InputPoint pt{};
  //     pt.id = track_id;
  //     pt.pos = track->observations[0].pixel_coord;
  //     pt.track_len = 1UL;
  //     pts.emplace(track_id, pt);
  //   }
  // } else {
  //   auto const features{sliding_window_.back().features};
  //   pts.reserve(features.size());
  //   for (auto const &[track_id, observation] : features) {
  //     if (tracks_.find(track_id) == tracks_.end()) {
  //       continue;
  //     }

  //     auto const track{tracks_.at(track_id)};
  //     if (!track->valid) {
  //       continue;
  //     }

  //     FeatureTracker::InputPoint pt{};
  //     pt.id = track_id;
  //     pt.pos = observation.pixel_coord;
  //     pt.track_len = track->observations.size();
  //     pts.emplace(track_id, pt);
  //   }
  // }

  return feature_tracker_->track(last_frame_info_.last_image, image,
                                 last_frame_info_.last_pts, dt);
}

bool Estimator::initialize(FeatureTracker::Result const &ft_result,
                           double const timestamp) noexcept {
  if (!ft_result.success) {
    std::cout << "[ERROR] Invalid tracking result for initialization. "
              << std::endl;
    return false;
  }

  if (is_initialized_) {
    std::cout << "[ERROR] System has been initialized. " << std::endl;
    return false;
  }

  if (!is_ref_frame_selected_) {
    std::cout << "[ERROR] Reference frame has not been selected. " << std::endl;
    return false;
  }

  size_t constexpr kMinNumInitMatches{30UL};
  size_t const num_tracked_existing_pts{ft_result.tracked_pts.size()};
  size_t const num_tracked_add_pts{ft_result.tracked_add_pts.size()};
  size_t const num_tracked_pts{num_tracked_existing_pts + num_tracked_add_pts};
  if (num_tracked_pts < kMinNumInitMatches) {
    std::cout << "[ERROR] Too few matches for initialization. Total match num: "
              << num_tracked_pts << std::endl;
    return false;
  }

  //* Extract valid matches
  // TODO Support wide-angle cameras (only for perspective cameras now)
  auto const &camera{config_.feature_tracker_config.camera};
  EigenVec<Eigen::Vector2d> pts0{};
  EigenVec<Eigen::Vector2d> pts1{};
  pts0.reserve(num_tracked_pts);
  pts1.reserve(num_tracked_pts);
  double constexpr kZEps{1.0e-6};
  double avg_par{0.0};
  for (auto const &[id, tp] : ft_result.tracked_pts) {
    if (!tp.valid) {
      continue;
    }

    Eigen::Vector3d const pt0{camera->pix2Sphere(tp.pixel_coord_src)};
    Eigen::Vector3d const pt1{tp.sphere_coord};
    if (std::abs(pt0.z()) < kZEps || std::abs(pt1.z()) < kZEps) {
      continue;
    }
    pts0.emplace_back(pt0.hnormalized());
    pts1.emplace_back(pt1.hnormalized());
    avg_par += std::acos(pt0.dot(pt1));
  }
  size_t const num_valid_pts{pts0.size()};
  std::cout << "[INFO] Extracted " << num_valid_pts
            << " valid points for initialization. " << std::endl;
  if (num_valid_pts < kMinNumInitMatches) {
    std::cout << "[ERROR] Too few valid matches for initialization. Total "
                 "valid match num: "
              << num_valid_pts << std::endl;
    return false;
  }

  //* Check average parallax
  avg_par /= static_cast<double>(num_valid_pts);
  if (avg_par < util::deg2Rad(config_.min_avg_parallax_init_deg)) {
    std::cout << "[ERROR] Insufficient average parallax for initialization. "
              << "avg par: " << util::rad2Deg(avg_par) << "deg(s), "
              << config_.min_avg_parallax_init_deg << "deg(s) required. "
              << std::endl;
    return false;
  }

  //* Estimate two-view geometry
  typedef estimator::LORANSAC<estimator::HomographyEstimator,
                              estimator::HomographyEstimator>
      HomographyEstimator;
  typedef estimator::LORANSAC<estimator::EssentialMatrixFivePointEstimator,
                              estimator::EssentialMatrixFivePointEstimator>
      EssentialEstimator5Pt;
  typedef estimator::LORANSAC<estimator::EssentialMatrixEightPointEstimator,
                              estimator::EssentialMatrixEightPointEstimator>
      EssentialEstimator8Pt;
  HomographyEstimator h_estimator{config_.init_h_ransac_config};
  EssentialEstimator5Pt e_estimator_5pt{config_.init_e_ransac_config};
  EssentialEstimator8Pt e_estimator_8pt{config_.init_e_ransac_config};
  auto const h_report{h_estimator.estimate(pts0, pts1)};
  auto const e_report_5pt{e_estimator_5pt.estimate(pts0, pts1)};
  auto const e_report_8pt{e_estimator_8pt.estimate(pts0, pts1)};
  if (!h_report.success || !e_report_5pt.success || !e_report_8pt.success) {
    std::cout << "[ERROR] Two-view geometry estimation failed. " << std::endl;
    return false;
  }
  struct EReport {
    Eigen::Matrix3d model{};
    size_t num_inliers{0UL};
    std::vector<bool> inlier_mask{};
  };
  EReport e_report{};
  if (e_report_5pt.num_inliers > e_report_8pt.num_inliers) {
    e_report.model = e_report_5pt.model;
    e_report.num_inliers = e_report_5pt.num_inliers;
    e_report.inlier_mask = e_report_5pt.inlier_mask;
  } else {
    e_report.model = e_report_8pt.model;
    e_report.num_inliers = e_report_8pt.num_inliers;
    e_report.inlier_mask = e_report_8pt.inlier_mask;
  }
  double const he_inlier_ratio{static_cast<double>(h_report.num_inliers) /
                               e_report.num_inliers};
  EigenVec<Eigen::Vector2d> inlier_pts0{};
  EigenVec<Eigen::Vector2d> inlier_pts1{};
  Eigen::Matrix3d init_rot{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d init_trans{Eigen::Vector3d::Zero()};
  if (he_inlier_ratio <= config_.max_h_inlier_ratio) {
    double const inlier_ratio{static_cast<double>(e_report.num_inliers) /
                              num_valid_pts};
    if (inlier_ratio < config_.min_init_two_view_geom_inlier_ratio) {
      std::cout << "[ERROR] Insufficient two-view geometry inlier ratio "
                << inlier_ratio << " for initialization. "
                << config_.min_init_two_view_geom_inlier_ratio << " required. "
                << std::endl;
      return false;
    }

    std::cout << "[ERROR] Using essential matrix model. inlier ratio: "
              << inlier_ratio << std::endl;

    inlier_pts0.reserve(e_report.num_inliers);
    inlier_pts1.reserve(e_report.num_inliers);
    for (size_t i{0UL}; i < num_valid_pts; ++i) {
      if (e_report.inlier_mask[i]) {
        inlier_pts0.emplace_back(pts0[i]);
        inlier_pts1.emplace_back(pts1[i]);
      }
    }
    EigenVec<Eigen::Vector3d> tri_pts{};
    estimator::poseFromEssentialMatrix(e_report.model, inlier_pts0, inlier_pts1,
                                       init_rot, init_trans, tri_pts);
  } else {
    double const inlier_ratio{static_cast<double>(h_report.num_inliers) /
                              num_valid_pts};
    if (inlier_ratio < config_.min_init_two_view_geom_inlier_ratio) {
      std::cout << "[ERROR] Insufficient two-view geometry inlier ratio "
                << inlier_ratio << " for initialization. "
                << config_.min_init_two_view_geom_inlier_ratio << " required. "
                << std::endl;
      return false;
    }

    std::cout << "[ERROR] Using homography model. inlier ratio: "
              << inlier_ratio << std::endl;

    inlier_pts0.reserve(h_report.num_inliers);
    inlier_pts1.reserve(h_report.num_inliers);
    for (size_t i{0UL}; i < num_valid_pts; ++i) {
      if (h_report.inlier_mask[i]) {
        inlier_pts0.emplace_back(pts0[i]);
        inlier_pts1.emplace_back(pts1[i]);
      }
    }
    EigenVec<Eigen::Vector3d> tri_pts{};
    Eigen::Vector3d normal{};
    Eigen::Matrix3d const cam_mat{Eigen::Matrix3d::Identity()};
    estimator::poseFromHomography(h_report.model, cam_mat, cam_mat, inlier_pts0,
                                  inlier_pts1, init_rot, init_trans, normal,
                                  tri_pts);
  }
  init_trans.normalize();
  init_trans *= config_.dist_init_pair;

  //* Triangulate points and check cheirality (for perspective cameras)
  size_t const num_inliers{inlier_pts0.size()};
  std::vector<bool> cheirality_mask(num_inliers, true);
  double const kMinDepth{1.0e-3};
  double const kMaxDepth{1.0e3 * init_trans.norm()};
  Eigen::Matrix3x4d const proj0{Eigen::Matrix3x4d::Identity()};
  Eigen::Matrix3x4d const proj1{base::composePose(init_rot, init_trans)};
  std::vector<double> depths_in_start_frame(num_inliers);
  size_t visible_count{0UL};
  for (size_t i{0UL}; i < num_inliers; ++i) {
    Eigen::Vector3d const pw{
        base::triangulatePoint(proj0, proj1, inlier_pts0[i], inlier_pts1[i])};
    //* For general use, the distance from the point to the optical center is
    //* used as the actual depth for application
    depths_in_start_frame[i] = pw.norm();
    double const depth0{pw.z()};
    double const depth1{(init_rot * pw + init_trans).z()};
    if (depth0 >= kMinDepth && depth0 <= kMaxDepth && depth1 >= kMinDepth &&
        depth1 <= kMaxDepth) {
      cheirality_mask[i] = true;
      ++visible_count;
    } else {
      cheirality_mask[i] = false;
    }
  }
  if (visible_count < config_.min_num_init_tracks) {
    std::cout << "[ERROR] Insufficient number of tracks for initialization. "
                 "track_num: "
              << visible_count << ", " << config_.min_num_init_tracks
              << " required. " << std::endl;
    return false;
  }

  //* Create tracks and frames (only when report.success == true)
  tracks_.clear();
  tracks_.reserve(num_inliers);
  for (size_t i{0UL}; i < num_inliers; ++i) {
    if (!cheirality_mask[i]) {
      continue;
    }

    Track track{};
    track.id = i;
    track.start_idx = 0UL;
    track.depth = depths_in_start_frame[i];
    track.observations.clear();
    track.observations.emplace_back(
        camera->project(inlier_pts0[i].homogeneous()),
        inlier_pts0[i].homogeneous().normalized());
    track.observations.emplace_back(
        camera->project(inlier_pts1[i].homogeneous()),
        inlier_pts1[i].homogeneous().normalized());
    track.position =
        depths_in_start_frame[i] * track.observations.front().sphere_coord;
    track.is_triangulated = true;
    track.valid = true;
    tracks_.emplace(i, std::make_shared<Track>(track));
  }
  track_cnt_ += tracks_.size();
  std::cout << "[INFO] Created " << tracks_.size()
            << " tracks in initialization. " << std::endl;

  sliding_window_.clear();
  sliding_window_.resize(2UL);
  sliding_window_[0].id = 0UL;
  sliding_window_[0].timestamp = last_frame_info_.last_timestamp;
  sliding_window_[0].rotation.setIdentity();
  sliding_window_[0].translation.setZero();
  sliding_window_[1].id = 1UL;
  sliding_window_[1].timestamp = timestamp;
  sliding_window_[1].rotation = init_rot.inverse();
  sliding_window_[1].translation = -init_rot.inverse() * init_trans;
  frame_cnt_ += 2UL;
  sliding_window_[0].features.clear();
  sliding_window_[0].features.reserve(num_inliers);
  sliding_window_[1].features.clear();
  sliding_window_[1].features.reserve(num_inliers);
  for (auto const &[track_id, track] : tracks_) {
    sliding_window_[0].features.emplace(track_id, track->observations[0]);
    sliding_window_[1].features.emplace(track_id, track->observations[1]);
  }

  regist_report_.frame_id = 1UL;
  regist_report_.is_keyframe = true;
  regist_report_.rotation = sliding_window_[1].rotation;
  regist_report_.translation = sliding_window_[1].translation;
  regist_report_.success = true;

  return true;
}

void Estimator::optimize() noexcept {
  if (!is_initialized_) {
    std::cout << "[ERROR] System has not been initialized"
                 "Optimization should only be applied after successful "
                 "initialization. "
              << std::endl;
    return;
  }

  size_t const window_size{sliding_window_.size()};
  size_t const min_num_frames_opt{std::max(window_size, 2UL)};
  if (window_size < config_.min_num_frames_opt) {
    std::cout << "[ERROR] Insufficient frames for optmization. window_size: "
              << window_size << ", " << min_num_frames_opt
              << " frames required. " << std::endl;
    return;
  }

  //* Initialize factor graph
  gopt::OptSolverBase *solver{new gopt::LevenbergMarquartSparseSchurSolver};
  // gopt::OptSolverBase *solver{new gopt::GaussNewtonSparseSchurSolver};
  gopt::FactorGraph graph;
  graph.setOptSolver(solver);
  gopt::optimization_config_t config;
  config.verbose = false;
  config.max_iteration_num = 10UL;
  graph.setOptConfig(config);

  //* Normalize the translations of all the frames in sliding window
  Eigen::Vector3d norm_t{Eigen::Vector3d::Zero()};
  double norm_s{1.0};
  normalizeFrameTranslations(norm_t, norm_s);

  //* Add frame vertices
  size_t vid{0UL};
  std::vector<std::shared_ptr<gopt::VertexCameraPose>> vertices_frame{};
  vertices_frame.reserve(sliding_window_.size());
  for (size_t i{0UL}; i < window_size; ++i) {
    Frame const &frame{sliding_window_[i]};
    auto v{std::make_shared<gopt::VertexCameraPose>()};
    v->setId(vid);
    v->setEstimate(gopt::CameraPose(frame.rotation,
                                    norm_s * (frame.translation + norm_t)));
    v->frame_idx = i;
    if (frame.id == 0UL) {
      v->fix_rot = true;
      v->fix_trans = true;
    } else if (frame.id == 1UL) {
      v->fix_rot = false;
      v->fix_trans = true;
    } else {
      v->fix_rot = false;
      v->fix_trans = false;
    }
    if (!graph.addVertex(v)) {
      std::cout << "[ERROR] Failed to add frame vertex for frame " << i
                << " in sliding window. "
                << "Optimization aborted. " << std::endl;
      return;
    }
    vertices_frame.emplace_back(v);
    ++vid;
  }

  //* Add point vertices
  std::unordered_map<size_t, std::shared_ptr<gopt::VertexTrackInvDepth>>
      vertices_track{};
  vertices_track.reserve(config_.feature_tracker_config.max_num_corners);
  double constexpr kMinConsideredTrackDepth{5.0e-2};
  // double constexpr kMaxConsideredTrackDepth{5.0e2};
  for (auto const &[track_id, track] : tracks_) {
    if (!track->valid || !track->is_triangulated) {
      std::cout << "line " << __LINE__ << std::endl;
      continue;
    }

    if (track->depth < kMinConsideredTrackDepth) {
      std::cout << "line " << __LINE__ << std::endl;
      continue;
    }

    if (track->observations.size() < std::min(window_size, 4UL)) {
      std::cout << "line " << __LINE__ << std::endl;
      continue;
    }

    if (track->start_idx > window_size - 2UL) {
      std::cout << "line " << __LINE__ << std::endl;
      continue;
    }

    auto v{std::make_shared<gopt::VertexTrackInvDepth>()};
    v->setId(vid);
    v->setEstimate(1.0 / (track->depth * norm_s));
    v->setMarginalized(true);
    // if (!regist_report_.is_keyframe)
    // {
    //   v->is_fixed = true;
    // } else {
    //   v->is_fixed = false;
    // }
    if (!graph.addVertex(v)) {
      std::cout << "[ERROR] Failed to add track vertex for track " << track_id
                << ". "
                << "Optimization will not be applied. " << std::endl;
      return;
    }
    vertices_track.emplace(track_id, v);
    ++vid;
  }

  //* Add projection edges
  std::unordered_map<
      size_t, std::unordered_map<size_t, std::shared_ptr<gopt::EdgeProjection>>>
      edges{};
  size_t eid{0UL};
  for (auto const &[track_id, vertex_track] : vertices_track) {
    auto const &track{tracks_.at(track_id)};
    size_t const start_idx = track->start_idx;
    auto const &vf0{vertices_frame[start_idx]};
    auto const &vt{vertices_track.at(track_id)};
    auto const observations{track->observations};
    Eigen::Vector3d const meas_sphere0{observations.front().sphere_coord};
    if (edges.find(start_idx) == edges.end()) {
      edges.emplace(
          start_idx,
          std::unordered_map<size_t, std::shared_ptr<gopt::EdgeProjection>>());
    }

    for (size_t i{1UL}; i < observations.size(); ++i) {
      size_t const frame_idx{start_idx + i};
      if (frame_idx >= window_size) {
        break;
      }

      auto const vf1{vertices_frame[frame_idx]};
      auto const meas_sphere1{observations[i].sphere_coord};
      gopt::EdgeProjectionMeasurement const measurement{meas_sphere0,
                                                        meas_sphere1};
      auto const e{std::make_shared<gopt::EdgeProjection>(
          eid, vf0, vf1, vt, measurement, Eigen::Matrix2d::Identity(),
          std::make_shared<gopt::HuberLoss>())};
      if (!graph.addEdge(e)) {
        std::cout << "[ERROR] Failed to add projection edge ("
                  << "frame0: " << start_idx << ", "
                  << "frame1: " << frame_idx << ", "
                  << "track_id: " << track_id << "). "
                  << "Optimization will not be applied. " << std::endl;
        return;
      }
      edges.at(start_idx).emplace(frame_idx, e);
      ++eid;
    }
  }
  std::cout << "[INFO][opt] Added " << vertices_frame.size() << " frame(s), "
            << vertices_track.size() << " track(s), " << edges.size()
            << " residual(s), " << std::endl;

  // TODO Add marginalization edges

  //* Solve BA
  int32_t const opt_status{graph.optimize()};
  if (opt_status < 0) {
    std::cout << "[ERROR] Optmization Failed ("
              << "status: " << opt_status << "). "
              << "States will not be updated. " << std::endl;
    return;
  } else {
    //* Update states
    for (auto const &v : vertices_frame) {
      auto const estimate{v->getEstimate()};
      sliding_window_[v->frame_idx].rotation = estimate.rotation;
      sliding_window_[v->frame_idx].translation = estimate.translation / norm_s - norm_t;
    }
    for (auto const &[track_id, v] : vertices_track) {
      double const estimate{v->getEstimate()};
      tracks_.at(track_id)->depth = (1.0 / estimate) / norm_s;
      tracks_.at(track_id)->position = getTrackPosition(track_id);
    }
  }
  std::cout << "[INFO] Optimization succeeded. " << std::endl;
}

void Estimator::filterTracks() noexcept {
  double constexpr kMinValidDepth{5.0e-2};
  double const max_valid_track_avg_proj_error{
      util::deg2Rad(config_.max_valid_track_avg_proj_error_deg)};
  std::unordered_set<size_t> removed_ids{};
  for (auto const &[track_id, track] : tracks_) {
    if (!track->is_triangulated) {
      continue;
    }

    //* Filter invalid tracks
    if (!track->valid) {
      track->valid = false;
      // removed_tracks_.emplace(track_id, track);
      removed_tracks_.insert(std::make_pair(track_id, track));
      removed_ids.emplace(track_id);
      continue;
    }

    //* Filter tracks with invalid depth
    double const depth{track->depth};
    if (depth < kMinValidDepth || std::isinf(depth) || std::isnan(depth)) {
      track->valid = false;
      // removed_tracks_.emplace(track_id, track);
      removed_tracks_.insert(std::make_pair(track_id, track));
      removed_ids.emplace(track_id);
      continue;
    }

    //* Filter tracks with large projection error
    double const avg_proj_error{computeTrackAvgProjAngleError(track_id)};
    if (avg_proj_error > max_valid_track_avg_proj_error) {
      track->valid = false;
      // removed_tracks_.emplace(track_id, track);
      removed_tracks_.insert(std::make_pair(track_id, track));
      removed_ids.emplace(track_id);
      continue;
    }
  }
  std::cout << "[INFO] Filtered " << removed_ids.size() << " tracks. "
            << std::endl;
  for (size_t const &id : removed_ids) {
    tracks_.erase(id);
  }
}

void Estimator::normalizeFrameTranslations(Eigen::Vector3d &trans,
                                           double &scale) const noexcept {
  trans.setZero();
  scale = 1.0;
  if (sliding_window_.empty()) {
    return;
  }

  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  for (auto const &frame : sliding_window_) {
    center += frame.translation;
  }
  center /= static_cast<double>(sliding_window_.size());
  trans = -center;

  return;
}

void Estimator::marginalize(MarginType const &margin_type) noexcept {
  // TODO Construct marginalization prior
}

void Estimator::slideWindow(MarginType const margin_type) noexcept {
  size_t const window_size{sliding_window_.size()};
  if (margin_type == MarginType::NONE) {
    return;
  }

  std::unordered_set<size_t> stored_ids{};
  std::unordered_set<size_t> removed_ids{};
  if (margin_type == MarginType::MARGIN_OLDEST) {
    sliding_window_.erase(sliding_window_.begin());
    for (auto const &[track_id, track] : tracks_) {
      if (!track->valid) {
        removed_tracks_.emplace(track_id, track);
        removed_ids.emplace(track_id);
        continue;
      }
      if (track->start_idx == 0UL) {
        track->observations.erase(track->observations.begin());
      } else if (track->start_idx > 0UL) {
        --track->start_idx;
      }
      if (track->observations.size() < 2UL) {
        stored_tracks_.emplace(track_id, track);
        stored_ids.emplace(track_id);
        continue;
      }
    }
  } else if (margin_type == MarginType::MARGIN_SECOND_NEWEST) {
    Frame const frame{sliding_window_[window_size - 2UL]};
    sliding_window_.erase(sliding_window_.end() - 2);
    for (auto const &[track_id, obs] : frame.features) {
      if (tracks_.find(track_id) == tracks_.end()) {
        continue;
      }
      auto const &track{tracks_.at(track_id)};
      if (!track->valid) {
        continue;
      }
      if (track->start_idx >= window_size - 1UL) {
        --track->start_idx;
        removed_tracks_.emplace(track_id, track);
        removed_ids.emplace(track_id);
        continue;
      }
      size_t const obs_idx{window_size - 2UL - track->start_idx};
      if (obs_idx < track->observations.size()) {
        track->observations.erase(track->observations.begin() + obs_idx);
      }
      if (obs_idx == 0UL) {
        removed_tracks_.emplace(track_id, track);
        removed_ids.emplace(track_id);
        continue;
      }
    }
  } else if (margin_type == MarginType::MARGIN_NEWEST) {
    Frame const frame{sliding_window_[window_size - 1UL]};
    sliding_window_.erase(sliding_window_.end() - 1);
    for (auto const &[track_id, obs] : frame.features) {
      if (tracks_.find(track_id) == tracks_.end()) {
        continue;
      }
      auto const &track{tracks_.at(track_id)};
      if (!track->valid) {
        continue;
      }
      if (track->start_idx >= window_size - 2UL) {
        removed_tracks_.emplace(track_id, track);
        removed_ids.emplace(track_id);
        continue;
      }
      size_t const obs_idx{window_size - 1UL - track->start_idx};
      if (obs_idx < track->observations.size()) {
        track->observations.erase(track->observations.begin() + obs_idx);
      }
    }
  }
  for (size_t const &id : stored_ids) {
    tracks_.erase(id);
  }
  for (size_t const &id : removed_ids) {
    tracks_.erase(id);
  }
}

Estimator::RegistReport
Estimator::registerFrame(FeatureTracker::Result const &tracked_pts,
                         double const timestamp) noexcept {
  RegistReport report{};
  //* Get 3D-2D correspondences
  if (tracked_pts.tracked_pts.size() <
          estimator::AP3PPoseEstimator::kMinNumSamples ||
      tracked_pts.tracked_pts.size() <
          estimator::EPnPPoseEstimator::kMinNumSamples) {
    std::cout << "Two few tracked points for absolute pose estimation. "
              << std::endl;
    report.success = false;
    return report;
  }

  auto const &camera{config_.feature_tracker_config.camera};
  EigenVec<Eigen::Vector2d> pts_2d{};
  EigenVec<Eigen::Vector3d> pts_3d{};
  std::unordered_map<size_t, size_t> track_ids{};
  for (auto const &[track_id, tracked_pt] : tracked_pts.tracked_pts) {
    if (track_ids.find(track_id) != track_ids.end()) {
      continue;
    }
    if (!tracked_pt.valid) {
      continue;
    }
    if (tracks_.find(track_id) == tracks_.end()) {
      // std::cout << "bbb ";
      // if (removed_tracks_.find(track_id) != removed_tracks_.end()) {
      //   std::cout << "1 ";
      // }
      // if (stored_tracks_.find(track_id) != stored_tracks_.end()) {
      //   std::cout << "2 ";
      // }
      // std::cout << std::endl;
      continue;
    }
    auto const &track{tracks_.at(track_id)};
    if (!track->valid || !track->is_triangulated) {
      continue;
    }

    // TODO Support wide-angle camera by using sphere observation
    track_ids.emplace(track_id, pts_2d.size());
    pts_2d.emplace_back(camera->pix2Norm(tracked_pt.pixel_coord));
    Eigen::Quaterniond const rot{sliding_window_[track->start_idx].rotation};
    Eigen::Vector3d const trans{sliding_window_[track->start_idx].translation};
    pts_3d.emplace_back(
        rot * (track->depth * track->observations.front().sphere_coord) +
        trans);
  }
  if (pts_2d.size() < estimator::AP3PPoseEstimator::kMinNumSamples ||
      pts_2d.size() < estimator::EPnPPoseEstimator::kMinNumSamples) {
    std::cout << "[ERROR] Insufficient number of valid 3D-2D correspondences "
                 "for absolute pose estimation. "
              << "Got " << pts_2d.size() << ". " << std::endl;
    report.success = false;
    return report;
  }
  std::cout << "[INFO] Got " << pts_2d.size()
            << " 3D-2D correspondences for abs pose estimation. " << std::endl;

  //* Estimator absolute pose using PnP
  // auto const abs_pose_report{estimateAbsPoseRANSACPnP(pts_3d, pts_2d)};
  auto const abs_pose_report{estimateAbsPoseIterativePnP(
      pts_3d, pts_2d, sliding_window_.back().rotation,
      sliding_window_.back().translation)};
  double const abs_pose_inlier_ratio{
      static_cast<double>(abs_pose_report.num_inliers) / pts_2d.size()};
  if (abs_pose_inlier_ratio < config_.min_abs_pose_inlier_ratio) {
    std::cout << "[ERROR] Insufficient abs pose inlier ratio "
              << abs_pose_inlier_ratio << " for frame registration. "
              << config_.min_abs_pose_inlier_ratio << " required. "
              << std::endl;
    // return false;
  }
  std::cout << "[INFO] Abs pose estimation result: \n"
            << "\ttrans: " << abs_pose_report.translation.transpose() << "\n"
            << "\trot: " << abs_pose_report.rotation.coeffs().transpose()
            << "\n"
            << "\tnum_inliers: " << abs_pose_report.num_inliers << "\n"
            << "\tinlier_ratio: " << abs_pose_inlier_ratio << std::endl;

  //* Update tracks and add new frame to sliding window
  Frame frame{};
  frame.id = frame_cnt_;
  frame.timestamp = timestamp;
  frame.rotation = abs_pose_report.rotation;
  frame.translation = abs_pose_report.translation;
  frame.features.clear();
  for (auto const &[track_id, track] : tracks_) {
    if (track_ids.find(track_id) == track_ids.end()) {
      continue;
    }
    size_t const index{track_ids.at(track_id)};
    if (abs_pose_report.inlier_mask[index]) {
      Observation obs{};
      obs.pixel_coord = tracked_pts.tracked_pts.at(track_id).pixel_coord;
      obs.sphere_coord = tracked_pts.tracked_pts.at(track_id).sphere_coord;
      track->observations.emplace_back(obs);
      frame.features.emplace(track_id, obs);
    } else {
      // // Delete the interrupted track
      // track->valid = false;
    }
  }
  // Add new tracks
  Eigen::Matrix3x4d const proj0{last_frame_info_.getProjectionMatrix()};
  Eigen::Matrix3x4d const proj1{frame.getProjectionMatrix()};
  double const kZEps{3.0e-2};
  for (auto const &pt : tracked_pts.tracked_add_pts) {
    if (!pt.valid) {
      continue;
    }

    size_t const track_id{track_cnt_++};
    //* Try to triangulate the two observation
    Eigen::Vector2d const pn0{camera->pix2Norm(pt.pixel_coord_src)};
    Eigen::Vector2d const pn1{camera->pix2Norm(pt.pixel_coord)};
    Eigen::Vector3d const pw{base::triangulatePoint(proj0, proj1, pn0, pn1)};
    Eigen::Vector3d const pc0{proj0 * pw.homogeneous()};
    Eigen::Vector3d const pc1{proj1 * pw.homogeneous()};
    if (pc0.z() < kZEps || pc1.z() < kZEps) {
      continue;
    }
    auto track{std::make_shared<Track>()};
    track->id = track_id;
    track->start_idx = sliding_window_.size() - 1UL;
    track->depth = pc0.norm();
    track->position = pw;
    track->observations.clear();
    track->observations.emplace_back(pt.pixel_coord_src,
                                     camera->pix2Sphere(pt.pixel_coord_src));
    track->observations.emplace_back(pt.pixel_coord,
                                     camera->pix2Sphere(pt.pixel_coord));
    track->is_triangulated = true;
    track->valid = true;
    tracks_.emplace(track_id, track);
    sliding_window_.back().features.emplace(track_id, track->observations[0]);
    frame.features.emplace(track_id, track->observations[1]);
  }

  sliding_window_.emplace_back(frame);
  frame_cnt_ += 1UL;

  report.success = true;
  report.rotation = sliding_window_.back().rotation;
  report.translation = sliding_window_.back().translation;
  report.is_keyframe = isKeyframe();
  report.frame_id = sliding_window_.back().id;

  return report;
}

bool Estimator::isKeyframe() const {
  size_t const window_size{sliding_window_.size()};
  double parallax{0.0};
  size_t covis_cnt{0UL};
  size_t existing_track_cnt{0UL};
  size_t new_track_cnt{0UL};
  size_t long_track_cnt{0UL};
  auto const &last_pts{last_frame_info_.last_pts};
  for (auto const &[track_id, track] : tracks_) {
    if (!track->valid || !track->is_triangulated) {
      continue;
    }
    if (last_pts.find(track_id) == last_pts.end()) {
      continue;
    }
    if (track->start_idx <= window_size - 2UL &&
        track->start_idx + track->observations.size() >= window_size) {
      size_t const idx0{window_size - 2UL - track->start_idx};
      size_t const idx1{idx0 + 1UL};
      parallax += std::acos(track->observations[idx0].sphere_coord.dot(
          track->observations[idx1].sphere_coord));
      ++covis_cnt;
      if (last_pts.at(track_id).track_len <= 2UL) {
        ++new_track_cnt;
      } else {
        ++existing_track_cnt;
        if (last_pts.at(track_id).track_len >= 4UL) {
          ++long_track_cnt;
        }
      }
    }
  }
  if (covis_cnt > 0UL) {
    parallax /= static_cast<double>(covis_cnt);
  }

  // if (covis_cnt <= config_.max_num_covis_keyframe) {
  if (existing_track_cnt < 20UL || long_track_cnt < 40UL) {
    // if (existing_track_cnt < 20UL || long_track_cnt < 40UL ||
    //     new_track_cnt > existing_track_cnt / 2UL) {
    return true;
  }

  double const min_trans{std::max(1.0e-3, config_.min_frame_rel_trans)};
  if ((sliding_window_[window_size - 1UL].translation -
       sliding_window_[window_size - 2UL].translation)
              .norm() >= min_trans &&
      parallax >= util::deg2Rad(config_.min_avg_parallax_keyframe_deg)) {
    return true;
  }
  // if (parallax >= util::deg2Rad(config_.min_avg_parallax_keyframe_deg)) {
  //   return true;
  // }

  return false;
}

double
Estimator::computeTrackAvgProjAngleError(size_t const track_id) const noexcept {
  auto const &track{tracks_.at(track_id)};
  Frame const &frame0{sliding_window_[track->start_idx]};
  Eigen::Quaterniond const rot0{frame0.rotation};
  Eigen::Vector3d const trans0{frame0.translation};
  Eigen::Vector3d const s0{track->observations[0].sphere_coord};
  double error{0.0};
  for (size_t i{1UL}; i < track->observations.size(); ++i) {
    size_t const frame_idx{i + track->start_idx};
    Eigen::Quaterniond const rot{sliding_window_[frame_idx].rotation};
    Eigen::Vector3d const trans{sliding_window_[frame_idx].translation};
    Eigen::Vector3d const s{
        (rot.inverse() * (rot0 * (track->depth * s0) + trans0 - trans))
            .normalized()};
    error += std::acos(s.dot(track->observations[i].sphere_coord));
  }
  if (track->observations.size() > 1UL) {
    error /= static_cast<double>(track->observations.size() - 1UL);
  }

  return error;
}

Estimator::AbsPoseEstReport Estimator::estimateAbsPoseRANSACPnP(
    EigenVec<Eigen::Vector3d> const &pts_3d,
    EigenVec<Eigen::Vector2d> const &pts_2d) const {
  AbsPoseEstReport report{};

  auto const &camera{config_.feature_tracker_config.camera};
  estimator::RANSACConfig estimator_config{};
  estimator_config.max_inlier_error =
      camera->thresholdPix2Norm(config_.max_abs_pose_reproj_error);
  estimator_config.min_iter_num = 10UL;
  estimator_config.max_iter_num = 100UL;
  estimator_config.min_inlier_ratio = config_.min_abs_pose_inlier_ratio;
  estimator_config.confidence = 0.99999;
  AbsolutePoseEstimator estimator(estimator_config);
  estimator.getEstimator().setEpsilon(1.0e-5);
  estimator.getEstimator().setK(Eigen::Matrix3d::Identity());
  estimator.getLocalEstimator().setK(Eigen::Matrix3d::Identity());
  auto const ransac_report{estimator.estimate(pts_3d, pts_2d)};
  if (!ransac_report.success) {
    std::cout << "[ERROR] Absolute pose estimation failed. " << std::endl;
    report.success = false;
    return report;
  }

  report.success = true;
  report.rotation = ransac_report.model.leftCols<3>();
  report.translation = ransac_report.model.rightCols<1>();
  report.inlier_mask = ransac_report.inlier_mask;
  report.num_inliers = ransac_report.num_inliers;

  return report;
}

Estimator::AbsPoseEstReport Estimator::estimateAbsPoseIterativePnP(
    EigenVec<Eigen::Vector3d> const &pts_3d,
    EigenVec<Eigen::Vector2d> const &pts_2d, Eigen::Quaterniond const &init_rot,
    Eigen::Vector3d const &init_trans) const {
  AbsPoseEstReport report{};

  size_t const num_pts{pts_2d.size()};
  assert(num_pts == pts_3d.size());

  Eigen::Quaterniond rotation{init_rot};
  Eigen::Vector3d translation{init_trans};
  double const kZEps{std::numeric_limits<double>::epsilon()};
  size_t constexpr kMaxNumIteration{10UL};
  double constexpr kDeltaEps{1.0e-6};
  auto const &camera{config_.feature_tracker_config.camera};
  double const kHuberDelta{
      camera->thresholdPix2Norm(config_.max_abs_pose_reproj_error)};
  double const kHuberDelta2{kHuberDelta * kHuberDelta};
  for (size_t iter{0UL}; iter < kMaxNumIteration; ++iter) {
    Eigen::MatrixXd H{Eigen::MatrixXd::Zero(6, 6)};
    Eigen::VectorXd b{Eigen::VectorXd::Zero(6)};
    double error{0.0};
    for (size_t i{0UL}; i < num_pts; ++i) {
      Eigen::Vector3d const tc_p{pts_3d[i] - translation};
      Eigen::Vector3d const pc{rotation.toRotationMatrix().transpose() * tc_p};
      if (std::abs(pc.z()) < kZEps) {
        report.success = false;
        return report;
      }

      Eigen::Vector2d const residual{pc.hnormalized() - pts_2d[i]};
      double const inv_zc{1.0 / pc.z()};
      double const sqr_inv_zc{inv_zc * inv_zc};
      Eigen::Matrix<double, 2, 3> dpcn_dpc{};
      dpcn_dpc << inv_zc, 0.0, -pc.x() * sqr_inv_zc, 0.0, inv_zc,
          -pc.y() * sqr_inv_zc;
      Eigen::Matrix<double, 3, 6> dpc_dpose{};
      dpc_dpose.leftCols<3>() = util::getSkewSymmetric(pc);
      dpc_dpose.rightCols<3>() = -rotation.toRotationMatrix().transpose();
      Eigen::Matrix<double, 2, 6> const jacobian{dpcn_dpc * dpc_dpose};
      double const error2{residual.squaredNorm()};
      double const j_huber{
          (error2 < kHuberDelta2) ? 1.0 : (kHuberDelta / std::sqrt(error2))};
      H += j_huber * jacobian.transpose() * jacobian;
      b += -j_huber * jacobian.transpose() * residual;
    }

    error = std::sqrt(error);
    std::cout << "\tGauss-Newton iter: " << iter << ", error: " << error
              << std::endl;

    Eigen::VectorXd const delta{H.ldlt().solve(b)};
    if (delta.norm() < kDeltaEps) {
      break;
    }
    rotation *=
        Eigen::Quaterniond(1.0, 0.5 * delta(0), 0.5 * delta(1), 0.5 * delta(2))
            .normalized();
    translation += delta.tail<3>();
  }

  report.success = true;
  report.rotation = rotation;
  report.translation = translation;
  report.inlier_mask.resize(num_pts);
  report.num_inliers = 0UL;
  double const max_inlier_error{
      camera->thresholdPix2Norm(config_.max_abs_pose_reproj_error)};
  double const sqr_max_inlier_error{max_inlier_error * max_inlier_error};
  for (size_t i{0UL}; i < num_pts; ++i) {
    Eigen::Vector2d const residual{
        (rotation.inverse() * (pts_3d[i] - translation)).hnormalized() -
        pts_2d[i]};
    if (residual.squaredNorm() <= sqr_max_inlier_error) {
      report.inlier_mask[i] = true;
      ++report.num_inliers;
    } else {
      report.inlier_mask[i] = false;
    }
  }

  return report;
}

} // namespace simple_vo
} // namespace slam
} // namespace my3d

namespace gopt {

void VertexCameraPose::setToOrigin() {
  estimate_.rotation.setIdentity();
  estimate_.translation.setZero();
}

void VertexCameraPose::plus(Eigen::VectorXd const &update) {
  if (!fix_rot) {
    estimate_.rotation *= Eigen::Quaterniond(1.0, 0.5 * update(0),
                                             0.5 * update(1), 0.5 * update(2));
  }
  if (!fix_trans) {
    estimate_.translation += update.tail<3>();
  }
}

void VertexTrackInvDepth::setToOrigin() { estimate_ = 0.0; }

void VertexTrackInvDepth::plus(Eigen::VectorXd const &update) {
  if (!is_fixed) {
    estimate_ += update(0);
  }
}

void EdgeProjection::computeResidual() {
  auto const camera_pose_0{
      std::dynamic_pointer_cast<VertexCameraPose>(this->getVertex(0UL))
          ->getEstimate()};
  auto const camera_pose_1{
      std::dynamic_pointer_cast<VertexCameraPose>(this->getVertex(1UL))
          ->getEstimate()};
  auto const inv_d{
      std::dynamic_pointer_cast<VertexTrackInvDepth>(this->getVertex(2UL))
          ->getEstimate()};

  // TODO Zero division protection
  double const d0{1.0 / inv_d};
  Eigen::Vector3d const pw{camera_pose_0.rotation *
                               (d0 * measurement_.sphere0) +
                           camera_pose_0.translation};
  Eigen::Vector3d const pc1{camera_pose_1.rotation.inverse() *
                            (pw - camera_pose_1.translation)};
  Eigen::Vector3d const s1{pc1.normalized()};
  residual_ = tangent_base.transpose() * (s1 - measurement_.sphere1);
}

// void EdgeProjection::computeJacobians() {
//   // TODO Support FEJ
//   auto const camera_pose_0{
//       std::dynamic_pointer_cast<VertexCameraPose>(this->getVertex(0UL))
//           ->getEstimate()};
//   auto const camera_pose_1{
//       std::dynamic_pointer_cast<VertexCameraPose>(this->getVertex(1UL))
//           ->getEstimate()};
//   auto const inv_d{
//       std::dynamic_pointer_cast<VertexTrackInvDepth>(this->getVertex(2UL))
//           ->getEstimate()};

//   Eigen::Matrix3d const R0{camera_pose_0.rotation.toRotationMatrix()};
//   Eigen::Matrix3d const R1{camera_pose_1.rotation.toRotationMatrix()};
//   Eigen::Vector3d const t0{camera_pose_0.translation};
//   Eigen::Vector3d const t1{camera_pose_1.translation};

//   double const d0{1.0 / inv_d};
//   Eigen::Vector3d const pw{R0 * (d0 * measurement_.sphere0) + t0};
//   Eigen::Vector3d const tc1_p{pw - t1};
//   Eigen::Vector3d const pc1{R1.transpose() * tc1_p};
//   Eigen::Vector3d const s1{pc1.normalized()};
//   double const d1{pc1.norm()};
//   double const inv_d1{1.0 / d1};
//   Eigen::Matrix3d const ds1_dpc1{inv_d1 * Eigen::Matrix3d::Identity() -
//                                  std::pow(inv_d1, 3.0) * pc1 *
//                                  pc1.transpose()};

//   jacobians_[0] = Eigen::MatrixXd::Zero(2, 6);
//   Eigen::Matrix3d const dpc1_dr0{
//       -d0 * R1.transpose() * R0 *
//       my3d::util::getSkewSymmetric(measurement_.sphere0)};
//   Eigen::Matrix3d const dpc1_dt0{R1.transpose()};
//   jacobians_[0].leftCols<3>() = tangent_base.transpose() * ds1_dpc1 *
//   dpc1_dr0; jacobians_[0].rightCols<3>() = tangent_base.transpose() *
//   ds1_dpc1 * dpc1_dt0;

//   jacobians_[1] = Eigen::MatrixXd::Zero(2, 6);
//   Eigen::Matrix3d const dpc1_dr1{
//       my3d::util::getSkewSymmetric(R1.transpose() * tc1_p)};
//   Eigen::Matrix3d const dpc1_dt1{-R1.transpose()};
//   jacobians_[1].leftCols<3>() = tangent_base.transpose() * ds1_dpc1 *
//   dpc1_dr1; jacobians_[1].rightCols<3>() = tangent_base.transpose() *
//   ds1_dpc1 * dpc1_dt1;

//   jacobians_[2] = Eigen::MatrixXd::Zero(2, 1);
//   Eigen::Vector3d const dpc1_dinvd{-R1.transpose() * R0 *
//   measurement_.sphere0 *
//                                    d0 * d0};
//   jacobians_[2] = tangent_base.transpose() * ds1_dpc1 * dpc1_dinvd;
// }

void EdgeProjection::setTangentBase(
    EdgeProjectionMeasurement const &measurement) {
  Eigen::Vector3d const m{measurement.sphere1.normalized()};
  Eigen::Vector3d const unit_z{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d const b1{std::abs(m.z() - 1.0) <
                                   std::numeric_limits<double>::epsilon()
                               ? Eigen::Vector3d::UnitX()
                               : m.cross(unit_z).normalized()};
  Eigen::Vector3d const b2{m.cross(b1).normalized()};
  tangent_base.col(0) = b1;
  tangent_base.col(1) = b2;
}

} // namespace gopt