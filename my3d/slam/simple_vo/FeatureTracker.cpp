/*
 * filename: FeatureTracker.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    feature tracker for SimpleVO
 */

#include "FeatureTracker.h"

namespace my3d {
namespace slam {
namespace simple_vo {

FeatureTracker::FeatureTracker(Config const &config) : config_{config} {
  assert(config_.Check());
}

FeatureTracker::Result
FeatureTracker::track(std::shared_ptr<base::Image> const &src,
                      std::shared_ptr<base::Image> const &tgt,
                      EigenUMap<size_t, InputPoint> const &pts,
                      double const dt) {
  size_t const num_pts{pts.size()};

  Result result{};
  if (num_pts == 0UL) {
    result.tracked_pts.clear();
    result.tracked_add_pts.clear();
    result.success = true;
    return result;
  }

  result.success = false;

  if (src->isEmpty() || tgt->isEmpty()) {
    return result;
  }

  if (src->channels() != 1UL) {
    return result;
  }

  if (src->rows() != tgt->rows() || src->cols() != tgt->cols() ||
      src->channels() != tgt->channels()) {
    return result;
  }

  double constexpr kDtEps{1.0e-8};

  //* Step 1: Track existing points
  std::vector<size_t> ids{};
  EigenVec<Eigen::Vector2d> pts_vec{};
  EigenVec<Eigen::Vector2d> tracked_pts_vec{};
  ids.reserve(num_pts);
  pts_vec.reserve(num_pts);
  for (auto const &pt : pts) {
    ids.emplace_back(pt.first);
    pts_vec.emplace_back(pt.second.pos);
  }
  std::vector<uint8_t> status{};
  std::vector<float> errors{};
  trackLK(src, tgt, pts_vec, tracked_pts_vec, status, errors);
  result.tracked_pts.clear();
  result.tracked_pts.reserve(num_pts);
  for (size_t i{0UL}; i < num_pts; ++i) {
    TrackedPoint tracked_pt{};
    tracked_pt.point_id = ids[i];
    tracked_pt.pixel_coord_src = pts.at(ids[i]).pos;
    tracked_pt.pixel_coord = tracked_pts_vec[i];
    tracked_pt.sphere_coord =
        config_.camera->pix2Sphere(tracked_pt.pixel_coord);
    if (dt > kDtEps) {
      tracked_pt.pixel_vel =
          (tracked_pt.pixel_coord - tracked_pt.pixel_coord_src) / dt;
    } else {
      tracked_pt.pixel_vel.setZero();
    }
    tracked_pt.valid = (status[i] > 0);
    tracked_pt.lk_error = errors[i];
    result.tracked_pts.emplace(ids[i], tracked_pt);
  }

  //* Step 2: If the number of existing points is less than max_num_corners,
  //* additional corners are extracted on src as a supplement
  EigenVec<Eigen::Vector2d> const add_pts_vec{
      extractAdditionalCorners(src, pts)};

  //* Step 3: Track the additional corners
  size_t const num_add_pts{add_pts_vec.size()};
  result.tracked_add_pts.clear();
  result.tracked_add_pts.reserve(num_add_pts);
  if (num_add_pts > 0UL) {
    std::vector<size_t> add_ids{};
    EigenVec<Eigen::Vector2d> tracked_add_pts_vec{};
    add_ids.reserve(num_add_pts);
    for (size_t i{0UL}; i < num_add_pts; ++i) {
      add_ids.emplace_back(add_pts_cnt_ + i);
    }
    std::vector<uint8_t> status_add{};
    std::vector<float> errors_add{};
    trackLK(src, tgt, add_pts_vec, tracked_add_pts_vec, status_add, errors_add);
    for (size_t i{0UL}; i < num_add_pts; ++i) {
      TrackedPoint tracked_pt{};
      tracked_pt.point_id = add_ids[i];
      tracked_pt.pixel_coord_src = add_pts_vec[i];
      tracked_pt.pixel_coord = tracked_add_pts_vec[i];
      tracked_pt.sphere_coord =
          config_.camera->pix2Sphere(tracked_pt.pixel_coord);
      if (dt > kDtEps) {
        tracked_pt.pixel_vel =
            (tracked_pt.pixel_coord - tracked_pt.pixel_coord_src) / dt;
      } else {
        tracked_pt.pixel_vel.setZero();
      }
      tracked_pt.valid = (status_add[i] > 0);
      tracked_pt.lk_error = errors_add[i];
      result.tracked_add_pts.emplace_back(tracked_pt);
    }
  }

  result.success = true;

  return result;
}

void FeatureTracker::reset() { add_pts_cnt_ = 0UL; }

void FeatureTracker::trackLK(std::shared_ptr<base::Image> const &src,
                             std::shared_ptr<base::Image> const &tgt,
                             EigenVec<Eigen::Vector2d> const &pts,
                             EigenVec<Eigen::Vector2d> &tracked_pts,
                             std::vector<uint8_t> &status,
                             std::vector<float> &lk_errors) const {
  assert(src->rows() == tgt->rows() && src->cols() == tgt->cols());
  assert(!src->isEmpty() && !tgt->isEmpty());
  assert(src->channels() == 1UL);
  assert(src->channels() == tgt->channels());
  size_t const num_pts{pts.size()};
  std::vector<cv::Point2f> pts_cv(num_pts);
  std::vector<cv::Point2f> tracked_pts_cv(num_pts);
  for (size_t i{0UL}; i < num_pts; ++i) {
    pts_cv[i].x = static_cast<float>(pts[i].x());
    pts_cv[i].y = static_cast<float>(pts[i].y());
    tracked_pts_cv[i].x = static_cast<float>(pts[i].x());
    tracked_pts_cv[i].y = static_cast<float>(pts[i].y());
  }
  cv::Mat const src_cv(cv::Size(static_cast<int32_t>(src->cols()),
                                static_cast<int32_t>(src->rows())),
                       CV_8UC1, (void *)src->data());
  cv::Mat const tgt_cv(cv::Size(static_cast<int32_t>(tgt->cols()),
                                static_cast<int32_t>(tgt->rows())),
                       CV_8UC1, (void *)tgt->data());
  cv::calcOpticalFlowPyrLK(
      src_cv, tgt_cv, pts_cv, tracked_pts_cv, status, lk_errors,
      cv::Size(21, 21), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                       0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);
  if (config_.enable_two_way_check) {
    std::vector<uint8_t> status_back{};
    std::vector<float> lk_errors_back{};
    std::vector<cv::Point2f> tracked_pts_cv_back{tracked_pts_cv};
    cv::calcOpticalFlowPyrLK(
        tgt_cv, src_cv, tracked_pts_cv, tracked_pts_cv_back, status_back,
        lk_errors_back, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    double const max_error2{config_.max_two_way_check_error *
                            config_.max_two_way_check_error};
    for (size_t i{0UL}; i < num_pts; ++i) {
      Eigen::Vector2d const diff(pts_cv[i].x - tracked_pts_cv_back[i].x,
                                 pts_cv[i].y - tracked_pts_cv_back[i].y);
      status[i] =
          (status[i] > 0 && status_back[i] && diff.squaredNorm() <= max_error2)
              ? 1
              : 0;
    }
  }

  if (config_.enable_two_view_geometry_outlier_rejection) {
    // TODO
  }

  tracked_pts.clear();
  tracked_pts.resize(num_pts);
  for (size_t i{0UL}; i < num_pts; ++i) {
    tracked_pts[i].x() = tracked_pts_cv[i].x;
    tracked_pts[i].y() = tracked_pts_cv[i].y;
    int32_t const xi{static_cast<int32_t>(std::round(tracked_pts_cv[i].x))};
    int32_t const yi{static_cast<int32_t>(std::round(tracked_pts_cv[i].y))};
    if (xi < 0 || xi > static_cast<int32_t>(tgt->cols() - 1UL) ||
        yi < 0 || yi > static_cast<int32_t>(tgt->rows() - 1UL)) {
      status[i] = 0;
    }
  }
}

EigenVec<Eigen::Vector2d> FeatureTracker::extractAdditionalCorners(
    std::shared_ptr<base::Image> const &src,
    EigenUMap<size_t, InputPoint> const &existing_pts) {
  size_t const num_existing_pts{existing_pts.size()};
  size_t const num_add_pts{
      num_existing_pts < static_cast<size_t>(config_.max_num_corners)
          ? static_cast<size_t>(config_.max_num_corners) - num_existing_pts
          : 0UL};

  cv::Mat const src_cv(cv::Size(static_cast<int32_t>(src->cols()),
                                static_cast<int32_t>(src->rows())),
                       CV_8UC1, (void *)src->data());
  //* Generate mask
  cv::Mat mask(static_cast<int32_t>(src->rows()),
               static_cast<int32_t>(src->cols()), CV_8UC1, cv::Scalar(255));
  if (!existing_pts.empty()) {
    std::vector<size_t> indices{};
    indices.reserve(existing_pts.size());
    for (auto const &p : existing_pts) {
      indices.emplace_back(p.first);
    }
    std::sort(indices.begin(), indices.end(),
              [&existing_pts](size_t const a, size_t const b) {
                return existing_pts.at(a).track_len >
                       existing_pts.at(b).track_len;
              });
    for (size_t i{0UL}; i < num_existing_pts; ++i) {
      cv::Point2f const center(
          static_cast<float>(existing_pts.at(indices[i]).pos.x()),
          static_cast<float>(existing_pts.at(indices[i]).pos.y()));
      cv::circle(mask, center, config_.min_dist, cv::Scalar(0), -1);
    }
  }

  //* Extract additional corners
  std::vector<cv::Point2f> add_pts_cv{};
  if (num_add_pts > 0UL) {
    cv::goodFeaturesToTrack(src_cv, add_pts_cv,
                            static_cast<int32_t>(num_add_pts), 0.01,
                            config_.min_dist, mask);
  }
  EigenVec<Eigen::Vector2d> add_pts(add_pts_cv.size());
  for (size_t i{0UL}; i < add_pts.size(); ++i) {
    add_pts[i].x() = add_pts_cv[i].x;
    add_pts[i].y() = add_pts_cv[i].y;
  }

  add_pts_cnt_ += add_pts.size();

  return add_pts;
}

} // namespace simple_vo
} // namespace slam
} // namespace my3d