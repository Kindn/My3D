/*
 * filename: feature_utils.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "feature/feature_utils.h"

namespace my3d {
namespace feature {

GaussianSpace::GaussianSpace()
    : width_{0}, height_{0}, sigma_{1.0}, scale_factor_{1.0} {}

GaussianSpace::GaussianSpace(const base::Image &base_image,
                             const size_t &num_levels,
                             const double &scale_factor, const double &sigma)
    : width_{base_image.cols()}, height_{base_image.rows()}, sigma_{sigma},
      scale_factor_{scale_factor} {
  assert(build(base_image, num_levels, sigma, scale_factor));
}

GaussianSpace::~GaussianSpace() {}

bool GaussianSpace::build(const base::Image &base_image,
                          const size_t &num_levels, const double &sigma,
                          const double &scale_factor) {
  if (base_image.isEmpty()) {
    return false;
  }

  if (sigma < 0 || scale_factor < 0) {
    return false;
  }

  sigma_ = sigma;
  scale_factor_ = scale_factor;

  if (num_levels == 0) {
    sigma_ = sigma;
    return true;
  }

  base::Image base_gray;
  if (base_image.channels() == 3) {
    base::convertImageColorBGR2GrayScale(base_image, base_gray);
  } else if (base_image.channels() != 1) {
    std::cout << __PRETTY_FUNCTION__
              << "[ERROR] The base image given for Gaussian space building "
                 "should be a BGR or gray-scale image, "
              << "but the given image has channel " << base_image.channels()
              << std::endl;
    return false;
  } else {
    base_gray = base_image;
  }

  levels_.resize(num_levels);
  levels_[0] = base_gray.toEigenMatrices<double>()[0] / 255.0;
  base::blurGaussian(levels_[0], levels_[0], sigma);
  double current_sigma = sigma;
  double last_sigma = sigma;
  for (size_t level = 1; level < num_levels; ++level) {
    current_sigma *= scale_factor;
    const double incre_sigma =
        std::sqrt(current_sigma * current_sigma - last_sigma * last_sigma);
    // EigenVec<Eigen::MatrixXd> tmp;
    // base::blurGaussian(base_gray, tmp, current_sigma);
    // levels_[level] = tmp[0];
    // base::blurGaussian(base_gray, levels_[level], current_sigma);
    base::blurGaussian(levels_[level - 1], levels_[level], incre_sigma);
    std::cout << "\r[INFO] Building Gaussian space. " << level + 1 << "/"
              << num_levels;
    last_sigma = current_sigma;
  }
  std::cout << std::endl;
  width_ = base_gray.cols();
  height_ = base_gray.rows();

  return true;
}

Eigen::Vector2d GaussianSpace::computeGradient(const size_t &x, const size_t &y,
                                               const size_t &level_idx) const {
  assert(x >= 1 && x <= width_ - 2);
  assert(y >= 1 && y <= height_ - 2);
  const Eigen::MatrixXd &level = levels_[level_idx];

  return Eigen::Vector2d(0.5 * (level(y, x + 1) - level(y, x - 1)),
                         0.5 * (level(y + 1, x) - level(y - 1, x)));
}

DoGSpace::DoGSpace() : width_{0}, height_{0}, sigma_{1.0} {}

DoGSpace::DoGSpace(const GaussianSpace &gaussian_space)
    : width_{gaussian_space.getWidth()}, height_{gaussian_space.getHeight()},
      sigma_{gaussian_space.getSigma()}, scale_factor_{
                                             gaussian_space.getScaleFactor()} {
  assert(build(gaussian_space));
}

DoGSpace::DoGSpace(const base::Image &base_image,
                   const size_t &num_valid_dog_levels,
                   const double &scale_factor, const double &sigma)
    : width_{base_image.cols()}, height_{base_image.rows()}, sigma_{sigma},
      scale_factor_{scale_factor} {
  assert(build(base_image, num_valid_dog_levels, scale_factor, sigma));
}

DoGSpace::~DoGSpace() {}

bool DoGSpace::build(const GaussianSpace &gaussian_space) {
  const size_t num_levels_gaussian = gaussian_space.getNumLevels();
  if (num_levels_gaussian < 2) {
    return false;
  }

  width_ = gaussian_space.getWidth();
  height_ = gaussian_space.getHeight();
  sigma_ = gaussian_space.getSigma();
  scale_factor_ = gaussian_space.getScaleFactor();

  levels_.resize(num_levels_gaussian - 1);
  for (size_t i = 0; i < levels_.size(); ++i) {
    assert(gaussian_space[i].rows() == static_cast<Eigen::Index>(height_) &&
           gaussian_space[i].cols() == static_cast<Eigen::Index>(width_)); //&&
    //    gaussian_space[i].channels() == 1);
    assert(gaussian_space[i + 1].rows() == static_cast<Eigen::Index>(height_) &&
           gaussian_space[i + 1].cols() ==
               static_cast<Eigen::Index>(width_)); // &&
    //    gaussian_space[i + 1].channels() == 1);
    levels_[i] =
        (gaussian_space[i + 1] -
         gaussian_space[i]); // / 255.0;
                             // base::computeChannelWiseImageDifference(gaussian_space[i
                             // + 1],
                             //                                         gaussian_space[i])[0]
                             //                                         / 255.0;
  }

  return true;
}

bool DoGSpace::build(const base::Image &base_image,
                     const size_t &num_valid_dog_levels,
                     const double &scale_factor, const double &sigma) {
  GaussianSpace gaussian_space;
  if (!gaussian_space.build(base_image, sigma, scale_factor,
                            num_valid_dog_levels + 3)) {
    return false;
  }

  return build(gaussian_space);
}

size_t DoGSpace::getNumValidDoGLevels() const {
  const size_t num_levels = getNumLevels();
  if (num_levels <= 2) {
    return 0;
  } else {
    return num_levels - 2;
  }
}

bool DoGSpace::isExtremumPoint(const size_t &valid_level_idx, const size_t &x,
                               const size_t &y, const double &thresh) const {
  assert(width_ >= 3 && height_ >= 3);
  const size_t num_valid_levels = getNumValidDoGLevels();
  if (valid_level_idx >= num_valid_levels || num_valid_levels == 0) {
    return false;
  }

  if (x < 1 || x >= width_ - 1 || y < 1 || y >= height_ - 1) {
    return false;
  }

  const Eigen::MatrixXd &last_level = levels_[valid_level_idx];
  const Eigen::MatrixXd &curr_level = levels_[valid_level_idx + 1];
  const Eigen::MatrixXd &next_level = levels_[valid_level_idx + 2];
  const double value = curr_level(y, x);
  // bool is_min = true, is_max = true;
  // for (size_t r = y - 1; r <= y + 1 && (is_min || is_max); ++r) {
  //     for (size_t c = x - 1; c <= x + 1 && (is_min || is_max); ++c) {
  //         if (last_level(r, c) >= value || next_level(r, c) >= value) {
  //             is_max = false;
  //         }
  //         if (last_level(r, c) <= value || next_level(r, c) <= value) {
  //             is_min = false;
  //         }
  //         if (r != y || c != x) {
  //             if (curr_level(r, c) >= value) {
  //                 is_max = false;
  //             }
  //             if (curr_level(r, c) <= value) {
  //                 is_min = false;
  //             }
  //         }
  //     }
  // }

  // return is_min || is_max;

  bool is_extremum =
      (value <= -thresh && value < curr_level(y - 1, x - 1) &&
       value < curr_level(y - 1, x) && value < curr_level(y - 1, x + 1) &&
       value < curr_level(y, x - 1) && value < curr_level(y, x + 1) &&
       value < curr_level(y + 1, x - 1) && value < curr_level(y + 1, x) &&
       value < curr_level(y + 1, x + 1) && value < last_level(y - 1, x - 1) &&
       value < last_level(y - 1, x) && value < last_level(y - 1, x + 1) &&
       value < last_level(y, x - 1) && value < last_level(y, x) &&
       value < last_level(y, x + 1) && value < last_level(y + 1, x - 1) &&
       value < last_level(y + 1, x) && value < last_level(y + 1, x + 1) &&
       value < next_level(y - 1, x - 1) && value < next_level(y - 1, x) &&
       value < next_level(y - 1, x + 1) && value < next_level(y, x - 1) &&
       value < next_level(y, x) && value < next_level(y, x + 1) &&
       value < next_level(y + 1, x - 1) && value < next_level(y + 1, x) &&
       value < next_level(y + 1, x + 1)) ||
      (value >= thresh && value > curr_level(y - 1, x - 1) &&
       value > curr_level(y - 1, x) && value > curr_level(y - 1, x + 1) &&
       value > curr_level(y, x - 1) && value > curr_level(y, x + 1) &&
       value > curr_level(y + 1, x - 1) && value > curr_level(y + 1, x) &&
       value > curr_level(y + 1, x + 1) && value > last_level(y - 1, x - 1) &&
       value > last_level(y - 1, x) && value > last_level(y - 1, x + 1) &&
       value > last_level(y, x - 1) && value > last_level(y, x) &&
       value > last_level(y, x + 1) && value > last_level(y + 1, x - 1) &&
       value > last_level(y + 1, x) && value > last_level(y + 1, x + 1) &&
       value > next_level(y - 1, x - 1) && value > next_level(y - 1, x) &&
       value > next_level(y - 1, x + 1) && value > next_level(y, x - 1) &&
       value > next_level(y, x) && value > next_level(y, x + 1) &&
       value > next_level(y + 1, x - 1) && value > next_level(y + 1, x) &&
       value > next_level(y + 1, x + 1));
  return is_extremum;
}

size_t DoGSpace::detectExtrema(const size_t &valid_level_idx,
                               std::vector<size_t> &xs, std::vector<size_t> &ys,
                               std::vector<double> &vals,
                               const double &thresh) const {
  xs.clear();
  ys.clear();
  vals.clear();
  const size_t num_valid_levels = getNumValidDoGLevels();
  if (valid_level_idx >= num_valid_levels || num_valid_levels == 0) {
    std::cout << "[ERROR] DoGSpace: Invalid valid level index. "
              << "valid_level_idx: " << valid_level_idx
              << " num_valid_levels: " << num_valid_levels << std::endl;
    return 0;
  }

  const Eigen::MatrixXd &curr_level = levels_[valid_level_idx + 1];
  for (size_t x = 1; x < width_ - 1; ++x) {
    for (size_t y = 1; y < height_ - 1; ++y) {
      if (isExtremumPoint(valid_level_idx, x, y, thresh)) {
        xs.push_back(x);
        ys.push_back(y);
        vals.push_back(curr_level(y, x));
      }
    }
  }

  return xs.size();
}

bool DoGSpace::computeSubPixelExtremumPosition(const size_t &x, const size_t &y,
                                               const size_t &valid_level_idx,
                                               Eigen::Vector3d &position,
                                               double *response,
                                               double *edge_score) const {
  assert(x >= 1 && x < width_ - 1);
  assert(y >= 1 && y < height_ - 1);
  assert(width_ >= 3 && height_ >= 3);
  const size_t num_valid_levels = getNumValidDoGLevels();
  assert(valid_level_idx < num_valid_levels);

  int x0 = static_cast<int>(x);
  int y0 = static_cast<int>(y);
  int step_x = 0;
  int step_y = 0;
  Eigen::Vector3d delta(0, 0, 0);
  const size_t kNumIterations = 5;
  position << x0, y0, valid_level_idx + 1;
  if (edge_score) {
    *edge_score = std::numeric_limits<double>::infinity();
  }
  if (response) {
    *response = levels_[valid_level_idx + 1].operator()(y0, x0);
  }
  for (size_t i = 0; i < kNumIterations; ++i) {
    x0 += step_x;
    y0 += step_y;
    if (x0 < 1 || x0 >= static_cast<int>(width_) - 1 || y0 < 1 ||
        y0 >= static_cast<int>(height_) - 1) {
      return false;
    }

    const Eigen::Vector3d gradient = computeGradient(x0, y0, valid_level_idx);
    const Eigen::Matrix3d hessian = computeHessian(x0, y0, valid_level_idx);
    if (std::abs(hessian.determinant()) <=
        std::numeric_limits<double>::epsilon()) {
      break;
    }

    // const double &gx = gradient.x();
    // const double &gy = gradient.y();
    // const double &gz = gradient.z();
    // const Eigen::Matrix3d inv_hessian = hessian.inverse();
    // delta.x() = -inv_hessian(0, 0) * gx - inv_hessian(0, 1) * gy -
    // inv_hessian(0, 2) * gz; delta.x() = -inv_hessian(1, 0) * gx -
    // inv_hessian(1, 1) * gy - inv_hessian(1, 2) * gz; delta.x() =
    // -inv_hessian(2, 0) * gx - inv_hessian(2, 1) * gy - inv_hessian(2, 2) *
    // gz; delta = -hessian.inverse() * gradient;
    delta = hessian.ldlt().solve(-gradient);
    if (std::isnan(delta(0)) || std::isinf(delta(0))) {
      return false;
    }
    position.x() = static_cast<double>(x0) + delta.x();
    position.y() = static_cast<double>(y0) + delta.y();
    position.z() = static_cast<double>(valid_level_idx + 1) + delta.z();
    if (response) {
      const double f0 = levels_[valid_level_idx + 1].operator()(y0, x0);
      *response = f0 + 0.5 * gradient.dot(delta);
    }
    if (edge_score) {
      const double trace = hessian(0, 0) + hessian(1, 1);
      const double determinant =
          hessian(0, 0) * hessian(1, 1) - hessian(0, 1) * hessian(1, 0);
      *edge_score = trace * trace / determinant;
    }

    step_x = -1 * int(delta.x() < -0.6 && x0 > 1) +
             1 * int(delta.x() > 0.6 && x0 < static_cast<int>(width_) - 2);
    step_y = -1 * int(delta.y() < -0.6 && y0 > 1) +
             1 * int(delta.y() > 0.6 && y0 < static_cast<int>(height_) - 2);
    if (step_x == 0 && step_y == 0) {
      break;
    }
  }

  const double kNumGaussianSpaceLevels =
      static_cast<double>(getNumLevels()) + 1;
  if (position.x() < 0 || position.x() > width_ - 1 || position.y() < 0 ||
      position.y() > height_ - 1 || position.z() < 0 ||
      position.z() > kNumGaussianSpaceLevels - 1) {
    return false;
  }

  if (std::abs(delta.x()) >= 1.5 || std::abs(delta.y()) >= 1.5 ||
      std::abs(delta.z()) >= 1.5) {
    return false;
  }

  return true;
}

Eigen::Vector3d DoGSpace::computeGradient(const size_t &x, const size_t &y,
                                          const size_t &valid_level_idx) const {
  assert(x >= 1 && x < width_ - 1);
  assert(y >= 1 && y < height_ - 1);
  assert(width_ >= 3 && height_ >= 3);
  const size_t num_valid_levels = getNumValidDoGLevels();
  assert(valid_level_idx < num_valid_levels);

  const Eigen::MatrixXd &last_level = levels_[valid_level_idx];
  const Eigen::MatrixXd &curr_level = levels_[valid_level_idx + 1];
  const Eigen::MatrixXd &next_level = levels_[valid_level_idx + 2];

  Eigen::Vector3d gradient;
  gradient.x() = 0.5 * (curr_level(y, x + 1) - curr_level(y, x - 1));
  gradient.y() = 0.5 * (curr_level(y + 1, x) - curr_level(y - 1, x));
  gradient.z() = 0.5 * (next_level(y, x) - last_level(y, x));

  return gradient;
}

Eigen::Matrix3d DoGSpace::computeHessian(const size_t &x, const size_t &y,
                                         const size_t &valid_level_idx) const {
  assert(x >= 1 && x < width_ - 1);
  assert(y >= 1 && y < height_ - 1);
  const size_t num_valid_levels = getNumValidDoGLevels();
  assert(valid_level_idx < num_valid_levels);

  const Eigen::MatrixXd &last_level = levels_[valid_level_idx];
  const Eigen::MatrixXd &curr_level = levels_[valid_level_idx + 1];
  const Eigen::MatrixXd &next_level = levels_[valid_level_idx + 2];

  double hxx, hyy, hss, hxy, hxs, hys;
  hxx = curr_level(y, x + 1) + curr_level(y, x - 1) - 2.0 * curr_level(y, x);
  hyy = curr_level(y + 1, x) + curr_level(y - 1, x) - 2.0 * curr_level(y, x);
  hss = next_level(y, x) + last_level(y, x) - 2.0 * curr_level(y, x);
  hxy = 0.25 * ((curr_level(y + 1, x + 1) + curr_level(y - 1, x - 1)) -
                (curr_level(y - 1, x + 1) + curr_level(y + 1, x - 1)));
  hxs = 0.25 * ((last_level(y, x - 1) + next_level(y, x + 1)) -
                (last_level(y, x + 1) + next_level(y, x - 1)));
  hys = 0.25 * ((last_level(y - 1, x) + next_level(y + 1, x)) -
                (last_level(y + 1, x) + next_level(y - 1, x)));

  Eigen::Matrix3d hessian;
  hessian(0, 0) = hxx;
  hessian(0, 1) = hxy;
  hessian(0, 2) = hxs;
  hessian(1, 0) = hxy;
  hessian(1, 1) = hyy;
  hessian(1, 2) = hys;
  hessian(2, 0) = hxs;
  hessian(2, 1) = hys;
  hessian(2, 2) = hss;

  return hessian;
}

bool PairWiseMatchingInfo::serialize(std::ofstream &ofs) const {
  if (!ofs.is_open() || !ofs.good()) {
    return false;
  }

  ofs.write((const char *)(&query_image_idx), sizeof(size_t));
  ofs.write((const char *)(&train_image_idx), sizeof(size_t));
  ofs.write((const char *)(query_image_name.data()),
            query_image_name.size() + 1);
  ofs.write((const char *)(train_image_name.data()),
            train_image_name.size() + 1);
  if (!ofs.good()) {
    return false;
  }

  const size_t num_matches = matches.size();
  ofs.write((const char *)(&num_matches), sizeof(size_t));
  ofs.write((const char *)(matches.data()), sizeof(Match) * num_matches);

  return ofs.good();
}

bool PairWiseMatchingInfo::deserialize(std::ifstream &ifs) {
  if (!ifs.is_open() || !ifs.good()) {
    return false;
  }

  matches.clear();

  ifs.read((char *)(&query_image_idx), sizeof(size_t));
  ifs.read((char *)(&train_image_idx), sizeof(size_t));
  if (!ifs.good()) {
    return false;
  }

  if (!(util::readStringFromFStream(query_image_name, ifs) &&
        util::readStringFromFStream(train_image_name, ifs))) {
    return false;
  }

  size_t num_matches;
  ifs.read((char *)(&num_matches), sizeof(size_t));
  matches.resize(num_matches);
  for (size_t i = 0; i < num_matches && ifs.good(); ++i) {
    ifs.read((char *)(&matches[i]), sizeof(Match));
  }

  return ifs.good();
}

feature::Matches
getInlierMatches(const EigenVec<Eigen::Vector2d> &key_points1,
                 const EigenVec<Eigen::Vector2d> &key_points2,
                 const feature::Matches &matches,
                 const estimator::RANSACConfig &ransac_config) {
  const size_t num_matches = matches.size();
  if (num_matches < 8) {
    return {};
  }

  EigenVec<Eigen::Vector2d> points_2D1(num_matches), points_2D2(num_matches);
  for (size_t i = 0; i < num_matches; ++i) {
    points_2D1[i] = key_points1[matches[i].query_index];
    points_2D2[i] = key_points2[matches[i].train_index];
  }

  feature::Matches inliers, inliers_h, inliers_f;
  // Estimate homography
  estimator::LORANSAC<estimator::HomographyEstimator,
                      estimator::HomographyEstimator>
      estimator_h(ransac_config);
  estimator::LORANSAC<estimator::HomographyEstimator,
                      estimator::HomographyEstimator>::Report report_h =
      estimator_h.estimate(points_2D1, points_2D2, false);
  if (report_h.success) {
    inliers_h.reserve(report_h.num_inliers);
    for (size_t i = 0; i < num_matches; ++i) {
      if (report_h.inlier_mask[i]) {
        inliers_h.push_back(matches[i]);
      }
    }
  }

  // Estimate fundamental matrix
  estimator::LORANSAC<estimator::FundamentalMatrixSevenPointEstimator,
                      estimator::FundamentalMatrixEightPointEstimator>
      estimator_f(ransac_config);
  estimator::LORANSAC<estimator::FundamentalMatrixSevenPointEstimator,
                      estimator::FundamentalMatrixEightPointEstimator>::Report
      report_f = estimator_f.estimate(points_2D1, points_2D2, false);
  if (report_f.success) {
    inliers_f.reserve(report_f.num_inliers);
    for (size_t i = 0; i < num_matches; ++i) {
      if (report_f.inlier_mask[i]) {
        inliers_f.push_back(matches[i]);
      }
    }
  }

  if (report_h.success && report_f.success) {
    return inliers_f.size() >= inliers_h.size() ? inliers_f : inliers_h;
  } else if (report_h.success && !report_f.success) {
    return inliers_h;
  } else if (!report_h.success && report_f.success) {
    return inliers_f;
  } else {
    std::cout << "[" << __PRETTY_FUNCTION__ << "]: "
              << "Warning: RANSAC estimation failed. Original matches will be "
                 "returned. "
              << std::endl;

    return matches;
  }
}

} // namespace feature
} // namespace my3d