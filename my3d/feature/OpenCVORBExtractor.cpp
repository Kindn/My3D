/*
 * filename: OpenCVORBExtractor.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "feature/OpenCVORBExtractor.h"

namespace my3d {
namespace feature {

OpenCVORBExtractor::OpenCVORBExtractor(const Config &config)
    : FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat>(config) {
  extractor_ = cv::ORB::create(
      config.max_feature_num, config.pyramid_scale_factor,
      config.pyramid_levels, config.edge_threshold,
      config.level_of_the_first_image, config.WTA_K, config.score_type,
      config.patch_size, config.fast_threshold);
  std::cout << "OpenCVORBExtractor configuration: " << std::endl;
  std::cout << "\tmax_feature_num: " << config.max_feature_num << std::endl;
  std::cout << "\tpyramid_scale_factor: " << config.pyramid_scale_factor
            << std::endl;
  std::cout << "\tpyramid_levels: " << config.pyramid_levels << std::endl;
  std::cout << "\tedge_threshold: " << config.edge_threshold << std::endl;
  std::cout << "\tlevel_of_the_first_image: " << config.level_of_the_first_image
            << std::endl;
  std::cout << "\tWTA_K,: " << config.WTA_K << std::endl;
  std::cout << "\tscore_type: " << config.score_type << std::endl;
  std::cout << "\tpatch_size: " << config.patch_size << std::endl;
  std::cout << "\tfast_threshold: " << config.fast_threshold << std::endl;
}

void OpenCVORBExtractor::extract(const base::Image &src,
                                 std::vector<cv::KeyPoint> &key_points,
                                 cv::Mat &descriptors) {
  assert(src.channels() == 1 || src.channels() == 3);

  int type;
  if (src.channels() == 1) {
    type = CV_8UC1;
  } else {
    type = CV_8UC3;
  }

  cv::Mat cv_src(src.rows(), src.cols(), type, cv::Scalar(0));
  std::copy(src.data(), src.data() + src.size() * src.channels(), cv_src.data);

  extractor_->detect(cv_src, key_points);
  extractor_->compute(cv_src, key_points, descriptors);
}

EigenVec<Eigen::VectorXd>
OpenCVORBExtractor::descriptorsToVectors(const cv::Mat &descriptors) const {
  const size_t num_descs = descriptors.rows;
  const size_t dim = descriptors.cols;
  EigenVec<Eigen::VectorXd> vectors(num_descs);
  for (size_t i = 0; i < num_descs; ++i) {
    vectors[i] = Eigen::VectorXd::Zero(dim);
    for (size_t j = 0; j < dim; ++j) {
      vectors[i](j) = descriptors.at<double>(i, j);
    }
  }

  return vectors;
}

EigenVec<Eigen::Vector2d> OpenCVORBExtractor::getKeyPointPositions(
    const std::vector<cv::KeyPoint> &key_points) const {
  const size_t num_kpts = key_points.size();
  EigenVec<Eigen::Vector2d> positions(num_kpts);
  for (size_t i = 0; i < num_kpts; ++i) {
    positions[i].x() = key_points[i].pt.x;
    positions[i].y() = key_points[i].pt.y;
  }

  return positions;
}

EigenVec<Eigen::Matrix<uint8_t, 3, 1>> OpenCVORBExtractor::getKeyPointColors(
    const base::Image &src, const std::vector<cv::KeyPoint> &key_points) const {
  const size_t num_kpts = key_points.size();
  EigenVec<Eigen::Matrix<uint8_t, 3, 1>> colors(num_kpts);
  for (size_t i = 0; i < num_kpts; ++i) {
    const size_t row = key_points[i].pt.y;
    const size_t col = key_points[i].pt.x;
    colors[i].x() = static_cast<int>(src.at(row, col, 2));
    colors[i].y() = static_cast<int>(src.at(row, col, 1));
    colors[i].z() = static_cast<int>(src.at(row, col, 0));
  }

  return colors;
}

} // namespace feature
} // namespace my3d
