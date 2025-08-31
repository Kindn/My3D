/*
 * filename: OpenCVORBExtractor.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_FEATURE_OPENCV_SIFT_EXTRACTOR_H_
#define _MY3D_FEATURE_OPENCV_SIFT_EXTRACTOR_H_

#include <iostream>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "feature/FeatureExtractorBase.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace feature {

// struct OpenCVORBKeyPoint {
//     cv::KeyPoint key_point;
//     cv::Mat descriptor;
// };

const int kDefaultMaximumFeatureNumber = 500;
const float kDefaultPyramidScaleFactor = 1.2f;
const int kDefaultPyramidLevels = 8;
const int kDefaultEdgeThreshold = 31;
const int kDefaultLevelOfTheFirstImage = 0;
const int kDefaultWTA_K = 2;
const cv::ORB::ScoreType kDefaultORBScoreType = cv::ORB::HARRIS_SCORE;
const int kDefaultPatchSize = 31;
const int kDefaultFastThreshold = 20;

/**
 * @brief OpenCV-based ORB extractor.
 */
class OpenCVORBExtractor
    : public FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat> {
public:
  struct Config : public FeatureExtractorBase<std::vector<cv::KeyPoint>,
                                              cv::Mat>::Config {
    float pyramid_scale_factor;
    int pyramid_levels;
    int edge_threshold;
    int level_of_the_first_image;
    int WTA_K;
    cv::ORB::ScoreType score_type;
    int patch_size;
    int fast_threshold;

    Config()
        : FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat>::Config(),
          pyramid_scale_factor(kDefaultPyramidScaleFactor),
          pyramid_levels(kDefaultPyramidLevels),
          edge_threshold(kDefaultEdgeThreshold),
          level_of_the_first_image(kDefaultLevelOfTheFirstImage),
          WTA_K(kDefaultWTA_K), score_type(kDefaultORBScoreType),
          patch_size(kDefaultPatchSize), fast_threshold(kDefaultFastThreshold) {
      max_feature_num = kDefaultMaximumFeatureNumber;
    }
  };

  OpenCVORBExtractor() = delete;
  explicit OpenCVORBExtractor(const Config &config);
  OpenCVORBExtractor(const OpenCVORBExtractor &other)
      : FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat>(other) {
    if (other.extractor_ != nullptr) {
      extractor_ = cv::ORB::create(
          other.extractor_->getMaxFeatures(),
          other.extractor_->getScaleFactor(), other.extractor_->getNLevels(),
          other.extractor_->getEdgeThreshold(),
          other.extractor_->getFirstLevel(), other.extractor_->getWTA_K(),
          other.extractor_->getScoreType(), other.extractor_->getPatchSize(),
          other.extractor_->getFastThreshold());
    } else {
      extractor_.reset();
    }
  }
  // OpenCVORBExtractor &operator = (const OpenCVORBExtractor &other) {
  //     FeatureExtractorBase<std::vector<cv::KeyPoint>,
  //     cv::Mat>::operator=(other); extractor_ =
  //     cv::ORB::create(other.extractor_->getMaxFeatures(),
  //                                  other.extractor_->getScaleFactor(),
  //                                  other.extractor_->getNLevels(),
  //                                  other.extractor_->getEdgeThreshold(),
  //                                  other.extractor_->getFirstLevel(),
  //                                  other.extractor_->getWTA_K(),
  //                                  other.extractor_->getScoreType(),
  //                                  other.extractor_->getPatchSize(),
  //                                  other.extractor_->getFastThreshold());
  // }
  virtual ~OpenCVORBExtractor() {}

  virtual std::shared_ptr<
      FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat>>
  clone() const override {
    return std::shared_ptr<
        FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat>>(
        new OpenCVORBExtractor(*this));
  }

  virtual void extract(const base::Image &src,
                       std::vector<cv::KeyPoint> &key_points,
                       cv::Mat &descriptors) override;

  virtual EigenVec<Eigen::VectorXd>
  descriptorsToVectors(const cv::Mat &descriptors) const override;

  virtual EigenVec<Eigen::Vector2d> getKeyPointPositions(
      const std::vector<cv::KeyPoint> &key_points) const override;

  virtual EigenVec<Eigen::Matrix<uint8_t, 3, 1>>
  getKeyPointColors(const base::Image &src,
                    const std::vector<cv::KeyPoint> &key_points) const override;

private:
  cv::Ptr<cv::ORB> extractor_;
};

} // namespace feature
} // namespace my3d

#endif // _MY3D_FEATURE_OPENCV_SIFT_EXTRACTOR_H_
