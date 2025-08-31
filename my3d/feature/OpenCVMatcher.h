/*
 * filename: OpenCVMatcher.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_FEATURE_OPENCV_MATCHER_H_
#define _MY3D_FEATURE_OPENCV_MATCHER_H_

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "feature/FeatureMatcherBase.h"

namespace my3d {
namespace feature {

class OpenCVMatcher : public FeatureMatcherBase<cv::Mat> {
public:
  struct Config : public FeatureMatcherBase<cv::Mat>::Config {
    cv::DescriptorMatcher::MatcherType matcher_type;
    double ratio_thresh{0.7};

    Config()
        : FeatureMatcherBase<cv::Mat>::Config(),
          matcher_type{cv::DescriptorMatcher::MatcherType::BRUTEFORCE} {}

    virtual bool check() const override { return ratio_thresh > 0; }
  };

public:
  OpenCVMatcher() = delete;
  explicit OpenCVMatcher(const FeatureMatcherBase<cv::Mat>::Config &config);
  virtual ~OpenCVMatcher(){};

  virtual void match(const cv::Mat &query_descriptors,
                     const cv::Mat &train_descriptors,
                     Matches &matches) override;

  std::vector<cv::DMatch> getOriginalCVMatches() const { return cv_matches_; }

protected:
  double ratio_thresh_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  std::vector<cv::DMatch> cv_matches_;
};

} // namespace feature
} // namespace my3d

#endif // _MY3D_FEATURE_OPENCV_MATCHER_H_
