/*
 * filename: OpenCVBatchFeatureMatcher.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "feature/BatchFeatureMatcherBase.h"
#include "feature/FeatureExtractorBase.h"

namespace my3d {
namespace feature {

class OpenCVBatchFeatureMatcher
    : public BatchFeatureMatcherBase<std::vector<cv::KeyPoint>, cv::Mat> {
public:
  OpenCVBatchFeatureMatcher() = delete;
  explicit OpenCVBatchFeatureMatcher(FeatureMatcherBase<cv::Mat> *matcher);
  ~OpenCVBatchFeatureMatcher();

  virtual void
  match(const FeatureImageList<std::vector<cv::KeyPoint>, cv::Mat>
            &feature_images,
        std::vector<PairWiseMatchingInfo> &pair_wise_matching) override;

  // virtual bool write(const std::string &path,
  //                    const std::vector<PairWiseMatchingInfo>
  //                    &pair_wise_matching) override;
};

} // namespace feature
} // namespace my3d