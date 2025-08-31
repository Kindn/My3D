/*
 * filename: OpenCVBatchFeatureExtractor.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "feature/BatchFeatureExtractorBase.h"

namespace my3d {
namespace feature {

class OpenCVBatchFeatureExtractor
    : public BatchFeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat> {
public:
  OpenCVBatchFeatureExtractor() = delete;
  explicit OpenCVBatchFeatureExtractor(
      FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat> *extractor);
  ~OpenCVBatchFeatureExtractor();

  virtual void extract(const std::vector<base::Image> &images,
                       FeatureImageList<std::vector<cv::KeyPoint>, cv::Mat>
                           &feature_images) override;

  // virtual bool write(const std::string &path,
  //                    const FeatureImageList<std::vector<cv::KeyPoint>,
  //                    cv::Mat> &feature_images) override;
};

} // namespace feature
} // namespace my3d
