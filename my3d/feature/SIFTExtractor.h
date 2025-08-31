/*
 * filename: SIFTExtractor.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_FEATURE_SIFT_EXTRACTOR_H_
#define _MY3D_FEATURE_SIFT_EXTRACTOR_H_

#include "feature/FeatureExtractorBase.h"
#include "feature/sift.h"

namespace my3d {
namespace feature {

class SIFTExtractor
    : public FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList> {
public:
  struct Config : public FeatureExtractorBase<SIFTKeyPointList,
                                              SIFTDescriptorList>::Config {
    /* Number of valid DoG levels per octave */
    size_t num_valid_levels_per_octave{
        SIFTDetector::kDefautNumValidLevelsPerOctave};
    /* Number of octaves */
    long int num_octaves{SIFTDetector::kDefaultNumOctaves};
    /* Octave index of the base image */
    long int octave_idx_base_image{SIFTDetector::kDefaultOctaveIdxBaseImage};
    /* Minimum absolute DoG response of a key-point. Should be positive */
    double min_abs_resp{SIFTDetector::kDefaultMinAbsDoGResponse};
    /* Edge threshold. Should be positive */
    double edge_score_thresh{SIFTDetector::kDefaultEdgeScoreThresh};
    /* Sigma of Gaussian kernal applied to the base image */
    double base_sigma{SIFTDetector::kDefaultBaseSigma};
    /* Whether print messages on the console */
    bool verbose{SIFTDetector::kDefaultVerbose};
    /* Maximum width of an octave. If octave_idx_base_image is set to -1, the
    actual octave index will be calculated according to this */
    size_t max_width{5000};
    /* Maximum height of an octave. If octave_idx_base_image is set to -1, the
    actual octave index will be calculated according to this */
    size_t max_height{5000};

    virtual bool check() const override {
      bool ret = FeatureExtractorBase<SIFTKeyPointList,
                                      SIFTDescriptorList>::Config::check();
      return ret && num_valid_levels_per_octave > 0 &&
             (num_octaves < 0 ||
              (num_octaves > 0 && (long)octave_idx_base_image < num_octaves)) &&
             min_abs_resp >= 0 && edge_score_thresh > 0 && base_sigma > 0;
    }
  };

public:
  SIFTExtractor() = delete;
  explicit SIFTExtractor(
      const FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList>::Config
          &config);
  SIFTExtractor(const SIFTExtractor &other)
      : FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList>(other) {
    if (other.detector_ != nullptr) {
      const auto config = other.detector_->getConfig();
      detector_ = std::make_unique<SIFTDetector>(config);
    } else {
      detector_.reset();
    }
  }
  ~SIFTExtractor() {}

  virtual std::shared_ptr<
      FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList>>
  clone() const override {
    return std::shared_ptr<
        FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList>>(
        new SIFTExtractor(*this));
  }

  virtual void extract(const base::Image &src, SIFTKeyPointList &key_points,
                       SIFTDescriptorList &descriptors) override;

  virtual EigenVec<Eigen::VectorXd>
  descriptorsToVectors(const SIFTDescriptorList &descriptors) const override;

  virtual EigenVec<Eigen::Vector2d>
  getKeyPointPositions(const SIFTKeyPointList &key_points) const override;

  virtual EigenVec<Eigen::Matrix<uint8_t, 3, 1>>
  getKeyPointColors(const base::Image &src,
                    const SIFTKeyPointList &key_points) const override;

protected:
  std::unique_ptr<SIFTDetector> detector_{nullptr};
};

} // namespace feature
} // namespace my3d

#endif // _MY3D_FEATURE_SIFT_EXTRACTOR_H_
