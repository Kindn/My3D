/*
 * filename: FeatureMatcherBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_FEATURE_FEATURE_MATCHER_BASE_H_
#define _MY3D_FEATURE_FEATURE_MATCHER_BASE_H_

#include <iostream>
#include <map>

#include "estimators/HomographyEstimator.h"
#include "estimators/RANSAC.h"
#include "feature/feature_utils.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace feature {

template <typename DescriptorListType> class FeatureMatcherBase {
public:
  struct Config {
    virtual bool check() const { return true; }
  };

public:
  FeatureMatcherBase() = delete;
  explicit FeatureMatcherBase(const Config &config) {}
  virtual ~FeatureMatcherBase(){};
  FeatureMatcherBase(const FeatureMatcherBase<DescriptorListType> &other) =
      delete;
  FeatureMatcherBase<DescriptorListType> &
  operator=(const FeatureMatcherBase<DescriptorListType> &other) = delete;

  virtual void
  config(const FeatureMatcherBase<DescriptorListType>::Config &config) {}

  virtual void match(const DescriptorListType &query_descriptors,
                     const DescriptorListType &train_descriptors,
                     Matches &matches) = 0;
};

} // namespace feature
} // namespace my3d

#endif // _MY3D_FEATURE_FEATURE_MATCHER_BASE_H_
