/*
 * filename: BatchFeatureMatcherBase.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_FEATURE_BATCH_FEATURE_MATCHER_BASE_H_
#define _MY3D_FEATURE_BATCH_FEATURE_MATCHER_BASE_H_

#include <fstream>
#include <iostream>
#include <vector>

#include "feature/FeatureMatcherBase.h"

namespace my3d {
namespace feature {

template <typename KeyPointListType, typename DescriptorListType>
class BatchFeatureMatcherBase {
public:
  BatchFeatureMatcherBase() = delete;
  explicit BatchFeatureMatcherBase(
      FeatureMatcherBase<DescriptorListType> *matcher)
      : matcher_{matcher} {
    assert(matcher_ != nullptr);
  }
  ~BatchFeatureMatcherBase() {
    if (matcher_ != nullptr) {
      delete matcher_;
    }
  }

  virtual void match(const FeatureImageList<KeyPointListType,
                                            DescriptorListType> &feature_images,
                     std::vector<PairWiseMatchingInfo> &pair_wise_matching) {
    assert(feature_images.size() >= 2);

    const size_t num_images = feature_images.size();
    pair_wise_matching.clear();
    pair_wise_matching.reserve((num_images - 1) * (num_images - 2) / 2);
    for (size_t i = 0; i < num_images - 1; ++i) {
      for (size_t j = i + 1; j < num_images; ++j) {
        PairWiseMatchingInfo matching;
        matching.query_image_idx = i;
        matching.train_image_idx = j;
        matching.query_image_name = feature_images[i].name;
        matching.train_image_name = feature_images[j].name;
        matcher_->match(feature_images[i].descriptors,
                        feature_images[j].descriptors, matching.matches);
        pair_wise_matching.push_back(matching);
        std::cout << "[BatchFeatureMatcherBase] Found "
                  << matching.matches.size() << " matches between "
                  << feature_images[i].image_idx << " and "
                  << feature_images[j].image_idx << std::endl;
      }
    }
    // pair_wise_matching.reserve((num_images - 1));
    // for (size_t i = 0; i < num_images - 1; ++i) {
    //     PairWiseMatchingInfo matching;
    //     matching.query_image_idx = i;
    //     matching.train_image_idx = i + 1;
    //     matcher_->match(feature_images[i].descriptors, feature_images[i +
    //     1].descriptors,
    //                     matching.matches);
    //     pair_wise_matching.push_back(matching);
    //     std::cout << "[BatchFeatureMatcherBase] Found " <<
    //     matching.matches.size() <<
    //                 " matches between " << feature_images[i].image_idx << "
    //                 and " << feature_images[i + 1].image_idx << std::endl;
    // }
  }

  // virtual bool write(const std::string &path,
  //                    const std::vector<PairWiseMatchingInfo>
  //                    &pair_wise_matching) = 0;

protected:
  FeatureMatcherBase<DescriptorListType> *matcher_;
};

} // namespace feature
} // namespace my3d

#endif // _MY3D_FEATURE_BATCH_FEATURE_MATCHER_BASE_H_