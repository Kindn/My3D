/*
 * filename: OpenCVBatchFeatureMatcher.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:   
 */

#include "feature/OpenCVBatchFeatureMatcher.h"

namespace my3d {
namespace feature {

OpenCVBatchFeatureMatcher::OpenCVBatchFeatureMatcher(FeatureMatcherBase<cv::Mat> *matcher): 
BatchFeatureMatcherBase<std::vector<cv::KeyPoint>, cv::Mat>(matcher) {

}

OpenCVBatchFeatureMatcher::~OpenCVBatchFeatureMatcher() {

}

void OpenCVBatchFeatureMatcher::match(const FeatureImageList<std::vector<cv::KeyPoint>, cv::Mat> &feature_images, 
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
            matcher_->match(feature_images[i].descriptors, feature_images[j].descriptors, 
                            matching.matches);
            pair_wise_matching.push_back(matching);
            std::cout << "[OpenCVBatchFeatureMatcher] Found " << matching.matches.size() << 
                         " matches between " << feature_images[i].image_idx << " and " << feature_images[j].image_idx << std::endl; 
        }
    }
}

}
}
