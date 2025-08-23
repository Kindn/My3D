/*
 * filename: OpenCVMatcher.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "feature/OpenCVMatcher.h"

namespace my3d {
namespace feature {

OpenCVMatcher::OpenCVMatcher(const FeatureMatcherBase<cv::Mat>::Config &config): 
FeatureMatcherBase<cv::Mat>(config) {
    const OpenCVMatcher::Config &derived_config = 
        dynamic_cast<const OpenCVMatcher::Config&>(config); 
    matcher_ = cv::DescriptorMatcher::create(derived_config.matcher_type);
    ratio_thresh_ = derived_config.ratio_thresh;
}

void OpenCVMatcher::match(const cv::Mat &query_descriptors, 
                          const cv::Mat &train_descriptors, 
                          Matches &matches) {
    std::vector<std::vector<cv::DMatch>> original_matches;
    matcher_->knnMatch(query_descriptors, train_descriptors, original_matches, 2);

    // Filter matching points
    // const auto min_max = std::minmax_element(original_matches.begin(), original_matches.end(), 
    //     [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
    // const double min_dist = min_max.first->distance;
    // // double max_dist = min_max.second->distance;

    cv_matches_.clear();
    matches.clear();
    for (const auto &match : original_matches) {
        std::cout << match.size() << std::endl; 
        if (match.size() >= 2) {
            std::cout << match[0].distance << " " << match[1].distance << std::endl; 
            if (match[0].distance < ratio_thresh_ * match[1].distance) {
                cv_matches_.push_back(match[0]); 
                matches.push_back(Match(match[0].queryIdx, 
                                        match[0].trainIdx, 
                                        match[0].distance)); 
            }
        }
    }
    // for (size_t i = 0; i < static_cast<size_t>(query_descriptors.rows); ++i) {
    //     if (original_matches[i].distance <= std::max(2.0 * min_dist, 30.0)) {
    //         cv_matches_.push_back(original_matches[i]);
    //         matches.push_back(Match(original_matches[i].queryIdx, 
    //                                 original_matches[i].trainIdx, 
    //                                 original_matches[i].distance));
    //     }
    // }
}

}
}
