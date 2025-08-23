/*
 * filename: FeatureExtractionBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_FEATURE_FEATURE_EXTRACTOR_BASE_H_
#define _MY3D_FEATURE_FEATURE_EXTRACTOR_BASE_H_

#include <iostream>
#include <vector>

#include "utils/eigen_types.h"
#include "base/image/Image.h"
#include "feature/feature_utils.h"

#include "simple_json/json.hpp"

namespace my3d {
namespace feature {

template <typename KeyPointListType, typename DescriptorListType>
class FeatureExtractorBase {
public: 
    struct Config {
        /* Maximum feature number in a single image. */
        int max_feature_num = 1000;
        /* Minimum size of a single image. */
        size_t min_image_size = 100;
        /* Maximum size of a single image. */
        size_t max_image_size = 100000000; 

        virtual bool check() const {
            return max_feature_num >= 0 && 
                   min_image_size > 0 && 
                   max_image_size >0; 
        }
    };

public:
    FeatureExtractorBase() {}
    explicit FeatureExtractorBase(const Config &config) {}
    FeatureExtractorBase(const FeatureExtractorBase<KeyPointListType, DescriptorListType> &other) {
        
    }
    // FeatureExtractorBase<KeyPointListType, DescriptorListType> &operator = (const FeatureExtractorBase<KeyPointListType, DescriptorListType> &other) {
        
    // }
    virtual ~FeatureExtractorBase() {}
    
    FeatureExtractorBase<KeyPointListType, DescriptorListType> 
    &operator=(const FeatureExtractorBase<KeyPointListType, DescriptorListType> &other) {
        return *this; 
    }

    virtual std::shared_ptr<FeatureExtractorBase<KeyPointListType, DescriptorListType>> clone() const  = 0;

    virtual void extract(const base::Image &src, KeyPointListType &key_points, DescriptorListType &descriptors) = 0; 

    virtual EigenVec<Eigen::VectorXd> descriptorsToVectors(const DescriptorListType &descriptors) const = 0;

    virtual EigenVec<Eigen::Vector2d> getKeyPointPositions(const KeyPointListType &key_points) const = 0; 

    virtual EigenVec<Eigen::Matrix<uint8_t, 3, 1>> getKeyPointColors(const base::Image &src, const KeyPointListType &key_points) const = 0;  
};

}
}

#endif // _MY3D_FEATURE_FEATURE_EXTRACTOR_BASE_H_
