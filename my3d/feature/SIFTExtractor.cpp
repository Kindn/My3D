/*
 * filename: SIFTExtractor.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "feature/SIFTExtractor.h"

namespace my3d {
namespace feature {

SIFTExtractor::SIFTExtractor(const FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList>::Config &config): 
FeatureExtractorBase<SIFTKeyPointList, SIFTDescriptorList>(config) {
    const SIFTExtractor::Config &derived_config = dynamic_cast<const SIFTExtractor::Config&>(config); 
    assert(derived_config.check()); 
    SIFTDetector::Config detector_config; 
    detector_config.num_valid_levels_per_octave = derived_config.num_valid_levels_per_octave; 
    detector_config.num_octaves = derived_config.num_octaves; 
    detector_config.octave_idx_base_image = derived_config.octave_idx_base_image; 
    detector_config.min_abs_resp = derived_config.min_abs_resp; 
    detector_config.edge_score_thresh = derived_config.edge_score_thresh; 
    detector_config.base_sigma = derived_config.base_sigma; 
    detector_config.max_width = derived_config.max_width; 
    detector_config.max_height = derived_config.max_height;
    detector_config.verbose = derived_config.verbose; 
    detector_ = std::make_unique<SIFTDetector>(detector_config); 
} 

void SIFTExtractor::extract(const base::Image &src, 
                            SIFTKeyPointList &key_points, 
                            SIFTDescriptorList &descriptors) {
    assert(detector_ != nullptr); 
    detector_->setImage(src); 
    SIFTKeyPointList orig_kpts; 
    // SIFTDescriptorList orig_descs; 
    detector_->detect(orig_kpts);
    detector_->compute(orig_kpts, descriptors); 
    key_points.resize(descriptors.size()); 
    for (size_t kpt_idx = 0; kpt_idx < key_points.size(); ++kpt_idx) {
        key_points[kpt_idx] = orig_kpts[descriptors[kpt_idx].key_point_idx]; 
    }
    detector_->clear();
}

EigenVec<Eigen::VectorXd> SIFTExtractor::descriptorsToVectors(const SIFTDescriptorList &descriptors) const {
    const size_t num_descs = descriptors.size(); 
    EigenVec<Eigen::VectorXd> vectors(num_descs); 
    for (size_t i = 0; i < num_descs; ++i) {
        const auto &desc = descriptors[i]; 
        Eigen::VectorXd vec(128); 
        for (size_t c = 0; c < 16; ++c) {
            for (size_t r = 0; r < 8; ++r) {
                vec(c * 8 + r) = desc.histograms(r, c); 
            }
        }
        vectors[i] = vec; 
    }

    return vectors; 
} 

EigenVec<Eigen::Vector2d> SIFTExtractor::getKeyPointPositions(const SIFTKeyPointList &key_points) const {
    const size_t num_kpts = key_points.size(); 
    EigenVec<Eigen::Vector2d> positions(num_kpts); 
    for (size_t i = 0; i < num_kpts; ++i) {
        positions[i] = key_points[i].point; 
    }

    return positions; 
}

EigenVec<Eigen::Matrix<uint8_t, 3, 1>> SIFTExtractor::getKeyPointColors(const base::Image &src, const SIFTKeyPointList &key_points) const {
    const size_t num_kpts = key_points.size(); 
    EigenVec<Eigen::Matrix<uint8_t, 3, 1>> colors(num_kpts); 
    for (size_t i = 0; i < num_kpts; ++i) {
        const size_t row = key_points[i].point.y(); 
        const size_t col = key_points[i].point.x(); 
        colors[i].x() = static_cast<int>(src.at(row, col, 2)); 
        colors[i].y() = static_cast<int>(src.at(row, col, 1)); 
        colors[i].z() = static_cast<int>(src.at(row, col, 0)); 
    }

    return colors; 
}

}
}