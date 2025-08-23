/*
 * filename: OpenCVBatchFeatureExtractor.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:   
 */

#include "feature/OpenCVBatchFeatureExtractor.h"

namespace my3d {
namespace feature {

OpenCVBatchFeatureExtractor::OpenCVBatchFeatureExtractor(FeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat> *extractor): 
BatchFeatureExtractorBase<std::vector<cv::KeyPoint>, cv::Mat>(extractor) {

}

OpenCVBatchFeatureExtractor::~OpenCVBatchFeatureExtractor() {

}

void OpenCVBatchFeatureExtractor::extract(const std::vector<base::Image> &images, 
                                          FeatureImageList<std::vector<cv::KeyPoint>, cv::Mat> &feature_images) {
    const size_t num_images = images.size();
    feature_images.clear();
    feature_images.reserve(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptors;
        extractor_->extract(images[i], key_points, descriptors);
        FeatureImage<std::vector<cv::KeyPoint>, cv::Mat> feature_image;
        feature_image.image_idx = i;
        feature_image.key_points = key_points;
        feature_image.descriptors = descriptors;
        feature_images.push_back(feature_image);
        std::cout << "[OpenCVBatchFeatureExtractor] " << 
                     "Extracted " << key_points.size() << " features from image " << images[i].getName() << std::endl; 
    }
}

}
}
