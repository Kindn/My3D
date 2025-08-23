/*
 * filename: BatchFeatureExtractor.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:   
 */

#ifndef _MY3D_FEATURE_BATCH_FEATURE_EXTRACTOR_H_
#define _MY3D_FEATURE_BATCH_FEATURE_EXTRACTOR_H_

#include <iostream>
#include <vector>
#include <fstream>

#include "utils/threading.h"
#include "feature/FeatureExtractorBase.h" 

namespace my3d {
namespace feature {

template <typename KeyPointListType, typename DescriptorListType>
class BatchFeatureExtractorBase {
public: 
    BatchFeatureExtractorBase() = delete;
    explicit BatchFeatureExtractorBase(FeatureExtractorBase<KeyPointListType, DescriptorListType> *extractor): 
    extractor_{extractor} {
        assert(extractor_ != nullptr);
    }
    virtual ~BatchFeatureExtractorBase() {
        if (extractor_ != nullptr) {
            delete extractor_;
        }
    }

    virtual void extract(const std::vector<base::Image> &images, 
                         FeatureImageList<KeyPointListType, DescriptorListType> &feature_images) {
        const size_t num_images = images.size();
        feature_images.clear();
        feature_images.reserve(num_images);
        for (size_t i = 0; i < num_images; ++i) {
            KeyPointListType key_points;
            DescriptorListType descriptors;
            extractor_->extract(images[i], key_points, descriptors);
            FeatureImage<KeyPointListType, DescriptorListType> feature_image;
            feature_image.image_idx = i;
            feature_image.name = images[i].getName(); 
            feature_image.num_pnts = key_points.size(); 
            feature_image.key_points = key_points;
            feature_image.descriptors = descriptors;
            feature_image.colors = extractor_->getKeyPointColors(images[i], key_points); 
            feature_image.points_2D = extractor_->getKeyPointPositions(key_points); 
            feature_images.push_back(feature_image);
            std::cout << "[BatchFeatureExtractorBase] " << 
                        "Extracted " << key_points.size() << " features from image " << images[i].getName() << std::endl; 
        }
    }

    // virtual bool write(const std::string &path, 
    //                    const FeatureImageList<KeyPointListType, DescriptorListType> &feature_images) = 0;

protected: 
    FeatureExtractorBase<KeyPointListType, DescriptorListType> *extractor_;
};

template <typename KeyPointListType, typename DescriptorListType>
class BatchFeatureExtractorMultiThreadBase : 
public BatchFeatureExtractorBase<KeyPointListType, DescriptorListType> {
public: 
    BatchFeatureExtractorMultiThreadBase() = delete; 
    explicit BatchFeatureExtractorMultiThreadBase(FeatureExtractorBase<KeyPointListType, DescriptorListType> *extractor, 
                                                  const size_t &min_num_images_per_thread = 1): 
    BatchFeatureExtractorBase<KeyPointListType, DescriptorListType>(extractor) {
        setMinNumImagesPerThread(min_num_images_per_thread); 
    }
    virtual ~BatchFeatureExtractorMultiThreadBase() {}

protected: 
    virtual void threadExtraction(const std::vector<base::Image> &all_images, 
                                  const std::vector<size_t> &image_idxs, 
                                  FeatureImageList<KeyPointListType, DescriptorListType> *feature_images) {
        /* Clone a new extractor from extractor_ */
        const std::shared_ptr<FeatureExtractorBase<KeyPointListType, DescriptorListType>> extractor 
            = this->extractor_->clone(); 
        
        const size_t num_images = all_images.size();

        if (feature_images != nullptr) {
            // mtx_feature_images_.lock(); 
            feature_images->clear();
            feature_images->reserve(num_images);
            // mtx_feature_images_.unlock(); 
        }
        
        for (const size_t &i : image_idxs) {
            KeyPointListType key_points;
            DescriptorListType descriptors;
            extractor->extract(all_images[i], key_points, descriptors);
            FeatureImage<KeyPointListType, DescriptorListType> feature_image;
            feature_image.image_idx = i;
            feature_image.key_points = key_points;
            feature_image.descriptors = descriptors;
            feature_image.colors = this->extractor_->getKeyPointColors(all_images[i], key_points); 
            feature_image.points_2D = this->extractor_->getKeyPointPositions(key_points); 
            feature_image.name = all_images[i].getName(); 
            if (feature_images != nullptr) {
                // mtx_feature_images_.lock(); 
                feature_images->push_back(feature_image);
                std::cout << "[BatchFeatureExtractorBase] " << 
                            "Extracted " << key_points.size() << " features from image " << all_images[i].getName() << std::endl; 
                // mtx_feature_images_.unlock(); 
            }
        }
    }

public: 
    void setMinNumImagesPerThread(const size_t &min_num_images_per_thread) {
        assert(min_num_images_per_thread > 0); 
        min_num_images_per_thread_ = min_num_images_per_thread; 
    }

    size_t getMinNumImagesPerThread() const {
        return min_num_images_per_thread_; 
    }

    virtual size_t computeNumThreads(const size_t &num_images) const {
        if (num_images == 0) {
            return 0; 
        }
        const size_t max_num_threads = 
            (num_images + min_num_images_per_thread_ - 1) / min_num_images_per_thread_; 
        const size_t hardware_concurrency = std::thread::hardware_concurrency(); 

        return std::min(hardware_concurrency != 0 ? hardware_concurrency : 2, max_num_threads); 
    }

    virtual std::vector<std::vector<size_t>> assignImages(const size_t &num_images, 
                                                          const size_t &num_threads) const {
        std::vector<std::vector<size_t>> assignment; 
        if (num_threads == 0) {
            return assignment; 
        }
        assignment.resize(num_threads); 
        if (num_images == 0) {
            return assignment; 
        }
        size_t curr_image_idx = 0;
        const size_t num_images_per_thread = std::max(num_images / num_threads, min_num_images_per_thread_); 
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            for (size_t i = 0; i < num_images_per_thread && curr_image_idx < num_images; ++i) {
                assignment[thread_idx].push_back(curr_image_idx); 
                ++curr_image_idx; 
            }
        }
        if (curr_image_idx <= num_images - 1) {
            for (size_t i = curr_image_idx; i < num_images; ++i) {
                assignment.back().push_back(i); 
            }
        }

        return assignment; 
    }

    virtual void extract(const std::vector<base::Image> &images, 
                         FeatureImageList<KeyPointListType, DescriptorListType> &feature_images) override {
        feature_images.clear(); 
        if (images.empty()) {
            return; 
        }
        /* Compute number of threads */
        const size_t num_images = images.size(); 
        const size_t num_threads = computeNumThreads(num_images); 
        /* Create threads and assign images to each thread */
        EigenVec<FeatureImageList<KeyPointListType, DescriptorListType>> temp_feature_images_lists(num_threads); 
        std::vector<std::thread> threads; 
        threads.reserve(num_threads); 
        std::vector<util::ThreadGuard *> thread_guards; 
        thread_guards.reserve(num_threads); 
        const std::vector<std::vector<size_t>> assignment = assignImages(num_images, num_threads); 
        std::cout << "[INFO] Creating " << num_threads << " threads... " << std::endl; 
        for (size_t i = 0; i < num_threads; ++i) {
            if (!assignment[i].empty()) {
                threads.emplace_back(&BatchFeatureExtractorMultiThreadBase::threadExtraction, 
                    this, images, assignment[i], &temp_feature_images_lists[i]); 
                thread_guards.push_back(new util::ThreadGuard(threads.back())); 
            } else {
                break;
            }
        }
        /* Wait until all the threads return */
        for (size_t i = 0; i < thread_guards.size(); ++i) {
            delete thread_guards[i]; 
            thread_guards[i] = nullptr; 
        }
        /* Collect the results */
        feature_images.reserve(threads.size()); 
        for (size_t i = 0; i < threads.size(); ++i) {
            for (const auto &feature_image : temp_feature_images_lists[i]) {
                feature_images.push_back(feature_image); 
            }
        }
    }

protected: 
    size_t min_num_images_per_thread_{1}; 
    std::mutex mtx_feature_images_; 
}; 

}
}

#endif // _MY3D_FEATURE_BATCH_FEATURE_EXTRACTOR_H_
