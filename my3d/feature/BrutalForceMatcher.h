/*
 * filename: BrutalForceMatcher.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_FEATURE_BRUTAL_FORCE_MATCHER_H_
#define _MY3D_FEATURE_BRUTAL_FORCE_MATCHER_H_

#include <functional>

#include "estimators/HomographyEstimator.h" 
#include "estimators/RANSAC.h"
#include "feature/FeatureMatcherBase.h" 

namespace my3d {
namespace feature {

/**
 * @brief Descriptor matcher using brutal force method 
*/
template <typename DescriptorType, typename DescriptorListType = EigenVec<DescriptorType>>
class BrutalForceMatcher : public FeatureMatcherBase<DescriptorListType> { 
public: 
    enum MatchType {
        ONE_WAY = 0, 
        TWO_WAY = 1
    }; 

    struct Config : public FeatureMatcherBase<DescriptorListType>::Config {
        double ratio_thresh{0.7}; 
        double max_inlier_dist{0.1}; 
        MatchType match_type{MatchType::ONE_WAY}; 
        /* For cosine similarity check */
        double dist_norm{1.0 / (512.0 * 512.0)}; 
        std::function<double(const DescriptorType &, const DescriptorType &)> dist_func; 

        virtual bool check() const override {
            return ratio_thresh > 0 && dist_func && dist_norm > 0.0; 
        }
    };  

    BrutalForceMatcher() = delete; 
    explicit BrutalForceMatcher(const typename FeatureMatcherBase<DescriptorListType>::Config &config): 
    FeatureMatcherBase<DescriptorListType>(config) {
        this->config(config); 
    } 

    void setDistanceFunction(const std::function<double(const DescriptorType &, const DescriptorType &)> &dist_func) {
        assert(dist_func); 
        dist_func_ = dist_func; 
    }

    virtual void config(const typename FeatureMatcherBase<DescriptorListType>::Config &config) override {
        FeatureMatcherBase<DescriptorListType>::config(config); 
        const BrutalForceMatcher::Config &derived_config = 
            dynamic_cast<const BrutalForceMatcher::Config&>(config); 
        ratio_thresh_ = derived_config.ratio_thresh; 
        max_inlier_dist_ = derived_config.max_inlier_dist; 
        dist_func_ = derived_config.dist_func; 
        match_type_ = derived_config.match_type; 
        dist_norm_ = derived_config.dist_norm; 
    } 

    virtual void match(const DescriptorListType &query_descriptors, 
                       const DescriptorListType &train_descriptors, 
                       Matches &matches) override { 
        switch (match_type_)
        {
        case MatchType::ONE_WAY:
            matchOneWay(query_descriptors, train_descriptors, matches);
            break;

        case MatchType::TWO_WAY: 
            matchTwoWays(query_descriptors, train_descriptors, matches); 
            break;
        
        default:
            matchOneWay(query_descriptors, train_descriptors, matches);
            break;
        }
    }

    void matchOneWay(const DescriptorListType &query_descriptors, 
                     const DescriptorListType &train_descriptors, 
                     Matches &matches) {
        brutalForceMatch(query_descriptors, train_descriptors, matches); 
    }

    void matchTwoWays(const DescriptorListType &query_descriptors, 
                      const DescriptorListType &train_descriptors, 
                      Matches &matches) {
        Matches matches1; 
        brutalForceMatch(query_descriptors, train_descriptors, matches1); 
        Matches matches2; 
        brutalForceMatch(train_descriptors, query_descriptors, matches2); 
        matches.clear(); 
        // matches.reserve(std::min(matches1.size(), matches2.size())); 
        double *dists = nullptr; 
        size_t *q_idxs = nullptr, *t_idxs = nullptr; 
        const size_t reserve_size = query_descriptors.size();
        dists = new double[reserve_size]{0.0}; 
        q_idxs = new size_t[reserve_size]{0}; 
        t_idxs = new size_t[reserve_size]{0}; 
        size_t num_matches = 0; 
        for (const auto &match1: matches1) {
            const size_t q_idx = match1.query_index; 
            const size_t t_idx = match1.train_index; 
            bool is_consistent = false; 
            for (const auto &match2 : matches2) {
                if (match2.query_index == t_idx && 
                    match2.train_index == q_idx) {
                    is_consistent = true; 
                    break;
                }
            }
            if (is_consistent) {
                dists[num_matches] = match1.distance; 
                q_idxs[num_matches] = q_idx; 
                t_idxs[num_matches] = t_idx; 
                ++num_matches;
                // dists.push_back(match1.distance); 
                // q_idxs.push_back(q_idx); 
                // t_idxs.push_back(t_idx); 
                // matches.emplace_back(q_idx, t_idx, match1.distance);
            }
        }
        matches.resize(num_matches); 
        for (size_t i = 0; i < num_matches; ++i) {
            matches[i].distance = dists[i]; 
            matches[i].query_index = q_idxs[i]; 
            matches[i].train_index = t_idxs[i]; 
        }
        delete [] dists;
        delete [] q_idxs;
        delete [] t_idxs; 
    }

    void brutalForceMatch(const DescriptorListType &query_descriptors, 
                          const DescriptorListType &train_descriptors, 
                          Matches &matches) {
        assert(dist_func_); 
        matches.clear(); 
        // std::cout << query_descriptors.size() << std::endl;
        // matches.reserve(10); 
        if (query_descriptors.size() == 0 || train_descriptors.size() < 2) {
            return; 
        }
    
        size_t num_matches = 0; 
        matches.reserve(query_descriptors.size()); 
        for (size_t q_idx = 0; q_idx < query_descriptors.size(); ++q_idx) {
            size_t nearest_t_idx; 
            double min_dist = std::numeric_limits<double>::infinity(); 
            double sec_min_dist = std::numeric_limits<double>::infinity(); 
            bool found_min = false; 
            bool found_sec_min = false; 
            for (size_t t_idx = 0; t_idx < train_descriptors.size(); ++t_idx) {
                const double dist = dist_func_(query_descriptors[static_cast<size_t>(q_idx)], 
                                               train_descriptors[static_cast<size_t>(t_idx)]); 
                if (dist < min_dist) {
                    sec_min_dist = min_dist; 
                    min_dist = dist; 
                    nearest_t_idx = t_idx;  
                    found_min = true; 
                } else {
                    if (dist < sec_min_dist) {
                        sec_min_dist = dist; 
                        found_sec_min = true; 
                    }
                }
            }

            // Check cosine similarity 
            const double angle_min_dist = 
                std::acos(std::min(0.5 * (2.0 - dist_norm_ * min_dist), 1.0)); 
            const double angle_sec_min_dist = 
                std::acos(std::min(0.5 * (2.0 - dist_norm_ * sec_min_dist), 1.0)); 
            // std::cout << angle_min_dist << " " << angle_sec_min_dist << std::endl; 
            if (angle_min_dist < angle_sec_min_dist * ratio_thresh_ && 
                angle_min_dist <= max_inlier_dist_ && 
                found_min && found_sec_min) {
                // std::cout << min_dist << " " << sec_min_dist << std::endl; 
                // dists[num_matches] = min_dist; 
                // q_idxs[num_matches] = q_idx; 
                // t_idxs[num_matches] = nearest_t_idx; 
                matches.emplace_back(q_idx, nearest_t_idx, min_dist); 
                ++num_matches; 
                
            }
        }
        // matches.resize(num_matches); 
        // for (size_t i = 0; i < num_matches; ++i) {
        //     matches[i].distance = dists[i]; 
        //     matches[i].query_index = static_cast<size_t>(q_idxs[i]); 
        //     matches[i].train_index = static_cast<size_t>(t_idxs[i]); 
        // }
        // delete [] dists;
        // delete [] q_idxs;
        // delete [] t_idxs; 
    }

protected: 
    // BrutalForceMatcher<DescriptorType, DescriptorListType>::Config config_; 

    double ratio_thresh_{0.7}; 
    double max_inlier_dist_{0.1};
    double dist_norm_{1.0 / (512.0 * 512.0)}; 
    std::function<double(const DescriptorType &, const DescriptorType &)> dist_func_; 
    MatchType match_type_{MatchType::ONE_WAY}; 
}; 

}
}

#endif // _MY3D_FEATURE_BRUTAL_FORCE_MATCHER_H_
