/*
 * filename: sift.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_FEATURE_SIFT_H_
#define _MY3D_FEATURE_SIFT_H_

#include "utils/math.h"
#include "feature/feature_utils.h"

namespace my3d {
namespace feature {

struct SIFTKeyPoint {  
    SIFTKeyPoint() = default; 
    SIFTKeyPoint(const Eigen::Vector2d &_point, 
                 const double &_response, 
                 const double &_edge_score, 
                 const size_t &_octave_idx, 
                 const size_t &_valid_level_idx, 
                 const double &_scale); 

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /* Subpixel coordinate of keypoint in original image */
    Eigen::Vector2d point; 
    /* DoG reponse */
    double response{-1};
    /* Edge score */ 
    double edge_score{-1}; 
    /* Octave index */
    size_t octave_idx{0}; 
    /* Valid DoG level index */
    size_t valid_level_idx{0}; 
    /* Absolute scale (i.e., scaled sigma value) of the keypoint */
    double scale{-1}; 
}; 

struct SIFTDescriptor {
    SIFTDescriptor() = default; 
    SIFTDescriptor(const size_t &_key_point_idx, 
                   const Eigen::Vector2d &_point, 
                   const double &_orientation, 
                   const double &_scale, 
                   const Eigen::Matrix<double, 8, 16> &_histograms); 

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /* Index of the corresponding key-point */
    size_t key_point_idx; 
    /* Subpixel coordinate of keypoint in original image */
    Eigen::Vector2d point; 
    /* Principal orientation (0, 2 * pi) */
    double orientation{-1}; 
    /* Scale (i.e., scaled sigma value) of the keypoint */
    double scale{-1}; 
    /* Gradient histograms */ 
    Eigen::Matrix<double, 8, 16> histograms; 

    void printHist() const {
        std::cout << "["; 
        for (size_t col = 0; col < 16; ++col) {
            for (size_t row = 0; row < 8; ++row) {
                std::cout << histograms(row, col) << " ";
            }
        }
        std::cout << "]" << std::endl; 
    }

    static double computeDistance(const SIFTDescriptor &desc1, const SIFTDescriptor &desc2) {
        return (desc1.histograms - desc2.histograms).squaredNorm(); 
        // double sqr_dist = 0.0; 
        // for (size_t i = 0; i < 128; ++i) {
        //     sqr_dist += std::pow(desc1.histograms.coeff(i) - desc2.histograms.coeff(i), 2.0); 
        // }
        // return sqr_dist;
    }
}; 

typedef EigenVec<SIFTKeyPoint> SIFTKeyPointList; 
typedef EigenVec<SIFTDescriptor> SIFTDescriptorList; 

/**
 * @brief SIFT feature detector. Used for key-point extraction and descriptor computation. 
*/ 
class SIFTDetector {
public: 
    static constexpr size_t kDefautNumValidLevelsPerOctave = 3; 
    static constexpr long int kDefaultNumOctaves = -1; 
    static constexpr long int kDefaultOctaveIdxBaseImage = 1; 
    static constexpr double kDefaultMinAbsDoGResponse = 0.02; 
    static constexpr double kDefaultEdgeScoreThresh = 10.0; 
    static constexpr double kDefaultBaseSigma = 1.6; 
    static constexpr bool kDefaultVerbose = false; 
    
    struct Config {
        /* Number of valid DoG levels per octave */
        size_t num_valid_levels_per_octave{kDefautNumValidLevelsPerOctave}; 
        /* Number of octaves */
        long int num_octaves{kDefaultNumOctaves}; 
        /* Octave index of the base image */
        long int octave_idx_base_image{kDefaultOctaveIdxBaseImage}; 
        /* Minimum absolute DoG response of a key-point. Should be positive */ 
        double min_abs_resp{kDefaultMinAbsDoGResponse}; 
        /* Edge threshold. Should be positive */ 
        double edge_score_thresh{kDefaultEdgeScoreThresh}; 
        /* Sigma of Gaussian kernal applied to the base image */ 
        double base_sigma{kDefaultBaseSigma}; 
        /* Maximum width of an octave. If octave_idx_base_image is set to -1, the actual 
        octave index will be calculated according to this */
        size_t max_width{5000};
        /* Maximum height of an octave. If octave_idx_base_image is set to -1, the actual 
        octave index will be calculated according to this */ 
        size_t max_height{5000}; 
        /* Whether print messages on the console */ 
        bool verbose{kDefaultVerbose}; 
    }; 

    struct Octave { 
        Octave() {} 
        Octave(const size_t _octave_idx, 
               const base::Image &_base_image, 
               const size_t &_num_valid_dog_levels, 
               const double &_scale_factor, 
               const double &_sigma, 
               const double &_image_scale_factor); 
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
        size_t octave_idx; 
        GaussianSpace gaussian_space; 
        DoGSpace dog_space; 
        double image_scale_factor; 
    }; 

    SIFTDetector() = delete; 
    explicit SIFTDetector(const Config &config); 

public: 
    void config(const Config &config); 

    void setImage(const base::Image &image); 

    void detect(SIFTKeyPointList &key_points); 

    void compute(const SIFTKeyPointList &key_points, 
                 SIFTDescriptorList &descriptors); 

    EigenVec<Octave> getOctaves() const { return octaves_; }

    void clear(); 

    Config getConfig() const { return config_; }

protected: 
    void createOctaves(); 

    void computeOrientations(const SIFTKeyPoint &key_point, 
                             std::vector<double> &orientations); 

    bool computeDescriptor(const SIFTKeyPoint &key_point, 
                           const double orientation, 
                           SIFTDescriptor &descriptor); 

protected: 
    Config config_; 
    base::Image image_; 
    EigenVec<Octave> octaves_; 
}; 

void drawSIFTKeyPoints(const base::Image &src, base::Image &dst, 
                       const SIFTKeyPointList &key_points, 
                       const SIFTDescriptorList &descriptors, 
                       const uint8_t color[3], 
                       const double scale = 1.0, 
                       const double thickness = 1.0);

// void drawSIFTMatches(const base::Image &image1, const SIFTDescriptorList &descriptors1, 
//                      const base::Image &image2, const SIFTDescriptorList &descriptors2, 
//                      const Matches &matches, 
//                      base::Image &out); 

} 

}

#endif // _MY3D_FEATURE_SIFT_H_

