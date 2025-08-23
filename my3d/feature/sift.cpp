/*
 * filename: sift.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "feature/sift.h" 

namespace my3d {
namespace feature { 

SIFTKeyPoint::SIFTKeyPoint(const Eigen::Vector2d &_point, 
                           const double &_response, 
                           const double &_edge_score, 
                           const size_t &_octave_idx, 
                           const size_t &_valid_level_idx, 
                           const double &_scale): 
point{_point}, response{_response}, 
edge_score{_edge_score}, 
octave_idx{_octave_idx}, valid_level_idx{_valid_level_idx}, 
scale{_scale} {

}

SIFTDescriptor::SIFTDescriptor(const size_t &_key_point_idx, 
                               const Eigen::Vector2d &_point, 
                               const double &_orientation, 
                               const double &_scale, 
                               const Eigen::Matrix<double, 8, 16> &_histograms): 
key_point_idx{_key_point_idx}, 
point{_point}, orientation{_orientation}, 
scale{_scale}, histograms{_histograms} {

}

SIFTDetector::Octave::Octave(const size_t _octave_idx, 
                             const base::Image &_base_image, 
                             const size_t &_num_valid_dog_levels, 
                             const double &_scale_factor, 
                             const double &_sigma, 
                             const double &_image_scale_factor): 
octave_idx{_octave_idx}, 
gaussian_space{_base_image, _num_valid_dog_levels + 3, _scale_factor, _sigma},  
image_scale_factor{_image_scale_factor} {
    // std::cout << "octave: " << _num_valid_dog_levels << std::endl;
    dog_space.build(gaussian_space); 
}

SIFTDetector::SIFTDetector(const Config &config): 
config_{config} {
    this->config(config); 
    std::cout << "num_valid_dog_levels: " << config_.num_valid_levels_per_octave << std::endl; 
} 

void SIFTDetector::config(const Config &config) {
    config_ = config; 
    if (config_.num_valid_levels_per_octave == 0) { 
        config_.num_valid_levels_per_octave = kDefautNumValidLevelsPerOctave; 
        if (config_.verbose) {
            std::cout << "[WARNING] SIFTDetector: " 
                      << "Invalid number of valid DoG levels. " 
                      << "Set it to default value: " 
                      << kDefautNumValidLevelsPerOctave << std::endl; 
        }
    } 
    if (config_.num_octaves == 0) {
        config_.num_octaves = kDefaultNumOctaves; 
        if (config_.verbose) {
            std::cout << "[WARNING] SIFTDetector: " 
                      << "Invalid number of octaves. " 
                      << "Set it to default value: " 
                      << kDefaultNumOctaves << std::endl; 
        }
    } 
    if (config_.octave_idx_base_image > 0 && 
        config_.num_octaves > 0 && 
        config_.octave_idx_base_image >= config_.num_octaves) {
        config_.octave_idx_base_image = kDefaultOctaveIdxBaseImage; 
        if (config_.verbose) {
            std::cout << "[WARNING] SIFTDetector: " 
                      << "Invalid octave index of the base image. " 
                      << "Set it to default value: " 
                      << kDefaultOctaveIdxBaseImage << std::endl; 
        }
    } 
    if (config_.min_abs_resp < 0.0) {
        config_.min_abs_resp = kDefaultMinAbsDoGResponse; 
        if (config_.verbose) {
            std::cout << "[WARNING] SIFTDetector: " 
                      << "Invalid minimum absolute DoG response. "  
                      << "Set it to default value: " 
                      << kDefaultMinAbsDoGResponse << std::endl; 
        }
    }
    if (config_.edge_score_thresh < 0.0) {
        config_.edge_score_thresh = kDefaultEdgeScoreThresh; 
        if (config_.verbose) {
            std::cout << "[WARNING] SIFTDetector: " 
                      << "Invalid edge score threshold. "  
                      << "Set it to default value: " 
                      << kDefaultEdgeScoreThresh << std::endl; 
        }
    }
    if (config_.base_sigma <= std::numeric_limits<double>::epsilon()) {
        config_.base_sigma = kDefaultBaseSigma; 
        if (config_.verbose) {
            std::cout << "[WARNING] SIFTDetector: " 
                      << "Invalid base sigma. "  
                      << "Set it to default value: " 
                      << kDefaultBaseSigma << std::endl; 
        }
    }
}

void SIFTDetector::setImage(const base::Image &image) {
    assert(!image.isEmpty() && 
           "SIFTDetector: The input image should not be empty. "); 
    assert((image.channels() == 1 || image.channels() == 3) && 
           "SIFTDetector: BGR or gray-scale image is needed. "); 
     
    if (image.channels() == 3) {
        base::convertImageColorBGR2GrayScale(image, image_); 
    } else {
        image_ = image; 
    }

    createOctaves(); 
} 

void SIFTDetector::detect(SIFTKeyPointList &key_points) {
    key_points.clear(); 
    const double edge_score_thresh = std::pow(config_.edge_score_thresh + 1.0, 2.0) / config_.edge_score_thresh; 
    // For all the octaves 
    for (const auto &octave : octaves_) {
        // For each valid DoG level 
        for (size_t valid_level_idx = 0; valid_level_idx < config_.num_valid_levels_per_octave; 
             ++valid_level_idx) {
            // Detect extrema 
            std::vector<size_t> xs;
            std::vector<size_t> ys; 
            std::vector<double> vals; 
            octave.dog_space.detectExtrema(valid_level_idx, xs, ys, vals, config_.min_abs_resp); 
            if (config_.verbose) {
                std::cout << "\t==> [INFO] SIFTDetector: " 
                          << xs.size() << " extrema detected in octave " << octave.octave_idx 
                          << ", level: " << valid_level_idx + 1 << std::endl; 
            }
            // Compute sub-pixel extremum postions 
            size_t num_kpts = 0; 
            for (size_t i = 0; i < xs.size(); ++i) { 
                double response; 
                double edge_score; 
                Eigen::Vector3d pos; 
                bool is_valid = octave.dog_space.computeSubPixelExtremumPosition(xs[i], ys[i], valid_level_idx, 
                                                                        pos, 
                                                                        &response, &edge_score); 
                if (is_valid && 
                    std::abs(response) >= config_.min_abs_resp / static_cast<double>(config_.num_valid_levels_per_octave) && 
                    edge_score > 0 && edge_score <= edge_score_thresh && 
                    pos.x() / octave.image_scale_factor >= 0 && 
                    pos.x() / octave.image_scale_factor <= image_.cols() - 1 && 
                    pos.y() / octave.image_scale_factor >= 0 && 
                    pos.y() / octave.image_scale_factor <= image_.rows() - 1 &&
                    true) {
                    // std::cout << edge_score << "   "; 
                    const double scale = 
                        octave.gaussian_space.getSigma(pos.z()) / octave.image_scale_factor; 
                    key_points.emplace_back(pos.head<2>() / octave.image_scale_factor, response, edge_score, 
                                            octave.octave_idx, static_cast<size_t>(std::floor(pos.z() - 0.5)), 
                                            scale); 
                    ++num_kpts; 
                }
            }
            if (config_.verbose) {
                // std::cout << std::endl; 
                std::cout << "\t==> [INFO] SIFTDetector: " 
                          << num_kpts << " key-points detected in octave " << octave.octave_idx 
                          << ", level: " << valid_level_idx + 1 << std::endl; 
            }
        }
    }
}

void SIFTDetector::compute(const SIFTKeyPointList &key_points, 
                           SIFTDescriptorList &descriptors) {
    descriptors.clear(); 
    for (size_t kpt_idx = 0; kpt_idx < key_points.size(); ++kpt_idx) {
        const SIFTKeyPoint &key_point = key_points[kpt_idx]; 
        std::vector<double> orientations; 
        computeOrientations(key_point, orientations); 
        for (const double &orientation : orientations) {
            SIFTDescriptor descriptor; 
            if (computeDescriptor(key_point, orientation, descriptor)) {
                descriptor.key_point_idx = kpt_idx; 
                descriptors.push_back(descriptor); 
                if (config_.verbose) {
                    std::cout << "\r[INFO] SIFTDetector: Computing descriptors: " 
                            << descriptors.size() << "/" << key_points.size(); 
                }
            }
        }
    }
    if (config_.verbose) {
        std::cout << std::endl; 
    }
}

void SIFTDetector::createOctaves() {
    long int octave_idx_base_image = config_.octave_idx_base_image; 
    if (octave_idx_base_image < 0) {
        if (config_.max_width <= image_.cols() || 
            config_.max_height <= image_.rows()) {
            octave_idx_base_image = 0; 
        } else {
            const long int sw = 
                static_cast<long int>(std::log2(static_cast<double>(config_.max_width) / 
                                                static_cast<double>(image_.cols()))); 
            const long int sh = 
                static_cast<long int>(std::log2(static_cast<double>(config_.max_height) / 
                                                static_cast<double>(image_.rows()))); 
            octave_idx_base_image = std::min(sw, sh); 
        }
    }
    long int num_octaves = 
        (config_.num_octaves <= 0) ? 
        std::max<long int>(1L, std::log2l(std::min(image_.rows(), image_.cols())) - 3L + 
                                static_cast<long int>(octave_idx_base_image)) : 
        static_cast<long int>(config_.num_octaves); 
     
    octaves_.clear(); 
    octaves_.reserve(num_octaves); 
    // Image scaling factor from base image to the first octave 
    const double image_scale_factor = std::pow(2.0 , octave_idx_base_image); 
    const double scale_factor = std::pow(2.0, 1.0 / static_cast<double>(config_.num_valid_levels_per_octave));  
    double curr_image_scale_factor = image_scale_factor; 
    for (long int oct_idx = 0; oct_idx < num_octaves; ++oct_idx) {
        base::Image curr_image; 
        // Blur different size of images using the same base sigma 
        base::scaleImage(image_, curr_image, curr_image_scale_factor);
        // std::cout <<  config_.num_valid_levels_per_octave << std::endl; 
        octaves_.emplace_back(static_cast<size_t>(oct_idx), 
                              curr_image, 
                              config_.num_valid_levels_per_octave, 
                              scale_factor, 
                              config_.base_sigma, 
                              curr_image_scale_factor); 
        if (config_.verbose) {
            std::cout << "\t ==> [INFO] SIFTDetector: " 
                      << "Added octave " << oct_idx << ". " << std::endl 
                      << "\t\timage size: " << curr_image.cols() << "x" << curr_image.rows() << std::endl 
                      << "\t\timage scaling factor: " << curr_image_scale_factor << std::endl 
                      << "\t\tnum Gaussian levels: " << octaves_.back().gaussian_space.getNumLevels() << std::endl 
                      << "\t\tnum DoG levels: " << octaves_.back().dog_space.getNumLevels() << std::endl 
                      << "\t\toctave idx of base image: " << octave_idx_base_image << std::endl 
                      << "\t\tnum octaves: " << num_octaves << std::endl; 

        }
        curr_image_scale_factor *= 0.5;  
    }
} 

void SIFTDetector::computeOrientations(const SIFTKeyPoint &key_point, 
                                       std::vector<double> &orientations) {
#define COMPUTE_SECONDARY_ORIENTATIONS 1
    orientations.clear(); 
    const Octave &octave = octaves_[key_point.octave_idx]; 
    // const base::Image &image = octave.gaussian_space[key_point.valid_level_idx + 1]; 
    // const double relative_scale = octave.dog_space.getSigma(key_point.valid_level_idx + 1);
    const double relative_scale = key_point.scale * octave.image_scale_factor; 
    const size_t width = octave.gaussian_space.getWidth(); 
    const size_t height = octave.gaussian_space.getHeight();  

    const Eigen::Vector2d &position = key_point.point * octave.image_scale_factor; 
    const int center_x = static_cast<int>(position.x() + 0.5); 
    const int center_y = static_cast<int>(position.y() + 0.5); 

    const int kNumBins = 36; 
    Eigen::VectorXd histogram = Eigen::VectorXd::Zero(kNumBins); 
    const double radius = 4.5 *  relative_scale; 
    const double radius2 = radius * radius; 
    const int half_patch_size = static_cast<int>(radius); 
    // if (center_x - half_patch_size < 0 || center_x + half_patch_size >= static_cast<int>(width) || 
    //     center_y - half_patch_size < 0 || center_y + half_patch_size >= static_cast<int>(height)) { 
    //     std::cout << "[WARNING] SIFTDetector: " 
    //               << "Patch exceeds image limit. "
    //               << "No orientation generated for key-point [" << key_point.point.transpose() << "] " 
    //               << "in octave " << key_point.octave_idx 
    //               << " level " << key_point.valid_level_idx << std::endl; 
    //     return; 
    // }

    /* Compute histogram */
    for (int dx = -half_patch_size; dx <= half_patch_size; ++dx) {
        const int x = center_x + dx; 
        if (x < 1 || x > static_cast<int>(width) - 2) {
            continue;
        }
        for (int dy = -half_patch_size; dy <= half_patch_size; ++dy) {
            const int y = center_y + dy; 
            if (y < 1 || y > static_cast<int>(height) - 2) {
                continue;
            }

            const double distance2 = 
                std::pow(x - position.x(), 2) + std::pow(y - position.y(), 2); 
            if (distance2 > radius2) {
                continue;
            } 

            // const Eigen::Vector2d gradient = image.getGrandient(static_cast<size_t>(x), 
            //                                                     static_cast<size_t>(y)); 
            const Eigen::Vector2d gradient = octave.gaussian_space.computeGradient(static_cast<size_t>(x), 
                                                                                   static_cast<size_t>(y), 
                                                                                   key_point.valid_level_idx + 1); 
            double angle = std::atan2(gradient.y(), gradient.x()); 
            if (angle < 0) {
                angle += 2.0 * M_PI; 
            }
            const double bin = kNumBins * angle / (2.0 * M_PI) - 0.5; 
            const int bin_idx = static_cast<int>(bin); 
                // util::clamp<int>(static_cast<int>(kNumBins * angle / (2.0 * M_PI)), 0, kNumBins - 1); 
            const double weight = 
                std::exp(-0.5 * distance2 / std::pow(1.5 * relative_scale, 2.0));
                // util::computeGaussian(std::sqrt(distance2), 1.5 * relative_scale, 0.0); 
            histogram((bin_idx + kNumBins) % kNumBins) += weight * (double(bin_idx) + 1.0 - bin) * gradient.norm(); 
            histogram((bin_idx + 1) % kNumBins) += weight * (bin - double(bin_idx)) * gradient.norm(); 
        }
    } 

    /* Smooth the histogram */ 
    const Eigen::VectorXd tmp_hist = histogram; 
    for (int i = 0; i < kNumBins; ++i) {
        const int i_pp = util::clamp<int>(i - 2, 0, kNumBins - 1); 
        const int i_p = util::clamp<int>(i - 1, 0, kNumBins - 1); 
        const int i_n = util::clamp<int>(i + 1, 0, kNumBins - 1); 
        const int i_nn = util::clamp<int>(i + 2, 0, kNumBins - 1); 
        histogram(i) = (tmp_hist(i_pp) + tmp_hist(i_nn)) * 1.0 / 16.0 + 
                       (tmp_hist(i_p) + tmp_hist(i_n)) * 4.0 / 16.0 + 
                       tmp_hist(i) * 6.0 / 16.0; 
    }
 

#if COMPUTE_SECONDARY_ORIENTATIONS
    /* Find maximum magnitude of histogram */ 
    const double max_mag = histogram.maxCoeff(); 

    /* Find peaks larger than 80% of maximum */
    for (int i = 0; i < kNumBins; ++i) {
        const double h0 = histogram((i + kNumBins - 1) % kNumBins); 
        const double h1 = histogram(i); 
        const double h2 = histogram((i + 1) % kNumBins); 

        if (h1 < 0.8 * max_mag || h1 < h0 || h1 < h2) {
            continue; 
        }

        const double x = -0.5 * (h2 - h0) / (h0 - 2.0 * h1 + h2); 
        double ori_idx = (i + 0.5 + x); 
        if (ori_idx < 0) {
            ori_idx += kNumBins; 
        } else if (ori_idx >= kNumBins) {
            ori_idx -= kNumBins; 
        }
        const double orientation =  ori_idx * 2.0 * M_PI / kNumBins; 
        orientations.push_back(orientation); 
    }
#else
    int arg_max_idx = 0; 
    double max_mag = -1 .0; 
    for (size_t i = 0; i < kNumBins; ++i) {
        if (histogram(i) > max_mag) {
            max_mag = histogram(i); 
            arg_max_idx = static_cast<int>(i); 
        }
    }
    const double orientation = (double)arg_max_idx * 2.0 * M_PI / kNumBins; 
    orientations.push_back(orientation); 
#endif

    if (orientations.empty() && config_.verbose) {
        std::cout << "[WARNING] SIFTDetector: " 
                  << "No orientation generated for key-point [" << key_point.point.transpose() << "] " 
                  << "in octave " << key_point.octave_idx 
                  << " level " << key_point.valid_level_idx << std::endl; 
    }

    // std::cout << "Generated " << orientations.size() << " orientations. " << std::endl; 
} 

bool SIFTDetector::computeDescriptor(const SIFTKeyPoint &key_point, 
                                     const double orientation, 
                                     SIFTDescriptor &descriptor) {
#define IMG_TO_DES 0
#define INTERP_GRAD 0

    const Octave &octave = octaves_[key_point.octave_idx]; 
    // const base::Image &image = octave.gaussian_space[key_point.valid_level_idx + 1]; 
    // const double relative_scale = octave.dog_space.getSigma(key_point.valid_level_idx + 1);
    const double relative_scale = key_point.scale * octave.image_scale_factor; 
    const size_t width = octave.gaussian_space.getWidth(); 
    const size_t height = octave.gaussian_space.getHeight();  

    const Eigen::Vector2d &position = key_point.point * octave.image_scale_factor; 
    const int center_x = static_cast<int>(position.x() + 0.5); 
    const int center_y = static_cast<int>(position.y() + 0.5); 

    const size_t d = 4UL; 
    const double size_bin = 3.0 * relative_scale; 
    const double radius = 0.5 * std::sqrt(2.0) * size_bin * (d + 1);// + 0.5; 
    const int size_ori_hist = 8;
    const int half_patch_size = static_cast<int>(radius); 

#if IMG_TO_DES
    // if (center_x - half_patch_size < 1 || center_x + half_patch_size >= static_cast<int>(width) - 1 || 
    //     center_y - half_patch_size < 1 || center_y + half_patch_size >= static_cast<int>(height) - 1) { 
    //     // if (config_.verbose) {
    //     //     std::cout << "[WARNING] SIFTDetector: " 
    //     //           << "No descriptor generated for key-point [" << position.transpose() << "] " 
    //     //           << "in octave " << key_point.octave_idx << "(" << width << "x" << height << ") " 
    //     //           << " level " << key_point.valid_level_idx << " "
    //     //           << "half patch size: " << half_patch_size << " "
    //     //           << "(" << center_x - half_patch_size << " " << center_x + half_patch_size << " " << width - 1 << ") " 
    //     //           << "(" << center_y - half_patch_size << " " << center_y + half_patch_size << " " << height - 1 << ") " << std::endl; 
    //     // }
    //     return false; 
    // }
#endif

    const double sin_ori = std::sin(orientation); 
    const double cos_ori = std::cos(orientation); 
    double ori = orientation; 
    if (ori < 0) {
        while(ori < 0) {
            ori += 2.0 * M_PI; 
        }
    } else if (ori >= 2.0 * M_PI) {
        while (ori >= 2.0 * M_PI) {
            ori -= 2.0 * M_PI; 
        }
    }
    Eigen::Matrix<double, 8, 16> &hists = descriptor.histograms; 
    hists.setZero(); 

#if IMG_TO_DES
    /* Iterate over the window using (dx, dy) in image frame  */
    for (int dx_img = -half_patch_size; dx_img <= half_patch_size; ++dx_img) {
        for (int dy_img = -half_patch_size; dy_img <= half_patch_size; ++dy_img) {
#else 
    /* Iterate over the window using (dx, dy) in descriptor frame  */
    for (int dx_des = -half_patch_size; dx_des <= half_patch_size; ++dx_des) {
        for (int dy_des = -half_patch_size; dy_des <= half_patch_size; ++dy_des) {
#endif

#if IMG_TO_DES
            // Compute (dx, dy) in descriptor frame 
            const double dx_acc = static_cast<double>(dx_img + center_x) - position.x(); 
            const double dy_acc = static_cast<double>(dy_img + center_y) - position.y(); 
            const double dx_des = cos_ori * dx_acc + sin_ori * dy_acc; 
            const double dy_des = -sin_ori * dx_acc + cos_ori * dy_acc; 
            // const double r = std::sqrt(dx_acc * dx_acc + dy_acc * dy_acc); 
            const double x_img = static_cast<double>(center_x) + static_cast<double>(dx_img); 
            const double y_img = static_cast<double>(center_y) + static_cast<double>(dy_img); 
#else
            // Compute (dx, dy) in image frame
            const double dx_img = cos_ori * dx_des - sin_ori * dy_des; 
            const double dy_img = sin_ori * dx_des + cos_ori * dy_des; 
            const double r = std::sqrt(dx_img * dx_img + dy_img * dy_img); 
            const double x_img = position.x() + dx_img; 
            const double y_img = position.y() + dy_img; 
#endif
            // const double x_img = position.x() + static_cast<double>(dx_img); 
            // const double y_img = position.y() + static_cast<double>(dy_img); 

#if INTERP_GRAD
            // Compute gradient using bilinear interpolation 
            const int x_img1 = static_cast<int>(x_img); 
            const int y_img1 = static_cast<int>(y_img); 
            const int x_img2 = x_img1 + 1; 
            const int y_img2 = y_img1 + 1; 

            if (x_img1 < 1 || x_img1 > static_cast<int>(width) - 2 || 
                x_img2 < 1 || x_img2 > static_cast<int>(width) - 2 || 
                y_img1 < 1 || y_img1 > static_cast<int>(height) - 2 || 
                y_img2 < 1 || y_img2 > static_cast<int>(height) - 2) {
                continue; 
            }
            const Eigen::Vector2d g11 = octave.gaussian_space.computeGradient(x_img1, y_img1, key_point.valid_level_idx + 1); 
            const Eigen::Vector2d g12 = octave.gaussian_space.computeGradient(x_img1, y_img2, key_point.valid_level_idx + 1); 
            const Eigen::Vector2d g21 = octave.gaussian_space.computeGradient(x_img2, y_img1, key_point.valid_level_idx + 1); 
            const Eigen::Vector2d g22 = octave.gaussian_space.computeGradient(x_img2, y_img2, key_point.valid_level_idx + 1); 
            const Eigen::Vector2d gradient = 
                util::computeBilinearInterpolation(g11, g12, g21, g22, 
                                                   x_img1, x_img2, y_img1, y_img2, 
                                                   x_img, y_img);
#else  
            const int x_img1 = static_cast<int>(std::floor(x_img + 0.5));
            const int y_img1 = static_cast<int>(std::floor(y_img + 0.5));
            if (x_img1 < 1 || x_img1 > static_cast<int>(width) - 2 || 
                y_img1 < 1 || y_img1 > static_cast<int>(height) - 2) {
                continue;
            }
            const Eigen::Vector2d gradient = octave.gaussian_space.computeGradient(x_img1, y_img1, key_point.valid_level_idx + 1);
#endif
            double ori_img = std::atan2(gradient.y(), gradient.x()); 
            if (ori_img < 0) {
                ori_img += 2.0 * M_PI;
            }
            double ori_des = ori_img - ori; 
            if (ori_des < 0) {
                ori_des += 2.0 * M_PI;
            }

            const double bin_off = (static_cast<double>(d) - 1.0) * 0.5;
            const double bin_x = dx_des / size_bin + bin_off; 
            const double bin_y = dy_des / size_bin + bin_off; 
            const double bin_r2 = (bin_x - bin_off) * (bin_x - bin_off) + (bin_y - bin_off) * (bin_y - bin_off);
            const double bin_ori = ori_des * size_ori_hist / (2.0 * M_PI) - 0.5; 
            // std::cout << bin_x << " " << bin_y << " " << bin_ori << std::endl; 
            
            const int bin_xs[2] = {static_cast<int>(std::floor(bin_x)), static_cast<int>(std::floor(bin_x)) + 1}; 
            const int bin_ys[2] = {static_cast<int>(std::floor(bin_y)), static_cast<int>(std::floor(bin_y)) + 1}; 
            int bin_oris[2] = {static_cast<int>(std::floor(bin_ori)), static_cast<int>(std::floor(bin_ori)) + 1}; 
            // Weights 
            const double sigma = 0.5 * static_cast<double>(d); 
            const double gaussian_weight = std::exp(-0.5 * bin_r2 / (sigma * sigma)); 
                // util::computeGaussian(r, sigma, 0.0);// * std::sqrt(2.0 * M_PI) * sigma; 
            const double bin_x_ws[2] = {1.0 - (bin_x - bin_xs[0]), bin_x - bin_xs[0]}; 
            const double bin_y_ws[2] = {1.0 - (bin_y - bin_ys[0]), bin_y - bin_ys[0]}; 
            const double bin_ori_ws[2] = {1.0 - (bin_ori - bin_oris[0]), bin_ori - bin_oris[0]}; 

            if (bin_oris[0] < 0) {
                bin_oris[0] += size_ori_hist; 
            }
            if (bin_oris[1] >= size_ori_hist) {
                bin_oris[1] -= size_ori_hist; 
            }

            for (int x = 0; x < 2; ++x) {
                for (int y = 0; y < 2; ++y) {
                    for (int ori = 0; ori < 2; ++ori) {
                        if (bin_xs[x] < 0 || bin_xs[x] >= static_cast<int>(d) || 
                            bin_ys[y] < 0 || bin_ys[y] >= static_cast<int>(d)) {
                            continue; 
                        }
            
                        const size_t hist_idx = bin_xs[x] + d * bin_ys[y]; 
                        // std::cout << bin_oris[ori] << " " << hist_idx << std::endl; 
                        hists(bin_oris[ori], hist_idx) += 
                            gradient.norm() * gaussian_weight * bin_x_ws[x] * bin_y_ws[y] * bin_ori_ws[ori]; 
                        // std::cout << gradient.norm() * gaussian_weight * bin_x_ws[x] * bin_y_ws[x] * bin_ori_ws[ori] << std::endl; 
                    }
                }
            }
            
        }
    }

    /* Normalize descriptor vector */ 
    double sum_sqr = 0.0; 
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            sum_sqr += hists(i, j) * hists(i, j); 
        }
    }
    // hists.normalize(); 

    /* Truncate descriptor values to 0.2 */
    const double thresh = std::sqrt(sum_sqr) * 0.2;
    sum_sqr = 0.0; 
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            const double tmp = std::min(hists(i, j), thresh); 
            hists(i, j) = tmp; 
            sum_sqr += tmp * tmp; 
        }
    }

    const double norm_factor = 512.0 / std::sqrt(sum_sqr); 
    // const double norm_factor = 1.0 / std::sqrt(sum_sqr); 
    hists *= norm_factor; 
    if (std::isnan(hists(0, 0)) || std::isinf(hists(0, 0))) {
        if (config_.verbose) {
            std::cout << "[WARN] SIFTDetector: " 
                      << "NaN detected in histogram. The decriptor will be rejected. " 
                      << std::endl;
        }
        return false; 
    }

    // hists.normalize(); 
    // std::cout << "[" << hists << "]" << std::endl;
    
    descriptor.point = key_point.point; 
    descriptor.scale = key_point.scale; 
    descriptor.orientation = orientation; 

    return true; 
}

void drawSIFTKeyPoints(const base::Image &src, base::Image &dst, 
                       const SIFTKeyPointList &key_points, 
                       const SIFTDescriptorList &descriptors, 
                       const uint8_t color[3], 
                       const double scale, 
                       const double thickness) {
    // Draw keypoints 
    dst = src;
    for (const auto &kpt : key_points) { 
        // std::cout << kpt.point.transpose() << " " << kpt.scale << std::endl; 
        base::drawCircle(dst, dst, kpt.point.x(), kpt.point.y(), kpt.scale, color, 
                         scale, thickness); 
    }

    // Draw orientation 
    for (const auto &des : descriptors) {
        // std::cout << des.scale << " " << des.orientation << std::endl; 
        const Eigen::Vector2d &start = des.point; 
        const Eigen::Vector2d end(start.x() + des.scale * scale * std::cos(des.orientation), 
                                  start.y() + des.scale * scale * std::sin(des.orientation)); 
        base::drawLine(dst, dst, start, end, color, thickness); 
    }
}

void SIFTDetector::clear() {
    image_.clear(); 
    octaves_.clear(); 
}

// void drawSIFTMatches(const base::Image &image1, const SIFTDescriptorList &descriptors1, 
//                      const base::Image &image2, const SIFTDescriptorList &descriptors2, 
//                      const Matches &matches, 
//                      base::Image &out) {
    
// }

}
}