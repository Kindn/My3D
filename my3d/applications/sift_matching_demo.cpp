#include <iostream>

#include "base/image/OpenCVImageReader.h"
#include "feature/OpenCVMatcher.h"
#include "feature/BrutalForceMatcher.h"
#include "feature/sift.h" 

using namespace my3d; 

void match(const feature::SIFTDescriptorList &descs1, const feature::SIFTDescriptorList &descs2, 
           feature::Matches &matches, std::vector<cv::DMatch> &cv_matches)  {
    matches.clear(); 
    typedef feature::BrutalForceMatcher<feature::SIFTDescriptor, feature::SIFTDescriptorList> Matcher;
    Matcher::Config matcher_config; 
    matcher_config.ratio_thresh = 0.8; 
    matcher_config.max_inlier_dist = 0.7;
    matcher_config.dist_func = feature::SIFTDescriptor::computeDistance; 
    matcher_config.match_type = Matcher::MatchType::TWO_WAY; 
    Matcher matcher(matcher_config); 
    matcher.match(descs1, descs2, matches); 
}

void drawMatches(const base::Image &image1, const feature::SIFTDescriptorList &descs1, 
                 const base::Image &image2, const feature::SIFTDescriptorList &descs2, 
                 const feature::Matches &matches, 
                 base::Image &out) {
    const size_t rows1 = image1.rows();
    const size_t rows2 = image2.rows(); 
    const size_t cols1 = image1.cols(); 
    const size_t cols2 = image2.cols(); 
    const size_t chs1 = image1.channels(); 
    const size_t chs2 = image2.channels(); 
    assert(chs1 == chs2); 
    const size_t rows = std::max(rows1, rows2); 
    const size_t cols = cols1 + cols2; 
    const size_t chs = chs1; 

    base::Image tmp_out = base::Image(rows, cols, chs); 
    for (size_t r = 0; r < rows1; ++r) {
        for (size_t c = 0; c < cols1; ++c) {
            for (size_t ch = 0; ch < chs; ++ch) {
                tmp_out.at(r, c, ch) = image1.at(r, c, ch); 
            }
        }
    }
    for (size_t r = 0; r < rows2; ++r) {
        for (size_t c = 0; c < cols2; ++c) {
            for (size_t ch = 0; ch < chs; ++ch) {
                tmp_out.at(r, c + cols1, ch) = image2.at(r, c, ch); 
            }
        }
    }
    const uint8_t color[3] = {0, 255, 0}; 
    for (const auto &match : matches) {
        const Eigen::Vector2d point1 = descs1[match.query_index].point; 
        const Eigen::Vector2d point2(descs2[match.train_index].point.x() + static_cast<double>(cols1), 
                                     descs2[match.train_index].point.y());
        base::drawLine(tmp_out, tmp_out, point1, point2, color); 
    }
    out = tmp_out;
}

int main(int argc, char **argv) {
    // feature::Matches test_matches; 
    // test_matches.reserve(10); 

    if (argc <= 2) {
        std::cerr << "Error! Invalid argument. Usage: \n" << 
                     "\tsparse_reconstruction <image_path1> <image_path2>" << std::endl; 
        return -1;
    } 

    /* Read image */
    const std::string image_path1 = argv[1]; 
    const std::string image_path2 = argv[2]; 
    base::OpenCVImageReader image_reader; 
    image_reader.setFocalLengthPrior(3205.9, 3185.7); 
    base::Image image1, image2; 
    image_reader.read(image_path1, image1, base::ImageReaderBase::IMREAD_COLOR); 
    image_reader.read(image_path2, image2, base::ImageReaderBase::IMREAD_COLOR); 
    cv::Mat cv_img1 = cv::imread(image_path1, cv::IMREAD_COLOR);
    cv::Mat cv_img2 = cv::imread(image_path2, cv::IMREAD_COLOR); 

    feature::SIFTKeyPointList origin_key_points_1, origin_key_points_2; 
    feature::SIFTDescriptorList origin_descriptors_1, origin_descriptors_2; 
    feature::SIFTDetector::Config config; 
    config.num_octaves = 4;
    config.min_abs_resp = 0.02 / 3; 
    config.octave_idx_base_image = 1;
    config.max_width = 5000; 
    config.max_height = 5000; 
    config.verbose = true; 
    feature::SIFTDetector detector(config); 
    detector.setImage(image1); 
    detector.detect(origin_key_points_1); 
    detector.compute(origin_key_points_1, origin_descriptors_1); 
    detector.setImage(image2); 
    detector.detect(origin_key_points_2); 
    detector.compute(origin_key_points_2, origin_descriptors_2); 
    detector.clear(); 

    const uint8_t color[3] = {0, 255, 0};
    base::Image image_sift1, image_sift2; 
    feature::drawSIFTKeyPoints(image1, image_sift1, origin_key_points_1, origin_descriptors_1, 
                               color, 1.0); 
    feature::drawSIFTKeyPoints(image2, image_sift2, origin_key_points_2, origin_descriptors_2, 
                               color, 1.0); 
    image_reader.save("./sift1.jpg", image_sift1); 
    image_reader.save("./sift2.jpg", image_sift2); 
    image_sift1.clear(); 
    image_sift2.clear(); 

    std::vector<cv::KeyPoint> cv_kpts1(origin_descriptors_1.size()); 
    std::vector<cv::KeyPoint> cv_kpts2(origin_descriptors_2.size()); 
    // cv::Mat cv_descs1(origin_descriptors_1.size(), 128, CV_32FC1);
    // cv::Mat cv_descs2(origin_descriptors_2.size(), 128, CV_32FC1); 

    for (size_t i = 0; i < origin_descriptors_1.size(); ++i) {
        cv::KeyPoint cv_kpt; 
        cv_kpt.pt.x = origin_descriptors_1[i].point.x();
        cv_kpt.pt.y = origin_descriptors_1[i].point.y();
        cv_kpts1.push_back(cv_kpt); 
        // std::cout << "[";
        for (size_t j = 0; j < 128; ++j) {
            // *cv_descs1.ptr<float>(i, j) = static_cast<float>(origin_descriptors_1[i].histograms.coeff(j)); 
            // std::cout << *cv_descs1.ptr<float>(i, j) << ", ";
            // std::cout << origin_descriptors_2[i].histograms.coeff(j) << ", ";
        }
        // std::cout << "]" << std::endl;
    }
    for (size_t i = 0; i < origin_descriptors_2.size(); ++i) {
        cv::KeyPoint cv_kpt; 
        cv_kpt.pt.x = origin_descriptors_2[i].point.x();
        cv_kpt.pt.y = origin_descriptors_2[i].point.y();
        cv_kpts2.push_back(cv_kpt); 
        // std::cout << "[";
        for (size_t j = 0; j < 128; ++j) {
            // *cv_descs1.ptr<float>(i, j) = static_cast<float>(origin_descriptors_2[i].histograms.coeff(j)); 
            // std::cout << *cv_descs1.ptr<float>(i, j) << ", ";
            // std::cout << origin_descriptors_2[i].histograms.coeff(j) << ", ";
        }
        // std::cout << "]" << std::endl;
    }
    feature::Matches matches; 
    std::vector<cv::DMatch> cv_matches;
    match(origin_descriptors_1, origin_descriptors_2, matches, cv_matches);
    // feature::OpenCVMatcher::Config matcher_config; 
    // matcher_config.matcher_type = cv::DescriptorMatcher::MatcherType::BRUTEFORCE_SL2;
    // matcher_config.ratio_thresh = 1.0; 
    // feature::OpenCVMatcher matcher(matcher_config); 
    // feature::Matches matches; 
    // matcher.match(cv_descs1, cv_descs2, matches); 
    // typedef feature::BrutalForceMatcher<feature::SIFTDescriptor, feature::SIFTDescriptorList> Matcher;
    // Matcher::Config matcher_config; 
    // matcher_config.ratio_thresh = 0.64; 
    // matcher_config.max_inlier_dist = std::numeric_limits<double>::infinity();
    // matcher_config.dist_func = feature::SIFTDescriptor::computeDistance; 
    // matcher_config.match_type = Matcher::MatchType::TWO_WAY; 
    // Matcher matcher(matcher_config); 
    // // matches.reserve(16);
    // matcher.match(origin_descriptors_1, origin_descriptors_2, matches); 
    // = matcher.getOriginalCVMatches(); 
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::DMatch cv_match; 
        cv_match.distance = matches[i].distance; 
        cv_match.queryIdx = matches[i].query_index;
        cv_match.trainIdx = matches[i].train_index; 
        cv_matches.push_back(cv_match); 
    }

    std::cout << "Found " << matches.size() << " matches. " << std::endl; 
    base::Image image_with_matches; 
    // cv::Mat test(3000, 3000, CV_8UC3); 
    // cv::imwrite("test.jpg", test); 
    drawMatches(image1, origin_descriptors_1, image2, origin_descriptors_2, matches, image_with_matches); 
    image_reader.save("./sift_matches.jpg", image_with_matches); 

    std::cout << "[";
    for (const auto &m : matches) {
        std::cout << m.distance << "          "; 
    }
    std::cout << "]" << std::endl;

    // RANSAC
    EigenVec<Eigen::Vector2d> points1, points2; 
    for (size_t i = 0; i < origin_key_points_1.size(); ++i) {
        points1.push_back(origin_key_points_1[i].point); 
    }
    for (size_t i = 0; i < origin_key_points_2.size(); ++i) {
        points2.push_back(origin_key_points_2[i].point); 
    }
    estimator::RANSACConfig ransac_config; 
    ransac_config.max_inlier_error = 4.0;
    ransac_config.confidence = 0.99; 
    ransac_config.min_iter_num = 30; 
    ransac_config.max_iter_num = 100000;
    feature::Matches inlier_matches = 
        feature::getInlierMatches(points1, points2, matches, ransac_config); 
    std::cout << "Extract " << inlier_matches.size() << " inliers matches. " 
              << "Inlier ratio: " << static_cast<double>(inlier_matches.size()) / static_cast<double>(matches.size()) 
              << std::endl; 
    base::Image image_with_inlier_matches; 
    drawMatches(image1, origin_descriptors_1, image2, origin_descriptors_2, inlier_matches, image_with_inlier_matches); 
    image_reader.save("./sift_inlier_matches.jpg", image_with_inlier_matches); 

    cv::Mat cv_img_matches;
    std::vector<cv::KeyPoint> cv_kpts_empty; 
    std::vector<cv::DMatch> cv_matches_empty;
    cv::drawMatches(cv_img1, cv_kpts1, cv_img2, cv_kpts2, cv_matches, cv_img_matches); 
    cv::drawMatches(cv_img1, cv_kpts_empty, cv_img2, cv_kpts_empty, cv_matches_empty, cv_img_matches); 
    // std::cout << cv_img_matches.rows << " " << cv_img_matches.cols << std::endl;
    
    cv::waitKey();

    return 0; 
}
