/*
 * filename: two_view_geomtry_demo.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Auto-sparse-reconstruction pipeline
 */

#include <iostream>

#include "utils/io.h"
#include "base/image/OpenCVImageReader.h"
#include "base/image/image_io.h"
#include "feature/OpenCVORBExtractor.h"
#include "feature/OpenCVMatcher.h"
#include "feature/OpenCVBatchFeatureExtractor.h"
#include "feature/OpenCVBatchFeatureMatcher.h"
#include "feature/SIFTExtractor.h"
#include "feature/BrutalForceMatcher.h"
#include "sfm/SceneBuilder.h"
#include "sfm/IncrementalSFM.h"

using namespace my3d; 

int main(int argc, char **argv) {
    if (argc <= 2) {
        std::cerr << "Error! Invalid argument. Usage: \n" << 
                     "\tsparse_reconstruction <image_path1> <image_path2>" << std::endl; 
        return -1;
    } 

    /* Read image */
    const std::string image_path1 = argv[1]; 
    const std::string image_path2 = argv[2]; 
    base::OpenCVImageReader image_reader; 
    // image_reader.setFocalLengthPrior(3205.9, 3185.7); 
    image_reader.setFocalLengthPrior(521, 521); 
    base::Image image1, image2; 
    image_reader.read(image_path1, image1, base::ImageReaderBase::IMREAD_COLOR); 
    image_reader.read(image_path2, image2, base::ImageReaderBase::IMREAD_COLOR); 
    cv::Mat cv_img1 = cv::imread(image_path1, cv::IMREAD_COLOR);
    cv::Mat cv_img2 = cv::imread(image_path2, cv::IMREAD_COLOR); 
    std::cout << "size of image1: (" << image1.cols() << ", " << image1.rows() << ")" << std::endl;
    std::cout << "size of image2: (" << image2.cols() << ", " << image2.rows() << ")" << std::endl;
    const auto camera1 = std::dynamic_pointer_cast<base::RadialPinHoleCamera>(image1.getCameraModel()); 
    const auto camera2 = std::dynamic_pointer_cast<base::RadialPinHoleCamera>(image2.getCameraModel()); 
    const Eigen::Matrix3d K1 = camera1->getCameraMatrix(); 
    const Eigen::Matrix3d K2 = camera2->getCameraMatrix(); 
    const cv::Mat cv_K = (cv::Mat_<double>(3, 3) << K1(0, 0), 0, K1(0, 2), 0, K1(1, 1), K1(1, 2), 0, 0, 1);
    std::cout << "cv_K: \n" << cv_K << std::endl; 

    /* Extract and match features */ 
    feature::FeatureImage<feature::SIFTKeyPointList, feature::SIFTDescriptorList> features1, features2; 
    feature::FeatureImageList<feature::SIFTKeyPointList, feature::SIFTDescriptorList> features;
    feature::SIFTExtractor::Config extractor_config; 
    extractor_config.num_octaves = -1; 
    extractor_config.octave_idx_base_image = 0; 
    feature::SIFTExtractor extractor(extractor_config); 
    extractor.extract(image1, features1.key_points, features1.descriptors); 
    extractor.extract(image2, features2.key_points, features2.descriptors); 
    // std::cout << features1.key_points.size() << ", " << features2.key_points.size() << std::endl; 
    // std::cout << "-------KPs in img1-------" << std::endl; 
    // for (const auto &kp : features1.key_points) {
    //     std::cout << "(" << kp.pt.x << ", " << kp.pt.y << ") ";  
    // }
    // std::cout << std::endl; 
    // std::cout << "-------KPs in img2-------" << std::endl; 
    // for (const auto &kp : features2.key_points) {
    //     std::cout << "(" << kp.pt.x << ", " << kp.pt.y << ") ";  
    // }
    // std::cout << std::endl; 
    
    feature::BrutalForceMatcher<feature::SIFTDescriptor, feature::SIFTDescriptorList>::Config matcher_config; 
    feature::Matches matches; 
    matcher_config.ratio_thresh = 0.8; 
    matcher_config.max_inlier_dist = std::numeric_limits<double>::infinity();
    matcher_config.dist_func = feature::SIFTDescriptor::computeDistance; 
    matcher_config.match_type = feature::BrutalForceMatcher<feature::SIFTDescriptor, feature::SIFTDescriptorList>::MatchType::TWO_WAY; 
    feature::BrutalForceMatcher<feature::SIFTDescriptor, feature::SIFTDescriptorList> matcher(matcher_config);
    matcher.match(features1.descriptors, features2.descriptors, matches); 
    std::cout << "Found " << matches.size() << " matches " << std::endl; 

    /* Draw features */
    // cv::Mat cv_img_out1, cv_img_out2; 
    // cv::Mat cv_img_match; 
    // cv::drawKeypoints(cv_img1, features1.key_points, cv_img_out1, 
    //     cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::drawKeypoints(cv_img2, features2.key_points, cv_img_out2, 
    //     cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::drawMatches(cv_img1, features1.key_points, 
    //                 cv_img2, features2.key_points, 
    //                 cv_matches, 
    //                 cv_img_match); 
    // cv::imshow("keypoints1", cv_img_out1); 
    // cv::imshow("keypoints2", cv_img_out2); 
    // cv::imshow("matches", cv_img_match); 
    // cv::waitKey(); 

    /* Estimate two-view geometry */
    EigenVec<Eigen::Vector2d> kpts1(features1.key_points.size()); 
    for (size_t i = 0; i < features1.key_points.size(); ++i) {
        kpts1[i] = Eigen::Vector2d(features1.key_points[i].point.x(), 
                                   features1.key_points[i].point.y()); 
        // std::cout << kpts1[i].transpose() << std::endl;
    }
    EigenVec<Eigen::Vector2d> kpts2(features2.key_points.size()); 
    for (size_t i = 0; i < features2.key_points.size(); ++i) {
        kpts2[i] = Eigen::Vector2d(features2.key_points[i].point.x(), 
                                   features2.key_points[i].point.y()); 
        // std::cout << kpts2[i].transpose() << std::endl;
    }
    EigenVec<Eigen::Vector2d> points_2D1, points_2D2; 
    EigenVec<Eigen::Vector2d> points_2DN1, points_2DN2; 
    std::vector<cv::Point2f> cv_points_2D1, cv_points_2D2;
    for (const auto &m : matches) {
        points_2D1.push_back(kpts1[m.query_index]); 
        points_2D2.push_back(kpts2[m.train_index]);
        points_2DN1.push_back(camera1->pixel2NormalizedIgnoringDistortion(kpts1[m.query_index])); 
        points_2DN2.push_back(camera2->pixel2NormalizedIgnoringDistortion(kpts2[m.train_index])); 
        cv_points_2D1.emplace_back(static_cast<float>(kpts1[m.query_index].x()), 
                                   static_cast<float>(kpts1[m.query_index].y()));
        cv_points_2D2.emplace_back(static_cast<float>(kpts2[m.train_index].x()), 
                                   static_cast<float>(kpts2[m.train_index].y()));
        // std::cout << kpts1[m.query_index].transpose() << " " << kpts2[m.train_index].transpose() << std::endl;
    }

    estimator::FundamentalMatrixEightPointEstimator f_estimator; 
    EigenVec<Eigen::Matrix3d> Fs; 
    f_estimator.estimate(points_2D1, points_2D2, Fs); 
    std::cout << "F w/o RANSAC: \n" << Fs[0] / Fs[0](2, 2) << std::endl; 

    estimator::EssentialMatrixEightPointEstimator e_estimator; 
    EigenVec<Eigen::Matrix3d> Es; 
    e_estimator.estimate(points_2DN1, points_2DN2, Es); 
    std::cout << "E w/o RANSAC: \n" << Es[0] / Es[0](2, 2) << std::endl;

    estimator::RANSACConfig ransac_config; 
    ransac_config.confidence = 0.99; 
    ransac_config.max_inlier_error = 1.0; 
    ransac_config.min_iter_num = 100; 
    std::cout << "------------------------" << std::endl;
    estimator::RANSAC<estimator::FundamentalMatrixEightPointEstimator> f_ransac_estimator(ransac_config); 
    const auto f_est_report = f_ransac_estimator.estimate(points_2D1, points_2D2); 
    if (f_est_report.success) {
        size_t num_inliers = 0; 
        for (const bool &is_inlier : f_est_report.inlier_mask) {
            if (is_inlier) {
                num_inliers++; 
            }
        }
        
        const double inlier_ratio = static_cast<double>(num_inliers) / static_cast<double>(points_2D1.size());
        std::cout << "F: \n" << f_est_report.model / f_est_report.model(2, 2) << std::endl;
        std::cout << "F inliers num: " << num_inliers << std::endl; 
        std::cout << "F inlier ratio: " << inlier_ratio << std::endl;
        std::cout << "F inlier error: " << f_est_report.inlier_error << std::endl; 
        std::cout << "iter num: " << f_est_report.num_iter << std::endl; 

        const Eigen::Matrix3d F = f_est_report.model; 
        const cv::Mat cv_F_mat = 
            (cv::Mat_<double>(3, 3) << F(0, 0), F(0, 1), F(0, 2), 
                                       F(1, 0), F(1, 1), F(1, 2), 
                                       F(2, 0), F(2, 1), F(2, 2));
        std::cout << "E from F using cv::Mat: \n" << cv_K.t() * cv_F_mat * cv_K << std::endl; 
    } else {
        std::cout << "F estimation failed" << std::endl;
    }
    cv::Mat cv_F =  
        cv::findFundamentalMat(cv_points_2D1, cv_points_2D2, cv::FM_RANSAC);
    std::cout << "cv_F: \n" << cv_F << std::endl;

    std::cout << "------------------------" << std::endl;
    estimator::RANSAC<estimator::HomographyEstimator> h_ransac_estimator(ransac_config); 
    const auto h_est_report = h_ransac_estimator.estimate(points_2D1, points_2D2); 
    if (h_est_report.success) {
        size_t num_inliers = 0; 
        for (const bool &is_inlier : h_est_report.inlier_mask) {
            if (is_inlier) {
                num_inliers++; 
            }
        }
        const double inlier_ratio = static_cast<double>(num_inliers) / static_cast<double>(points_2D1.size());
        std::cout << "H: \n" << h_est_report.model / h_est_report.model(2, 2) << std::endl;
        std::cout << "H inliers num: " << num_inliers << std::endl; 
        std::cout << "H inlier ratio: " << inlier_ratio << std::endl;
        std::cout << "H inlier error: " << h_est_report.inlier_error << std::endl; 
        std::cout << "iter num: " << h_est_report.num_iter << std::endl; 
    } else {
        std::cout << "H estimation failed" << std::endl;
    }
    cv::Mat cv_H = 
        cv::findHomography(cv_points_2D1, cv_points_2D2, cv::RANSAC);
    std::cout << "cv_H: \n" << cv_H << std::endl;
    
    std::cout << "------------------------" << std::endl;
    const Eigen::Matrix3d E_from_F = K2.transpose() * f_est_report.model * K1; 
    std::cout << "E from F: \n" << E_from_F / E_from_F(2, 2) << std::endl; 
    ransac_config.confidence = 0.999; 
    ransac_config.max_inlier_error = 0.5 * (ransac_config.max_inlier_error / K1(0, 0) + ransac_config.max_inlier_error / K2(0, 0)); 
    estimator::RANSAC<estimator::EssentialMatrixFivePointEstimator> e_ransac_estimator(ransac_config);
    const auto e_est_report = e_ransac_estimator.estimate(points_2DN1, points_2DN2); 
    if (e_est_report.success) {
        size_t num_inliers = 0; 
        for (const bool &is_inlier : e_est_report.inlier_mask) {
            if (is_inlier) {
                num_inliers++; 
            }
        }
        const double inlier_ratio = static_cast<double>(num_inliers) / static_cast<double>(points_2DN1.size());
        std::cout << "E: \n" << e_est_report.model / e_est_report.model(2, 2) << std::endl;
        std::cout << "E inliers num: " << num_inliers << std::endl; 
        std::cout << "E inlier ratio: " << inlier_ratio << std::endl;
        std::cout << "E inlier error: " << e_est_report.inlier_error << std::endl; 
        std::cout << "iter num: " << e_est_report.num_iter << std::endl; 

        Eigen::Matrix3d R_E; 
        Eigen::Vector3d t_E; 
        EigenVec<Eigen::Vector3d> ps3D_E;
        estimator::poseFromEssentialMatrix(e_est_report.model, 
            points_2DN1, points_2DN2, R_E, t_E, ps3D_E); 
        std::cout << "R_E: \n" << R_E << std::endl; 
        std::cout << "t_E: \n" << t_E << std::endl; 

        // const cv::Mat cv_E_copy = 
        //     (cv::Mat_<double>(3, 3) << e_est_report.model(0, 0), e_est_report.model(0, 1), e_est_report.model(0, 2), 
        //                                e_est_report.model(1, 0), e_est_report.model(1, 1), e_est_report.model(1, 2), 
        //                                e_est_report.model(2, 0), e_est_report.model(2, 1), e_est_report.model(2, 2)); 
        // cv::Mat cv_R_E, cv_t_E; 
        // cv::recoverPose(cv_E_copy, cv_points_2D1, cv_points_2D2, cv_K, cv_R_E, cv_t_E); 
        // std::cout << "cv_R_E: \n" << cv_R_E << std::endl; 
        // std::cout << "cv_t_E: \n" << cv_t_E << std::endl; 
    } else {
        std::cout << "E estimation failed" << std::endl;
    } 
    const cv::Mat cv_E_from_F = cv_K.t() * cv_F * cv_K;
    std::cout << "cv_E_from_F: \n" << cv_E_from_F / cv_E_from_F.ptr<double>(2, 2)[0] << std::endl;
    cv::Mat cv_E = 
        cv::findEssentialMat(cv_points_2D1, cv_points_2D2, cv_K); 
    std::cout << "cv_E: \n" << cv_E / cv_E.ptr<double>(2, 2)[0] << std::endl;
    cv::Mat cv_R, cv_t; 
    cv::recoverPose(cv_E, cv_points_2D1, cv_points_2D2, cv_K, cv_R, cv_t);
    std::cout << "cv_t: \n" << cv_t << std::endl; 
    std::cout << "cv_R: \n" << cv_R << std::endl;

    feature::PairWiseMatchingInfo pair_wise_matching_info; 
    pair_wise_matching_info.query_image_idx = 0; 
    pair_wise_matching_info.train_image_idx = 1; 
    pair_wise_matching_info.matches = matches; 
    base::Scene scene; 
    sfm::SceneBuilder::Config scene_builder_config; 
    scene_builder_config.ransac_config.max_inlier_error = 3.0; 
    scene_builder_config.ransac_config.confidence = 0.99; 
    scene_builder_config.ransac_config.min_iter_num = 0; 
    scene_builder_config.min_num_inliers = 10; 
    scene_builder_config.min_epsilon_EF = 0.8;
    sfm::SceneBuilder scene_builder(scene_builder_config); 
    feature::PairWiseMatchingInfo inliers; 
    base::Scene::Edge::SceneEdgeModelType model_type; 
    base::Scene::TwoViewGeometryInfo two_view_geom_info; 
    sfm::SceneBuilder::ImagePairVerificationReport veri_report = 
        scene_builder.verifyImagePair(image1, image2, 
                                     kpts1, kpts2, 
                                     pair_wise_matching_info, 
                                     model_type, 
                                     inliers, 
                                     &two_view_geom_info); 
    std::cout << two_view_geom_info.t.transpose() << " " << 
                 Eigen::Quaterniond(two_view_geom_info.R).coeffs().transpose() << std::endl; 
    {
                std::string edge_type_str; 
                switch (model_type)
                {
                case base::Scene::Edge::SceneEdgeModelType::INVALID:
                    edge_type_str = "INVALID"; 
                    break;
                case base::Scene::Edge::SceneEdgeModelType::UNDETERMINED:
                    edge_type_str = "UNDETERMINED"; 
                    break;
                case base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED:
                    edge_type_str = "GENERAL_AND_CALIBRATED"; 
                    break;
                case base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_UNCALIBRATED:
                    edge_type_str = "GENERAL_AND_UNCALIBRATED"; 
                    break;
                case base::Scene::Edge::SceneEdgeModelType::PANORAMIC:
                    edge_type_str = "PANORAMIC"; 
                    break;
                case base::Scene::Edge::SceneEdgeModelType::PLANNAR:
                    edge_type_str = "PLANNAR"; 
                    break;
                default:
                    edge_type_str = "INVALID"; 
                    break;
                }
                std::cout  << "\timage 1: " << image1.getName() << std::endl 
                          << "\timage 2: " << image2.getName() << std::endl 
                          << "\tedge type: " << edge_type_str << std::endl 
                          << "\tinlier number: " << inliers.matches.size() << std::endl 
                          << "\tepsilon EF: " << veri_report.epsilon_EF << std::endl 
                          << "\tepsilon HF: " << veri_report.epsilon_HF << std::endl 
                          << "\trelative pose from two-view-geometry: " << two_view_geom_info.t.transpose() << " | " << Eigen::Quaterniond(two_view_geom_info.R).coeffs().transpose() << std::endl 
                          << "\tverification information: " << std::endl 
                          << "\t\tfundamental matrix estimation time: " << veri_report.f_mat_est_time_secs << " secs" << std::endl 
                          << "\t\thomography estimation time: " << veri_report.h_mat_est_time_secs << " secs" << std::endl 
                          << "\t\tessential matrix estimation time: " << veri_report.e_mat_est_time_secs << " secs" << std::endl 
                          << "\t\ttriangulated point number: " << two_view_geom_info.triangulated_points.size() << std::endl 
                          << "\t\tmedian triangulation angle: " << util::rad2Deg(two_view_geom_info.median_triangulation_angle) << " deg(s)" << std::endl; 
    }
    std::cout << "F: \n" << two_view_geom_info.fundamental_matrix / two_view_geom_info.fundamental_matrix(2, 2) << std::endl; 
    std::cout << "H: \n" << two_view_geom_info.homography << std::endl; 
    std::cout << "E: \n" << two_view_geom_info.essential_matrix / two_view_geom_info.essential_matrix(2, 2) << std::endl; 
    std::cout << std::endl;
    std::cout << (-two_view_geom_info.R.inverse() * two_view_geom_info.t).transpose() << " | \n\n" << two_view_geom_info.R.inverse() << std::endl; 

    for (const auto &inlier_match : inliers.matches) {
        const Eigen::Vector2d point_2D1 = kpts1[inlier_match.query_index]; 
        const Eigen::Vector2d point_2D2 = kpts2[inlier_match.train_index]; 
        const Eigen::Matrix3x4d proj1 = K1 * Eigen::Matrix3x4d::Identity(); 
        const Eigen::Matrix3x4d proj2 = K2 * base::composePose(two_view_geom_info.R, two_view_geom_info.t);
        const Eigen::Vector3d tri_point_3D = 
            base::triangulatePoint(proj1, proj2, point_2D1, point_2D2); 
        const double tri_angle = 
            base::computeTriangulationAngle(Eigen::Vector3d::Zero(), -two_view_geom_info.R.inverse() * two_view_geom_info.t, 
                                            tri_point_3D); 
        std::cout << inlier_match.query_index << " " << inlier_match.train_index << " | " 
                  << point_2D1.transpose() << " | " << point_2D2.transpose() << " | " << tri_point_3D.transpose() << " | " << tri_angle << std::endl;
    }

    std::cout << "K1: \n" << K1 << std::endl; 
    std::cout << "K2: \n" << K2 << std::endl; 

    return 0; 
}
