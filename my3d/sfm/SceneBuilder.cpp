/*
 * filename: SceneBuilder.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "sfm/SceneBuilder.h"

namespace my3d {
namespace sfm {

bool SceneBuilder::build(
    const std::vector<base::Image> &images,
    const EigenVec<EigenVec<Eigen::Vector2d>> &key_points_list,
    const std::vector<feature::PairWiseMatchingInfo> &pair_wise_matchings,
    base::Scene &scene) {
  const size_t num_images = images.size();
  assert(key_points_list.size() == num_images);

  scene.clear();
  /* First add all the images as nodes to the scene */
  for (size_t i = 0; i < num_images; ++i) {
    const EigenVec<Eigen::Vector2d> &key_points = key_points_list[i];
    EigenVec<base::Point2D> points_2D;
    // Normalize each key-point
    for (size_t j = 0; j < key_points.size(); ++j) {
      const Eigen::Vector2d position(key_points[j].x(), key_points[j].y());
      const Eigen::Matrix<uint8_t, 3, 1> color_bgr = images[i].getChannel(
          static_cast<size_t>(position.y()), static_cast<size_t>(position.x()));
      points_2D.emplace_back(
          position, Eigen::Matrix<uint8_t, 3, 1>(color_bgr.z(), color_bgr.y(),
                                                 color_bgr.x()));
    }
    base::Scene::CameraInfo camera_info;
    const auto camera_model =
        std::dynamic_pointer_cast<base::RadialPinHoleCamera>(
            images[i].getCameraModel());
    camera_info.fx = camera_model->fx_;
    camera_info.fy = camera_model->fy_;
    camera_info.cx = camera_model->cx_;
    camera_info.cy = camera_model->cy_;
    camera_info.alpha = camera_model->alpha_;
    camera_info.k1 = camera_model->k1_;
    camera_info.k2 = camera_model->k2_;
    // The node id equals the corresponding image's idx in the vector
    bool success = scene.addNode(i, images[i].cols(), images[i].rows(),
                                 points_2D, images[i].getName(), camera_info);
    if (!success) {
      return false;
    }
  }

  /* Then conducting geometric verifications for each image pair. If
     a pair is valid, add the corresponding edge to the scene. */
  size_t edge_id = 0;
  for (auto &matching : pair_wise_matchings) {
    const base::Image &image1 = images[matching.query_image_idx];
    const base::Image &image2 = images[matching.train_image_idx];
    const EigenVec<Eigen::Vector2d> &key_points1 =
        key_points_list[matching.query_image_idx];
    const EigenVec<Eigen::Vector2d> &key_points2 =
        key_points_list[matching.train_image_idx];

    base::Scene::Edge::SceneEdgeModelType model_type;
    feature::PairWiseMatchingInfo inliers;
    base::Scene::TwoViewGeometryInfo two_view_geom_info;
    ImagePairVerificationReport veri_report =
        verifyImagePair(image1, image2, key_points1, key_points2, matching,
                        model_type, inliers, &two_view_geom_info);
    if (veri_report.success &&
        inliers.matches.size() >= config_.min_num_inliers &&
        model_type != base::Scene::Edge::SceneEdgeModelType::INVALID) {
      std::vector<std::pair<size_t, size_t>> correspondences;
      for (auto &inlier : inliers.matches) {
        correspondences.push_back(
            std::make_pair(inlier.query_index, inlier.train_index));
      }
      bool success = scene.addEdge(
          edge_id,
          std::array<size_t, 2>{static_cast<size_t>(matching.query_image_idx),
                                static_cast<size_t>(matching.train_image_idx)},
          correspondences, model_type, two_view_geom_info);

      {
        std::string edge_type_str;
        switch (model_type) {
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
        case base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR:
          edge_type_str = "PANORAMIC_OR_PLANNAR";
          break;
        default:
          edge_type_str = "INVALID";
          break;
        }
        std::cout
            << "[SceneBuilder] Added edge " << edge_id << std::endl
            << "\timage 1: " << image1.getName()
            << "; id = " << matching.query_image_idx << std::endl
            << "\timage 2: " << image2.getName()
            << "; id = " << matching.train_image_idx << std::endl
            << "\tedge type: " << edge_type_str << std::endl
            << "\tmatch inlier ratio: " << veri_report.match_inlier_ratio
            << std::endl
            << "\tcorrepondence number: " << correspondences.size() << std::endl
            << "\tepsilon EF: " << veri_report.epsilon_EF << std::endl
            << "\tepsilon HF: " << veri_report.epsilon_HF << std::endl
            << "\tH inlier ratio: " << veri_report.homography_inilier_ratio
            << std::endl
            << "\trelative pose from two-view-geometry: "
            << two_view_geom_info.t.transpose() << " | "
            << Eigen::Quaterniond(two_view_geom_info.R).coeffs().transpose()
            << std::endl
            << "\ttriangulated "
            << two_view_geom_info.triangulated_points.size() << " points "
            << std::endl
            << "\tmedian triangulation angle: "
            << util::rad2Deg(two_view_geom_info.median_triangulation_angle)
            << " deg(s) " << std::endl
            << "\tverification information: " << std::endl
            << "\t\tfundamental matrix estimation time: "
            << veri_report.f_mat_est_time_secs << " secs" << std::endl
            << "\t\thomography estimation time: "
            << veri_report.h_mat_est_time_secs << " secs" << std::endl
            << "\t\tessential matrix estimation time: "
            << veri_report.e_mat_est_time_secs << " secs" << std::endl
            << "\t\tinlier ratio of fundamental matrix estimation: "
            << veri_report.inlier_ratio_f << std::endl
            << "\t\tinlier ratio of homography estimation: "
            << veri_report.inlier_ratio_h << std::endl
            << "\t\tinlier ratio of essential matrix estimation: "
            << veri_report.inlier_ratio_e << std::endl;
      }

      if (!success) {
        return false;
      } else {
        ++edge_id;
      }
    }
  }

  /* Remove the node(s) with no edge connected to it/them */
  scene.removeIsolatedNodes();

  return true;
}

SceneBuilder::ImagePairVerificationReport SceneBuilder::verifyImagePair(
    const base::Image &image1, const base::Image &image2,
    const EigenVec<Eigen::Vector2d> &key_points1,
    const EigenVec<Eigen::Vector2d> &key_points2,
    const feature::PairWiseMatchingInfo &pair_wise_matching,
    base::Scene::Edge::SceneEdgeModelType &model_type,
    feature::PairWiseMatchingInfo &inliers,
    base::Scene::TwoViewGeometryInfo *two_view_geom_info) {
  ImagePairVerificationReport veri_report;
  util::TicToc tic_toc;

  inliers.query_image_idx = pair_wise_matching.query_image_idx;
  inliers.train_image_idx = pair_wise_matching.train_image_idx;
  inliers.matches.clear();
  /* Check if there are enough matchings for verification. If not, return false.
   */
  const size_t num_matches = pair_wise_matching.matches.size();
  if (num_matches < config_.min_num_inliers ||
      num_matches <
          estimator::FundamentalMatrixSevenPointEstimator::kMinNumSamples ||
      num_matches <
          estimator::FundamentalMatrixEightPointEstimator::kMinNumSamples ||
      num_matches < estimator::HomographyEstimator::kMinNumSamples ||
      num_matches <
          estimator::EssentialMatrixEightPointEstimator::kMinNumSamples) {
    model_type = base::Scene::Edge::SceneEdgeModelType::UNDETERMINED;
    // inliers = pair_wise_matching;
    veri_report.success = false;
    // veri_report.success = true;
    return veri_report;
  }

  typedef estimator::LORANSAC<estimator::FundamentalMatrixSevenPointEstimator,
                              estimator::FundamentalMatrixEightPointEstimator>
      FundamentalEstimator;
  typedef estimator::LORANSAC<estimator::HomographyEstimator,
                              estimator::HomographyEstimator>
      HomographyEstimator;
  typedef estimator::LORANSAC<estimator::EssentialMatrixFivePointEstimator,
                              estimator::EssentialMatrixFivePointEstimator>
      EssentialEstimator;
  // const size_t kf =
  // estimator::FundamentalMatrixEightPointEstimator::kMinNumSamples; const
  // size_t kh = estimator::HomographyEstimator::kMinNumSamples; const size_t ke
  // = estimator::EssentialMatrixEightPointEstimator::kMinNumSamples;

  //* Prepare data for transform estimation
  // Filter matches with too large distances
  feature::Matches close_matches = pair_wise_matching.matches;
  // for (const auto &m : pair_wise_matching.matches) {
  //     if (m.distance <= config_.max_match_dist) {
  //         close_matches.push_back(m);
  //     }
  // }
  // Extract inlier matches
  const feature::Matches inlier_matches = feature::getInlierMatches(
      key_points1, key_points2, close_matches, config_.ransac_config);
  EigenVec<Eigen::Vector2d> points1(inlier_matches.size());
  EigenVec<Eigen::Vector2d> points2(inlier_matches.size());
  for (size_t i = 0; i < inlier_matches.size(); ++i) {
    points1[i] = key_points1[inlier_matches[i].query_index];
    points2[i] = key_points2[inlier_matches[i].train_index];
  }

  veri_report.match_inlier_ratio = static_cast<double>(inlier_matches.size()) /
                                   static_cast<double>(num_matches);

  if (inlier_matches.size() < config_.min_num_inliers &&
      veri_report.match_inlier_ratio < config_.min_inlier_ratio) {
    model_type = base::Scene::Edge::SceneEdgeModelType::INVALID;
    veri_report.success = false;
  }

  /* Estimate the fundamental matrix and find the inliers. This step
      will determine whether the image pair is valid */
  feature::PairWiseMatchingInfo inliers_fundamental;
  Eigen::Matrix3d fundamental_matrix;
  inliers_fundamental.query_image_idx = pair_wise_matching.query_image_idx;
  inliers_fundamental.train_image_idx = pair_wise_matching.train_image_idx;
  inliers_fundamental.matches.clear();
  {
    estimator::RANSACConfig f_mat_ransac_config = config_.ransac_config;
    // f_mat_ransac_config.max_iter_num =
    // std::max(2 * util::computeCombinationNumber(points1.size(), kf),
    //                                             f_mat_ransac_config.min_iter_num);
    FundamentalEstimator estimator(f_mat_ransac_config);
    tic_toc.tic();
    auto report = estimator.estimate(points1, points2);
    // std::cout << "f" << std::endl;
    veri_report.f_mat_est_time_secs = tic_toc.toc();
    if (!report.success) {
      // model_type = base::Scene::Edge::SceneEdgeModelType::UNDETERMINED;
      // inliers = pair_wise_matching;
      veri_report.success = false;
      // veri_report.success = true;
      return veri_report;
    }
    for (size_t i = 0; i < inlier_matches.size(); ++i) {
      if (report.inlier_mask[i]) {
        inliers_fundamental.matches.push_back(inlier_matches[i]);
      }
    }
    fundamental_matrix = report.model;
    veri_report.inlier_ratio_f =
        static_cast<double>(inliers_fundamental.matches.size()) /
        static_cast<double>(inlier_matches.size());
    if (two_view_geom_info) {
      two_view_geom_info->fundamental_matrix = fundamental_matrix;
    }
  }

  /* Determine the model type */
  //* Estimate the homography and determine whether the model type is general
  feature::PairWiseMatchingInfo inliers_homography;
  EigenVec<Eigen::Vector2d> inlier_h_points1, inlier_h_points2;
  Eigen::Matrix3d homography;
  inliers_homography.query_image_idx = pair_wise_matching.query_image_idx;
  inliers_homography.train_image_idx = pair_wise_matching.train_image_idx;
  inliers_homography.matches.clear();
  {
    estimator::RANSACConfig h_mat_ransac_config = config_.ransac_config;
    // h_mat_ransac_config.max_iter_num = std::max(2 *
    // util::computeCombinationNumber(points1.size(), kh),
    //                                             h_mat_ransac_config.min_iter_num);
    HomographyEstimator estimator(h_mat_ransac_config);
    tic_toc.tic();
    const auto report = estimator.estimate(points1, points2);
    // std::cout << "h" << std::endl;
    veri_report.h_mat_est_time_secs = tic_toc.toc();
    if (!report.success) {
      // model_type = base::Scene::Edge::SceneEdgeModelType::UNDETERMINED;
      // inliers = pair_wise_matching;
      veri_report.success = false;
      // veri_report.success = true;
      return veri_report;
    }
    for (size_t i = 0; i < inlier_matches.size(); ++i) {
      if (report.inlier_mask[i]) {
        inliers_homography.matches.push_back(inlier_matches[i]);
        inlier_h_points1.push_back(points1[i]);
        inlier_h_points2.push_back(points2[i]);
      }
    }
    homography = report.model;
    veri_report.inlier_ratio_h =
        static_cast<double>(inliers_homography.matches.size()) /
        static_cast<double>(inlier_matches.size());
    if (two_view_geom_info) {
      two_view_geom_info->homography = homography;
    }
  }

  if (inliers_fundamental.matches.size() < config_.min_num_inliers) {
    model_type = base::Scene::Edge::SceneEdgeModelType::UNDETERMINED;
    inliers = inliers_homography;
    veri_report.success = true;
    // veri_report.success = true;
    return veri_report;
  }

  const double epsilon_HF =
      static_cast<double>(inliers_homography.matches.size()) /
      static_cast<double>(inliers_fundamental.matches.size());
  const double homography_inlier_ratio =
      static_cast<double>(inliers_homography.matches.size()) /
      static_cast<double>(inlier_matches.size());
  veri_report.epsilon_HF = epsilon_HF;
  veri_report.homography_inilier_ratio = homography_inlier_ratio;
  feature::PairWiseMatchingInfo inliers_essential;
  inliers_essential.query_image_idx = pair_wise_matching.query_image_idx;
  inliers_essential.train_image_idx = pair_wise_matching.train_image_idx;
  inliers_essential.matches.clear();
  //* Compute the essential matrix using the calibration information and
  //determine
  //* whether the image pair is well calibrated
  const std::shared_ptr<base::RadialPinHoleCamera> camera_model1 =
      std::dynamic_pointer_cast<base::RadialPinHoleCamera>(
          image1.getCameraModel());
  const std::shared_ptr<base::RadialPinHoleCamera> camera_model2 =
      std::dynamic_pointer_cast<base::RadialPinHoleCamera>(
          image2.getCameraModel());
  const Eigen::Matrix3d K1 = camera_model1->getCameraMatrix();
  const Eigen::Matrix3d K2 = camera_model2->getCameraMatrix();
  Eigen::Matrix3d essential_matrix;
  //* Find inliers of essential matrix
  EigenVec<Eigen::Vector2d> points_2DC1, points_2DC2;
  EigenVec<Eigen::Vector2d> inlier_points_2DC1, inlier_points_2DC2;
  {
    for (size_t i = 0; i < points1.size(); ++i) {
      points_2DC1.push_back(
          camera_model1->pixel2NormalizedIgnoringDistortion(points1[i]));
      points_2DC2.push_back(
          camera_model2->pixel2NormalizedIgnoringDistortion(points2[i]));
    }
    estimator::RANSACConfig e_mat_ransac_config = config_.ransac_config;
    e_mat_ransac_config.max_inlier_error =
        0.5 * (camera_model1->thresholdPix2Norm(
                   config_.ransac_config.max_inlier_error) +
               camera_model2->thresholdPix2Norm(
                   config_.ransac_config.max_inlier_error));

    EssentialEstimator estimator(e_mat_ransac_config);
    tic_toc.tic();
    auto report = estimator.estimate(points_2DC1, points_2DC2);
    // std::cout << "e" << std::endl;
    veri_report.e_mat_est_time_secs = tic_toc.toc();
    if (!report.success) {
      // model_type = base::Scene::Edge::SceneEdgeModelType::INVALID;
      // inliers = pair_wise_matching;
      veri_report.success = false;
      // veri_report.success = true;
      return veri_report;
    }
    essential_matrix = report.model;
    for (size_t i = 0; i < inlier_matches.size(); ++i) {
      if (report.inlier_mask[i]) {
        inliers_essential.matches.push_back(inlier_matches[i]);
        inlier_points_2DC1.push_back(points_2DC1[i]);
        inlier_points_2DC2.push_back(points_2DC2[i]);
      }
    }
    veri_report.inlier_ratio_e =
        static_cast<double>(inliers_essential.matches.size()) /
        static_cast<double>(inlier_matches.size());
    if (two_view_geom_info) {
      two_view_geom_info->essential_matrix = essential_matrix;
    }
  }

  const double epsilon_HE =
      static_cast<double>(inliers_homography.matches.size()) /
      static_cast<double>(inliers_essential.matches.size());
  veri_report.epsilon_HE = epsilon_HE;

  const double epsilon_EF =
      static_cast<double>(inliers_essential.matches.size()) /
      static_cast<double>(inliers_fundamental.matches.size());
  veri_report.epsilon_EF = epsilon_EF;

  if (epsilon_EF >= config_.min_epsilon_EF &&
      inliers_essential.matches.size() >=
          config_.min_num_inliers) { //*  This indicates that the image pair is
                                     //well calibrated
    // If the image pair is well calibrated, we decompose the essential matrix,
    // triangulate points from inlier correspondences, and determine the median
    // triangulation angle α_m. Using α_m, we distinguish between the case of
    // pure rotation (panoramic) and planar scenes (plannar)
    if (inliers_essential.matches.size() >=
        inliers_fundamental.matches.size()) {
      inliers = inliers_essential;
    } else {
      inliers = inliers_fundamental;
    }

    if (epsilon_HE > config_.max_H_inlier_ratio) {
      model_type = base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR;
      if (inliers_homography.matches.size() > inliers.matches.size()) {
        inliers = inliers_homography;
      }
    } else {
      model_type =
          base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED;
    }
  } else if (inliers_fundamental.matches.size() >= config_.min_num_inliers) {
    inliers = inliers_fundamental;

    if (epsilon_HF > config_.max_H_inlier_ratio) {
      model_type = base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR;
      if (inliers_homography.matches.size() > inliers.matches.size()) {
        inliers = inliers_homography;
      }
    } else {
      model_type =
          base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_UNCALIBRATED;
    }
  } else if (inliers_homography.matches.size() >= config_.min_num_inliers) {
    inliers = inliers_homography;
    model_type = base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR;
  } else {
    model_type = base::Scene::Edge::SceneEdgeModelType::INVALID;
    veri_report.success = true;
    return veri_report;
  }

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  double median_alpha;
  EigenVec<Eigen::Vector3d> points_3D;
  std::vector<double> alphas;

  if (!estimateRelativePose(inlier_points_2DC1, inlier_points_2DC2,
                            essential_matrix, inlier_h_points1,
                            inlier_h_points2, homography, K1, K2, model_type, R,
                            t, points_3D, alphas, median_alpha)) {
    model_type = base::Scene::Edge::SceneEdgeModelType::INVALID;
    veri_report.success = true;
    return veri_report;
  }
  if (two_view_geom_info) {
    two_view_geom_info->triangulated_points = points_3D;
    two_view_geom_info->triangulation_angles = alphas;
    two_view_geom_info->median_triangulation_angle = median_alpha;
    two_view_geom_info->R = R;
    two_view_geom_info->t = t;
  }

  veri_report.success = true;

  return veri_report;
}

bool SceneBuilder::estimateRelativePose(
    const EigenVec<Eigen::Vector2d> &essential_inliers1,
    const EigenVec<Eigen::Vector2d> &essential_inliers2,
    const Eigen::Matrix3d &E,
    const EigenVec<Eigen::Vector2d> &homography_inliers1,
    const EigenVec<Eigen::Vector2d> &homography_inliers2,
    const Eigen::Matrix3d &H, const Eigen::Matrix3d &K1,
    const Eigen::Matrix3d &K2,
    base::Scene::Edge::SceneEdgeModelType &model_type, Eigen::Matrix3d &R,
    Eigen::Vector3d &t, EigenVec<Eigen::Vector3d> &tri_points,
    std::vector<double> &tri_angles, double &median_tri_angle) {
  tri_points.clear();
  tri_angles.clear();
  if (model_type ==
          base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED ||
      model_type ==
          base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_UNCALIBRATED) {
    estimator::poseFromEssentialMatrix(E, essential_inliers1,
                                       essential_inliers2, R, t, tri_points);
  } else if (model_type ==
                 base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR ||
             model_type == base::Scene::Edge::SceneEdgeModelType::PANORAMIC ||
             model_type == base::Scene::Edge::SceneEdgeModelType::PLANNAR) {
    Eigen::Vector3d normal;
    estimator::poseFromHomography(H, K1, K2, homography_inliers1,
                                  homography_inliers2, R, t, normal,
                                  tri_points);
  } else {
    return false;
  }

  if (tri_points.empty()) {
    return false;
  }

  //* Compute the median triangulation angle
  tri_angles.resize(tri_points.size());
  for (size_t i = 0; i < tri_points.size(); ++i) {
    tri_angles[i] = base::computeTriangulationAngle(
        Eigen::Vector3d::Zero(), -R.transpose() * t, tri_points[i]);
  }
  std::sort(tri_angles.begin(), tri_angles.end());

  if (tri_angles.size() % 2 == 0) {
    median_tri_angle = 0.5 * (tri_angles[tri_angles.size() / 2 - 1] +
                              tri_angles[tri_angles.size() / 2]);
  } else {
    median_tri_angle = tri_angles[tri_angles.size() / 2];
  }

  if (model_type ==
      base::Scene::Edge::SceneEdgeModelType::PANORAMIC_OR_PLANNAR) {
    if (t.norm() == 0) {
      model_type = base::Scene::Edge::SceneEdgeModelType::PANORAMIC;
    } else {
      model_type = base::Scene::Edge::SceneEdgeModelType::PLANNAR;
    }
  }

  return true;
}

} // namespace sfm
} // namespace my3d
