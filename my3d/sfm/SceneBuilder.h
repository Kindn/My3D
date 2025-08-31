/*
 * filename: SceneBuilder.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_SFM_SCENE_BUILDER_H_
#define _MY3D_SFM_SCENE_BUILDER_H_

#include <iostream>
#include <vector>

#include "base/camera/RadialPinHoleCamera.h"
#include "base/image/Image.h"
#include "base/image/ImageReaderBase.h"
#include "base/reconstruction/Scene.h"
#include "base/reconstruction/Track.h"
#include "estimators/EssentialMatrixEstimator.h"
#include "estimators/FundamentalMatrixEstimator.h"
#include "estimators/HomographyEstimator.h"
#include "estimators/LORANSAC.h"
#include "feature/FeatureExtractorBase.h"
#include "feature/FeatureMatcherBase.h"
#include "utils/TicToc.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace sfm {

/**
 * Given a list of images and their pair-wise matchings, this class will build
 * a scene graph.
 */
class SceneBuilder {
public:
  struct Config {
    /* The maximum inlier ratio of homography of a general pair */
    double max_H_inlier_ratio{0.8};
    /* The maximum value of the ratio of the homography inliers to the
       fundamental matrix inliers in general pair */
    double max_epsilon_HF{0.6};
    /* The minimum value of the ratio of the essential matrix inliers to
       the fundamental matrix inliers in correctly calibrated pair */
    double min_epsilon_EF{0.95};
    /* Minimum triangulation angle (in degree) of moving camera. This parameter
       is used to distinguish between pure rotation (panoramic) and plannar
       scene (plannar) for non-genera scene */
    double min_triangulation_angle_deg{1.0};
    /* Minimum inlier ratio of a valid image pair */
    double min_inlier_ratio{0.5};
    /* Mininum inlier number of a valid image pair */
    uint32_t min_num_inliers{100};
    /* Minimum matching distance */
    double max_match_dist{0.4};

    estimator::RANSACConfig ransac_config;
  };

  struct ImagePairVerificationReport {
    double match_inlier_ratio{-1.0};
    bool success{false};
    double homography_inilier_ratio;
    double epsilon_HE;
    double epsilon_EF;
    double epsilon_HF;
    double f_mat_est_time_secs{-1.0};
    double h_mat_est_time_secs{-1.0};
    double e_mat_est_time_secs{-1.0};
    double inlier_ratio_f{-1.0};
    double inlier_ratio_h{-1.0};
    double inlier_ratio_e{-1.0};
  };

public:
  explicit SceneBuilder(const Config &config) : config_{config} {}

  bool
  build(const std::vector<base::Image> &images,
        const EigenVec<EigenVec<Eigen::Vector2d>> &key_points_list,
        const std::vector<feature::PairWiseMatchingInfo> &pair_wise_matchings,
        base::Scene &scene);

  ImagePairVerificationReport verifyImagePair(
      const base::Image &image1, const base::Image &image2,
      const EigenVec<Eigen::Vector2d> &key_points1,
      const EigenVec<Eigen::Vector2d> &key_points2,
      const feature::PairWiseMatchingInfo &pair_wise_matching,
      base::Scene::Edge::SceneEdgeModelType &model_type,
      feature::PairWiseMatchingInfo &inliers,
      base::Scene::TwoViewGeometryInfo *two_view_geom_info = nullptr);

  bool estimateRelativePose(
      const EigenVec<Eigen::Vector2d> &essential_inliers1,
      const EigenVec<Eigen::Vector2d> &essential_inliers2,
      const Eigen::Matrix3d &E,
      const EigenVec<Eigen::Vector2d> &homography_inliers1,
      const EigenVec<Eigen::Vector2d> &homography_inliers2,
      const Eigen::Matrix3d &H, const Eigen::Matrix3d &K1,
      const Eigen::Matrix3d &K2,
      base::Scene::Edge::SceneEdgeModelType &model_type, Eigen::Matrix3d &R,
      Eigen::Vector3d &t, EigenVec<Eigen::Vector3d> &tri_points,
      std::vector<double> &tri_angles, double &median_tri_angle);

private:
  Config config_;
};

} // namespace sfm
} // namespace my3d

#endif // _MY3D_SFM_SCENE_BUILDER_H_
