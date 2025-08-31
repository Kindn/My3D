#define TEST_NAME "sfm/scene_builder"
#include "utils/unit_test.h"

#include "sfm/SceneBuilder.h"
#include "utils/RandomNumberGenerator.h"

using namespace my3d;

BOOST_AUTO_TEST_CASE(TestVerifyImagePairGeneralScene) {
  const int kNumPoints3D = 100;
  base::Image image1, image2;
  Eigen::Matrix3d K;
  K << 3205.9, 7.79215, 1971.26, 0, 3185.74, 1544.62, 0, 0, 1;
  base::RadialPinHoleCamera *camera1 = new base::RadialPinHoleCamera(K);
  base::RadialPinHoleCamera *camera2 = new base::RadialPinHoleCamera(K);
  image1.setCameraModel(camera1);
  image2.setCameraModel(camera2);

  Eigen::Matrix3d R1, R2;
  Eigen::Vector3d t1, t2;
  double theta = -M_PI / 4.0;
  R1.setIdentity();
  R2 << std::cos(theta), -std::sin(theta), 0, std::sin(theta), std::cos(theta),
      0, 0, 0, 1;
  t1.setZero();
  t2 << 0, 1, 0;

  // Genrate random points on a half cylinder;
  EigenVec<Eigen::Vector3d> points_3D;
  EigenVec<Eigen::Vector2d> points_2D1, points_2D2;
  feature::PairWiseMatchingInfo matching_info;
  util::RandomNumberGenerator rng;
  for (int i = 0; i < kNumPoints3D; ++i) {
    double t = rng.uniformReal(-M_PI / 2, M_PI / 2);
    double x = 2.0 - 0.8 * std::cos(t);
    double y = 0.8 * std::sin(t);
    double z = rng.uniformReal(-1.0, 1.0);
    Eigen::Vector3d point_3D = Eigen::Vector3d(x, y, z);
    Eigen::Vector3d point_3D_c1 = R1.transpose() * (point_3D - t1);
    Eigen::Vector3d point_3D_c2 = R1.transpose() * (point_3D - t2);
    Eigen::Vector2d point_2D1 = camera1->project(point_3D_c1);
    Eigen::Vector2d point_2D2 = camera1->project(point_3D_c2);
    points_3D.push_back(point_3D);
    points_2D1.push_back(point_2D1);
    points_2D2.push_back(point_2D2);
    matching_info.matches.push_back(feature::Match{i, i, 0});
  }

  sfm::SceneBuilder::Config config;
  sfm::SceneBuilder builder(config);
  base::Scene::Edge::SceneEdgeModelType model_type;
  feature::PairWiseMatchingInfo inliers;
  auto veri_report =
      builder.verifyImagePair(image1, image2, points_2D1, points_2D2,
                              matching_info, model_type, inliers);

  BOOST_CHECK(veri_report.success);
  BOOST_CHECK_EQUAL(
      model_type,
      base::Scene::Edge::SceneEdgeModelType::GENERAL_AND_CALIBRATED);
  BOOST_CHECK_EQUAL(inliers.matches.size(), matching_info.matches.size());
}

BOOST_AUTO_TEST_CASE(TestVerifyImagePairPanoramicScene) {
  const int kNumPoints3D = 100;
  base::Image image1, image2;
  Eigen::Matrix3d K;
  K << 3205.9, 7.79215, 1971.26, 0, 3185.74, 1544.62, 0, 0, 1;
  base::RadialPinHoleCamera *camera1 = new base::RadialPinHoleCamera(K);
  base::RadialPinHoleCamera *camera2 = new base::RadialPinHoleCamera(K);
  image1.setCameraModel(camera1);
  image2.setCameraModel(camera2);

  Eigen::Matrix3d R1, R2;
  Eigen::Vector3d t1, t2;
  double theta = -M_PI / 4.0;
  R1.setIdentity();
  R2 << std::cos(theta), -std::sin(theta), 0, std::sin(theta), std::cos(theta),
      0, 0, 0, 1;
  t1.setZero();
  t2 << 0, 0.01, 0;

  // Genrate random points on a half cylinder;
  EigenVec<Eigen::Vector3d> points_3D;
  EigenVec<Eigen::Vector2d> points_2D1, points_2D2;
  feature::PairWiseMatchingInfo matching_info;
  util::RandomNumberGenerator rng;
  for (int i = 0; i < kNumPoints3D; ++i) {
    double t = rng.uniformReal(-M_PI / 2, M_PI / 2);
    double x = 2.0 - 0.8 * std::cos(t);
    double y = 0.8 * std::sin(t);
    double z = rng.uniformReal(-1.0, 1.0);
    Eigen::Vector3d point_3D = Eigen::Vector3d(x, y, z);
    Eigen::Vector3d point_3D_c1 = R1.transpose() * (point_3D - t1);
    Eigen::Vector3d point_3D_c2 = R1.transpose() * (point_3D - t2);
    Eigen::Vector2d point_2D1 = camera1->project(point_3D_c1);
    Eigen::Vector2d point_2D2 = camera1->project(point_3D_c2);
    points_3D.push_back(point_3D);
    points_2D1.push_back(point_2D1);
    points_2D2.push_back(point_2D2);
    matching_info.matches.push_back(feature::Match{i, i, 0});
  }

  sfm::SceneBuilder::Config config;
  sfm::SceneBuilder builder(config);
  base::Scene::Edge::SceneEdgeModelType model_type;
  feature::PairWiseMatchingInfo inliers;
  auto veri_report =
      builder.verifyImagePair(image1, image2, points_2D1, points_2D2,
                              matching_info, model_type, inliers);

  BOOST_CHECK(veri_report.success);
  BOOST_CHECK_EQUAL(model_type,
                    base::Scene::Edge::SceneEdgeModelType::PANORAMIC);
  // BOOST_CHECK_EQUAL(inliers.matches.size(), matching_info.matches.size());
}

BOOST_AUTO_TEST_CASE(TestVerifyImagePairPlannarScene) {
  const int kNumPoints3D = 100;
  base::Image image1, image2;
  Eigen::Matrix3d K;
  K << 3205.9, 7.79215, 1971.26, 0, 3185.74, 1544.62, 0, 0, 1;
  base::RadialPinHoleCamera *camera1 = new base::RadialPinHoleCamera(K);
  base::RadialPinHoleCamera *camera2 = new base::RadialPinHoleCamera(K);
  image1.setCameraModel(camera1);
  image2.setCameraModel(camera2);

  Eigen::Matrix3d R1, R2;
  Eigen::Vector3d t1, t2;
  double theta = -M_PI / 4.0;
  R1.setIdentity();
  R2 << std::cos(theta), -std::sin(theta), 0, std::sin(theta), std::cos(theta),
      0, 0, 0, 1;
  t1.setZero();
  t2 << 0, 1, 0;

  // Genrate random points on a plane z = -2(x - 2);
  EigenVec<Eigen::Vector3d> points_3D;
  EigenVec<Eigen::Vector2d> points_2D1, points_2D2;
  feature::PairWiseMatchingInfo matching_info;
  util::RandomNumberGenerator rng;
  for (int i = 0; i < kNumPoints3D; ++i) {
    double x = rng.uniformReal(1.5, 2.5);
    double y = rng.uniformReal(-1.0, 2.0);
    double z = -2.0 * (x - 2.0);
    Eigen::Vector3d point_3D = Eigen::Vector3d(x, y, z);
    Eigen::Vector3d point_3D_c1 = R1.transpose() * (point_3D - t1);
    Eigen::Vector3d point_3D_c2 = R1.transpose() * (point_3D - t2);
    Eigen::Vector2d point_2D1 = camera1->project(point_3D_c1);
    Eigen::Vector2d point_2D2 = camera1->project(point_3D_c2);
    points_3D.push_back(point_3D);
    points_2D1.push_back(point_2D1);
    points_2D2.push_back(point_2D2);
    matching_info.matches.push_back(feature::Match{i, i, 0});
  }

  sfm::SceneBuilder::Config config;
  sfm::SceneBuilder builder(config);
  base::Scene::Edge::SceneEdgeModelType model_type;
  feature::PairWiseMatchingInfo inliers;
  auto veri_report =
      builder.verifyImagePair(image1, image2, points_2D1, points_2D2,
                              matching_info, model_type, inliers);

  BOOST_CHECK(veri_report.success);
  BOOST_CHECK_EQUAL(model_type, base::Scene::Edge::SceneEdgeModelType::PLANNAR);
  BOOST_CHECK_EQUAL(inliers.matches.size(), matching_info.matches.size());
}
