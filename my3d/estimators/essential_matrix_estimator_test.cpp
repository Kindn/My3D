#define TEST_NAME "estimators/essential_matrix_estimator_test"
#include "utils/unit_test.h"

#include "estimators/EssentialMatrixEstimator.h" 
#include "estimators/RANSAC.h"
#include "utils/RandomNumberGenerator.h"

using namespace my3d;

BOOST_AUTO_TEST_CASE(TestEightPoint) {
    const double points1_raw[] = {1.839035, 1.924743, 0.543582,  0.375221,
                                    0.473240, 0.142522, 0.964910,  0.598376,
                                    0.102388, 0.140092, 15.994343, 9.622164,
                                    0.285901, 0.430055, 0.091150,  0.254594};

    const double points2_raw[] = {
        1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134,
        0.293328, 0.588992, 0.839509, 0.087290, 1.779735, 1.116857,
        0.878616, 0.602447, 0.642616, 1.028681,
    };

    const size_t kNumPoints = 8;
    EigenVec<Eigen::Vector2d> points1(kNumPoints);
    EigenVec<Eigen::Vector2d> points2(kNumPoints);
    for (size_t i = 0; i < kNumPoints; ++i) {
        points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
        points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
    }

    estimator::EssentialMatrixEightPointEstimator estimator;
    EigenVec<Eigen::Matrix3d> models;
    estimator.estimate(points1, points2, models);
    auto E = models[0];
    std::cout << "estimated E: \n" << E << std::endl;

    // Reference values.
    BOOST_CHECK(std::abs(E(0, 0) - -0.0368602) < 1e-5);
    BOOST_CHECK(std::abs(E(0, 1) - 0.265019) < 1e-5);
    BOOST_CHECK(std::abs(E(0, 2) - -0.0625948) < 1e-5);
    BOOST_CHECK(std::abs(E(1, 0) - -0.299679) < 1e-5);
    BOOST_CHECK(std::abs(E(1, 1) - -0.110667) < 1e-5);
    BOOST_CHECK(std::abs(E(1, 2) - 0.147114) < 1e-5);
    BOOST_CHECK(std::abs(E(2, 0) - 0.169381) < 1e-5);
    BOOST_CHECK(std::abs(E(2, 1) - -0.21072) < 1e-5);
    BOOST_CHECK(std::abs(E(2, 2) - -0.00401306) < 1e-5);

    // Check that the internal constraint is satisfied (two singular values equal
    // and one zero).
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
    Eigen::Vector3d s = svd.singularValues();
    BOOST_CHECK(std::abs(s(0) - s(1)) < 1e-5);
    BOOST_CHECK(std::abs(s(2)) < 1e-5);
}

BOOST_AUTO_TEST_CASE(TestFivePoint) {
    const double points1_raw[] = {
      0.4964, 1.0577, 0.3650,  -0.0919, -0.5412, 0.0159, -0.5239, 0.9467,
      0.3467, 0.5301, 0.2797,  0.0012,  -0.1986, 0.0460, -0.1622, 0.5347,
      0.0796, 0.2379, -0.3946, 0.7969,  0.2,     0.7,    0.6,     0.3};

  const double points2_raw[] = {
      0.7570, 2.7340, 0.3961,  0.6981, -0.6014, 0.7110, -0.7385, 2.2712,
      0.4177, 1.2132, 0.3052,  0.4835, -0.2171, 0.5057, -0.2059, 1.1583,
      0.0946, 0.7013, -0.6236, 3.0253, 0.5,     0.9,    0.9,     0.2};

    const size_t kNumPoints = 12;
    EigenVec<Eigen::Vector2d> points1(kNumPoints);
    EigenVec<Eigen::Vector2d> points2(kNumPoints);
    for (size_t i = 0; i < kNumPoints; ++i) {
        points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
        points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
    }

    estimator::RANSACConfig config; 
    config.max_inlier_error = 0.02;
    config.confidence = 0.9999;  
    estimator::RANSAC<estimator::EssentialMatrixFivePointEstimator> estimator(config); 
    const auto report = estimator.estimate(points1, points2); 

    for (size_t i = 0; i < 10; ++i) {
        BOOST_CHECK(report.inlier_mask[i]); 
    }
    BOOST_CHECK(!report.inlier_mask[10]); 
    BOOST_CHECK(!report.inlier_mask[11]); 
}

BOOST_AUTO_TEST_CASE(TestDecompose) {
    const Eigen::Matrix3d R = base::rpy2RotationMatrix(Eigen::Vector3d(1, 1.2, 1.3));
    const Eigen::Vector3d t = Eigen::Vector3d(1, 3, 5).normalized();
    const Eigen::Matrix3d E = estimator::essentialMatrixFromPose(R, t);

    EigenVec<Eigen::Matrix3d> Rs;
    EigenVec<Eigen::Vector3d> ts;
    estimator::decomposeEssentialMatrix(E, Rs, ts);
    BOOST_CHECK_EQUAL(Rs.size(), 4);
    BOOST_CHECK_EQUAL(ts.size(), 4);

    bool solution_exists = false;
    for (size_t i = 0; i < Rs.size(); ++i) {
        if ((Rs[i] - R).norm() < 1e-5 && (ts[i] - t).norm() < 1e-5) {
            solution_exists = true;
        }
    }
    BOOST_CHECK(solution_exists);
}

BOOST_AUTO_TEST_CASE(TestPoseFromEssentialMatrix) {
    const Eigen::Matrix3d R = base::rpy2RotationMatrix(util::deg2Rad(20), util::deg2Rad(10), 0);
    const Eigen::Vector3d t = Eigen::Vector3d(1, 0, 0).normalized();
    const Eigen::Matrix3d E = estimator::essentialMatrixFromPose(R, t);

    const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
    const Eigen::Matrix3x4d proj_matrix2 = base::composePose(R, t);

    EigenVec<Eigen::Vector3d> points3D(4);
    points3D[0] = Eigen::Vector3d(0, 0, 1);
    points3D[1] = Eigen::Vector3d(0, 0.1, 1);
    points3D[2] = Eigen::Vector3d(0.1, 0, 1);
    points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

    EigenVec<Eigen::Vector2d> points1(4);
    EigenVec<Eigen::Vector2d> points2(4);
    for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d point1 = proj_matrix1 * points3D[i].homogeneous();
        points1[i] = point1.hnormalized();
        const Eigen::Vector3d point2 = proj_matrix2 * points3D[i].homogeneous();
        points2[i] = point2.hnormalized();
    }

    points3D.clear();

    Eigen::Matrix3d RR;
    Eigen::Vector3d tt;
    estimator::poseFromEssentialMatrix(E, points1, points2, RR, tt, points3D);

    BOOST_CHECK_EQUAL(points3D.size(), 4);

    std::cout << "RR: \n" << RR << std::endl; 
    std::cout << "R: \n" << R << std::endl; 
    std::cout << "tt: " << tt.transpose() << std::endl; 
    std::cout << "t: " << t.transpose() << std::endl; 
    BOOST_CHECK(RR.isApprox(R));
    BOOST_CHECK(tt.isApprox(t));
}
