#define TEST_NAME "estimators/homography_estimator_test"
#include "utils/unit_test.h"

#include "estimators/HomographyEstimator.h"
#include "utils/RandomNumberGenerator.h"

BOOST_AUTO_TEST_CASE(TestEstimate1) {
    typedef my3d::estimator::HomographyEstimator::ModelType ModelType;
    typedef my3d::estimator::HomographyEstimator::VecXData VecXData;
    typedef my3d::estimator::HomographyEstimator::VecYData VecYData;

    // Generate random homographies
    ModelType model;
    for (double x = 1; x <= 10; ++x) {
        model << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;
        VecXData src;
        src.emplace_back(x, 0);
        src.emplace_back(1, 0);
        src.emplace_back(2, 1);
        src.emplace_back(10, 30);

        VecYData dst;

        for (size_t i = 0; i < 4; ++i) {
            const Eigen::Vector3d dsth = model * src[i].homogeneous();
            dst.push_back(dsth.hnormalized());
        }

        my3d::estimator::HomographyEstimator estimator;
        my3d::EigenVec<Eigen::Matrix3d> models;
        estimator.estimate(src, dst, models);
        auto result_model = models[0];
        // std::cout << "[" << result_model << "]" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            Eigen::Vector2d residual = estimator.computeResidual(src[i], dst[i], result_model);
            BOOST_CHECK_LT(residual.norm(), 1e-6);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestEstimate2) {
    typedef my3d::estimator::HomographyEstimator::ModelType ModelType;
    typedef my3d::estimator::HomographyEstimator::VecXData VecXData;
    typedef my3d::estimator::HomographyEstimator::VecYData VecYData;

    ModelType model;
    model << 1, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    const size_t num_points = 20;
    my3d::util::RandomNumberGenerator rng;
    VecXData x_data, sampled_x_data(4);
    VecYData y_data, sampled_y_data(4);
    for (size_t i = 0; i < num_points; ++i) {
        x_data.push_back(Eigen::Vector2d(rng.uniformReal(-10, 10), 
                                         rng.uniformReal(-10, 10)));
        y_data.push_back((model * x_data.back().homogeneous()).hnormalized());
    }

    my3d::estimator::HomographyEstimator estimator;
    for (size_t i1 = 0; i1 < num_points - 3UL; ++i1) {
        for (size_t i2 = i1 + 1; i2 < num_points - 2UL; ++i2) {
            for (size_t i3 = i2 + 1; i3 < num_points - 1UL; ++i3) {
                for (size_t i4 = i3 + 1; i4 < num_points; ++i4) {
                    sampled_x_data[0] = x_data[i1]; 
                    sampled_x_data[1] = x_data[i2]; 
                    sampled_x_data[2] = x_data[i3]; 
                    sampled_x_data[3] = x_data[i4]; 
                    sampled_y_data[0] = y_data[i1]; 
                    sampled_y_data[1] = y_data[i2]; 
                    sampled_y_data[2] = y_data[i3]; 
                    sampled_y_data[3] = y_data[i4]; 

                    my3d::EigenVec<ModelType> models;
                    estimator.estimate(sampled_x_data, sampled_y_data, models);
                    BOOST_CHECK_LT((models[0].normalized() - model.normalized()).norm(), 0.1);
                    // std::cout << "[" << result_model << "]" << std::endl;
                    // for (size_t i = 0; i < num_points; ++i) {
                    //     Eigen::Vector2d residual = estimator.computeResidual(x_data[i], y_data[i], result_model);
                    //     BOOST_CHECK_LT(residual.norm(), 1e-3);
                    // }
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TestEstimateRandomData) {
    typedef my3d::estimator::HomographyEstimator::ModelType ModelType;
    typedef my3d::estimator::HomographyEstimator::VecXData VecXData;
    typedef my3d::estimator::HomographyEstimator::VecYData VecYData;

    // Generate random homographies
    ModelType model;
    my3d::util::RandomNumberGenerator rng;
    for (double x = 1; x <= 10; ++x) {
        model << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;
        VecXData src;
        for (size_t i = 0; i < 4; ++i) {
            src.emplace_back(rng.uniformReal(-1000, 1000), rng.uniformReal(-1000, 1000));
        }
        VecYData dst;

        for (size_t i = 0; i < 4; ++i) {
            const Eigen::Vector3d dsth = model * src[i].homogeneous();
            dst.push_back(dsth.hnormalized());
        }

        my3d::estimator::HomographyEstimator estimator;
        my3d::EigenVec<ModelType> models;
        estimator.estimate(src, dst, models);
        ModelType result_model = models[0];
        // std::cout << "[" << result_model << "]" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            Eigen::Vector2d residual = estimator.computeResidual(src[i], dst[i], result_model);
            BOOST_CHECK_LT(residual.norm(), 1e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestDecompose) {
    Eigen::Matrix3d H;
  H << 2.649157564634028, 4.583875997496426, 70.694447785121326,
      -1.072756858861583, 3.533262150437228, 1513.656999614321649,
      0.001303887589576, 0.003042206876298, 1;
  H *= 3;

  Eigen::Matrix3d K;
  K << 640, 0, 320, 0, 640, 240, 0, 0, 1;

  my3d::EigenVec<Eigen::Matrix3d> R;
  my3d::EigenVec<Eigen::Vector3d> t;
  my3d::EigenVec<Eigen::Vector3d> n;
  my3d::estimator::decomposeHomography(H, K, K, R, t, n);

  BOOST_CHECK_EQUAL(R.size(), 4);
  BOOST_CHECK_EQUAL(t.size(), 4);
  BOOST_CHECK_EQUAL(n.size(), 4);

  Eigen::Matrix3d R_ref;
  R_ref << 0.43307983549125, 0.545749113549648, -0.717356090899523,
      -0.85630229674426, 0.497582023798831, -0.138414255706431,
      0.281404038139784, 0.67421809131173, 0.682818960388909;
  const Eigen::Vector3d t_ref(1.826751712278038, 1.264718492450820,
                              0.195080809998819);
  const Eigen::Vector3d n_ref(-0.244875830334816, -0.480857890778889,
                              -0.841909446789566);

  bool ref_solution_exists = false;
  for (size_t i = 0; i < 4; ++i) {
    const double kEps = 1e-6;
    std::cout << "[" << R[i] << "], [" << t[i].transpose() << "], [" << n[i].transpose() << "]" << std::endl;
    if ((R[i] - R_ref).norm() < kEps && (t[i] - t_ref).norm() < kEps &&
        (n[i] - n_ref).norm() < kEps) {
      ref_solution_exists = true;
    }
  }
  BOOST_CHECK(ref_solution_exists);
}

BOOST_AUTO_TEST_CASE(TestPoseFromHomography) {
    const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R_ref = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t_ref(1, 0, 0);
    const Eigen::Vector3d n_ref(-1, 0, 0);
    const double d_ref = 1;
    const Eigen::Matrix3d H =
        my3d::estimator::homographyFromPose(K1, K2, R_ref, t_ref, n_ref, d_ref);

    my3d::EigenVec<Eigen::Vector2d> points1;
    points1.emplace_back(0.1, 0.4);
    points1.emplace_back(0.2, 0.3);
    points1.emplace_back(0.3, 0.2);
    points1.emplace_back(0.4, 0.1);

    my3d::EigenVec<Eigen::Vector2d> points2;
    for (const auto& point1 : points1) {
        const Eigen::Vector3d point2 = H * point1.homogeneous();
        points2.push_back(point2.hnormalized());
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Vector3d n;
    my3d::EigenVec<Eigen::Vector3d> points3D;
    my3d::estimator::poseFromHomography(H, K1, K2, points1, points2, R, t, n, points3D);

    BOOST_CHECK_EQUAL(R, R_ref);
    BOOST_CHECK_EQUAL(t, t_ref);
    BOOST_CHECK_EQUAL(n, n_ref);
    BOOST_CHECK_EQUAL(points3D.size(), points1.size());

    std::cout << "result R: \n[" << R << "]" << std::endl;
    std::cout << "result t: [" << t.transpose() << "]" << std::endl;
    std::cout << "result n: [" << n.transpose() << "]" << std::endl;
}
