#define TEST_NAME "estimators/ransac_test"
#include "utils/unit_test.h"

#include "estimators/HomographyEstimator.h"
#include "estimators/RANSAC.h"

BOOST_AUTO_TEST_CASE(TestCountInliers) {
    typedef my3d::estimator::HomographyEstimator::ModelType ModelType;
    typedef my3d::estimator::HomographyEstimator::VecXData VecXData;
    typedef my3d::estimator::HomographyEstimator::VecYData VecYData;

    const size_t num_points = 500;
    const size_t num_outliers = 100;

    // Generate a random homography
    ModelType model;
    model << 1, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    // Generate exact data
    my3d::util::RandomNumberGenerator rng;
    VecXData x_data;
    VecYData y_data;
    for (size_t i = 0; i < num_points; ++i) {
        x_data.push_back(Eigen::Vector2d(rng.uniformReal(-1000, 1000), 
                                         rng.uniformReal(-1000, 1000)));
        y_data.push_back((model * x_data.back().homogeneous()).hnormalized());
    }

    // Generate some faulty data
    for (size_t i = 0; i < num_outliers; ++i) {
        y_data[i] += Eigen::Vector2d(rng.uniformReal(500, 1000), 
                                     rng.uniformReal(-1000, -500));
    }

    my3d::estimator::RANSACConfig config;
    config.max_inlier_error = 10;
    my3d::estimator::RANSAC<typename my3d::estimator::HomographyEstimator> ransac(config);

    std::vector<bool> inlier_mask;
    std::vector<double> errors; 
    size_t num_inliers = ransac.countInliers(x_data, y_data, model, inlier_mask, errors);

    BOOST_CHECK_EQUAL(num_inliers, num_points - num_outliers);
    for (size_t i = 0; i < x_data.size(); ++i) {
        if (i < num_outliers) {
            BOOST_CHECK(!inlier_mask[i]);
        } else {
            BOOST_CHECK(inlier_mask[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestHomography) {
    typedef my3d::estimator::HomographyEstimator::ModelType ModelType;
    typedef my3d::estimator::HomographyEstimator::VecXData VecXData;
    typedef my3d::estimator::HomographyEstimator::VecYData VecYData;

    const size_t num_points = 500;
    const size_t num_outliers = 100;

    // Generate a random homography
    ModelType model;
    model << 1, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    // Generate exact data
    my3d::util::RandomNumberGenerator rng;
    VecXData x_data;
    VecYData y_data;
    for (size_t i = 0; i < num_points; ++i) {
        x_data.push_back(Eigen::Vector2d(rng.uniformReal(-10, 10), 
                                         rng.uniformReal(-10, 10)));
        y_data.push_back((model * x_data.back().homogeneous()).hnormalized());
    }

    // Generate some faulty data
    for (size_t i = 0; i < num_outliers; ++i) {
        y_data[i] += Eigen::Vector2d(rng.uniformReal(500, 1000), 
                                     rng.uniformReal(-1000, -500));
    }

    // Estimate using RANSAC
    my3d::estimator::HomographyEstimator estimator;
    my3d::estimator::RANSACConfig config;
    config.max_inlier_error = 5;
    my3d::estimator::RANSAC<typename my3d::estimator::HomographyEstimator> ransac(config);

    my3d::estimator::RANSAC<typename my3d::estimator::HomographyEstimator>::Report report = ransac.estimate(x_data, y_data);

    BOOST_CHECK(report.success);
    BOOST_CHECK_GT(report.num_iter, 0);
    BOOST_CHECK_GT(report.inlier_error, 0);

    for (size_t i = 0; i < x_data.size(); ++i) {
        if (i < num_outliers) {
            BOOST_CHECK(!report.inlier_mask[i]);
        } else {
            BOOST_CHECK(report.inlier_mask[i]);
        }
    }

    std::cout << "result error: " << report.inlier_error << std::endl;
    std::cout << "result model: " << std::endl << report.model << std::endl;
    double mat_diff = (model - report.model).norm();
    BOOST_CHECK_LT(mat_diff, 1e-6);
}
