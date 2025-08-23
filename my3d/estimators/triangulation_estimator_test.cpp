#define TEST_NAME "estimators/triangulation_estimator_test"
#include "utils/unit_test.h"

#include "estimators/TriangulationEstimator.h"
#include "estimators/RANSAC.h"
#include "utils/RandomNumberGenerator.h"
#include "base/camera/RadialPinHoleCamera.h"

using namespace my3d; 

BOOST_AUTO_TEST_CASE(TestEstimate) {
    const Eigen::Vector3d point_3D(1, 2, 5); 
    Eigen::Matrix3d K; 
    K << 3205.9, 7.8, 1971.3, 0, 3185.7, 1544.6, 0, 0, 1; 
    const base::CameraBase *camera = new base::RadialPinHoleCamera(K); 
    const size_t kNumViews = 10; 
    util::RandomNumberGenerator rng; 
    estimator::TriangulationEstimator::VecXData x_data; 
    estimator::TriangulationEstimator::VecYData y_data; 
    for (size_t i = 0; i < kNumViews; ++i) {
        const double x = 3.0 * std::cos(2.0 * M_PI / kNumViews * i); 
        const double y = 3.0 * std::sin(2.0 * M_PI / kNumViews * i); 
        const double z = -1.0 + 2.0 / kNumViews * i; 
        const Eigen::Vector3d t = Eigen::Vector3d(x, y, z); 
        const double roll = rng.uniformReal(-M_PI / 6.0, M_PI / 6.0); 
        const double pitch = rng.uniformReal(-M_PI / 6.0, M_PI / 6.0); 
        const double yaw = rng.uniformReal(-M_PI, M_PI); 
        const Eigen::Matrix3d R = base::rpy2RotationMatrix(Eigen::Vector3d(roll, pitch, yaw)); 
        const Eigen::Matrix3x4d proj_n = base::composePose(R.inverse(), -R.inverse() * t);
        const Eigen::Matrix3x4d proj = K * proj_n;
        const Eigen::Vector2d point_2D = (proj * point_3D.homogeneous()).hnormalized(); 
        x_data.emplace_back(point_2D, camera->pix2Norm(point_2D));
        y_data.emplace_back(proj_n, t, camera); 
    }
    estimator::TriangulationEstimator estimator; 
    estimator.setMinTriAngleDeg(2.0);
    EigenVec<typename estimator::TriangulationEstimator::ModelType> models; 
    estimator.estimate(x_data, y_data, models); 

    delete camera; 
    camera = nullptr; 

    BOOST_CHECK(!models.empty()); 
    BOOST_CHECK_LE((models[0] - point_3D).norm(), 1e-10); 
}

BOOST_AUTO_TEST_CASE(TestRANSAC) {
    const Eigen::Vector3d point_3D(1, 2, 5); 
    Eigen::Matrix3d K; 
    K << 3205.9, 0.0, 1971.3, 0, 3185.7, 1544.6, 0, 0, 1; 
    const base::CameraBase *camera = new base::RadialPinHoleCamera(K); 
    const size_t kNumViews = 10; 
    util::RandomNumberGenerator rng; 
    estimator::TriangulationEstimator::VecXData x_data; 
    estimator::TriangulationEstimator::VecYData y_data; 
    for (size_t i = 0; i < kNumViews; ++i) {
        const double x = 3.0 * std::cos(2.0 * M_PI / kNumViews * i); 
        const double y = 3.0 * std::sin(2.0 * M_PI / kNumViews * i); 
        const double z = -1.0 + 2.0 / kNumViews * i; 
        const Eigen::Vector3d t = Eigen::Vector3d(x, y, z); 
        const double roll = rng.uniformReal(-M_PI / 6.0, M_PI / 6.0); 
        const double pitch = rng.uniformReal(-M_PI / 6.0, M_PI / 6.0); 
        const double yaw = rng.uniformReal(-M_PI, M_PI); 
        const Eigen::Matrix3d R = base::rpy2RotationMatrix(Eigen::Vector3d(roll, pitch, yaw)); 
        const Eigen::Matrix3x4d proj_n = base::composePose(R.inverse(), -R.inverse() * t);
        const Eigen::Matrix3x4d proj = K * proj_n; 
        const Eigen::Vector2d point_2D = (proj * point_3D.homogeneous()).hnormalized(); 
        x_data.emplace_back(point_2D, camera->pix2Norm(point_2D));
        y_data.emplace_back(proj_n, t, camera); 
    }
    // Make some data faulty
    for (size_t i = 0; i < 3; ++i) {
        x_data[i].point += Eigen::Vector2d(rng.uniformReal(10, 15), rng.uniformReal(10, 15)); 
        x_data[i].point_normalized = camera->pix2Norm(x_data[i].point); 
    }

    estimator::RANSACConfig ransac_config; 
    ransac_config.max_inlier_error = camera->thresholdPix2Norm(1.0);
    std::cout << "max_inlier_error: " << ransac_config.max_inlier_error << std::endl; 
    // ransac_config.max_iter_num = 10000;
    estimator::RANSAC<estimator::TriangulationEstimator> estimator(ransac_config); 
    estimator.getEstimator().setMinTriAngleDeg(2.0); 
    estimator.getEstimator().setEstimationErrorType(estimator::TriangulationEstimator::EstimationErrorType::REPROJECTION_ERROR);
    estimator::RANSAC<estimator::TriangulationEstimator>::Report report = 
        estimator.estimate(x_data, y_data); 

    delete camera; 
    camera = nullptr; 
    
    BOOST_CHECK(report.success); 
    BOOST_CHECK_LE((report.model - point_3D).norm(), 1e-10); 
    for (size_t i = 0; i < kNumViews; ++i) {
        std::cout << report.errors[i] << std::endl; 
        if (i < 3) {
            BOOST_CHECK(!report.inlier_mask[i]); 
        } else {
            BOOST_CHECK(report.inlier_mask[i]);
        }
    }
    std::cout << "model: " << report.model.transpose() << std::endl; 
}
