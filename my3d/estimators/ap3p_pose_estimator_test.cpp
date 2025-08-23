#define TEST_NAME "estimators/ap3p_pose_estimator_test"
#include "utils/unit_test.h"

#include "estimators/AP3PPoseEstimator.h"
#include "base/pose.h"
#include "base/camera/RadialPinHoleCamera.h"

#include <opencv2/calib3d/calib3d.hpp>

using namespace my3d;

BOOST_AUTO_TEST_CASE(TestEstimate) {
    /* Camera pose ground truth */
    const Eigen::Matrix3d R = base::axisAngle2RotationMatrix(Eigen::Vector3d(1, -1, 0), 2.0 * M_PI / 3.0);
    const Eigen::Vector3d t(3, 3, 1.2);

    std::cout << "R: \n[" << R << "]" << std::endl;
    // std::cout << "R * R^T: \n[" << R * R.transpose() << "]" << std::endl;
    std::cout << "t: [" << t.transpose() << "]" << std::endl;

    Eigen::Matrix3d K; 
    K << 3205.9, 7.79215, 1971.26, 
         0, 3185.74, 1544.62, 
         0, 0, 1; 
    base::RadialPinHoleCamera camera(K);
    EigenVec<Eigen::Vector3d> pgs;
    pgs.push_back(Eigen::Vector3d(1, 0, 0));
    pgs.push_back(Eigen::Vector3d(0, 2, 0));
    pgs.push_back(Eigen::Vector3d(0, 0, 3));
    EigenVec<Eigen::Vector2d> pcs;
    for (int i = 0; i < 3; ++i) {
        pcs.push_back(camera.project(R.transpose() * (pgs[i] - t)));
    }

    estimator::AP3PPoseEstimator estimator(K);
    EigenVec<Eigen::Matrix3x4d> poses;
    estimator.estimate(pgs, pcs, poses);

    bool exists = false;
    for (size_t i = 0; i < poses.size(); ++i) {
        if ((poses[i].block<3, 3>(0, 0) * R.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-6 && 
            (poses[i].col(3) - t).norm() < 1e-6) {
            exists = true;
            break;
        }
    }
    BOOST_CHECK(exists);
}

// BOOST_AUTO_TEST_CASE(TestOpenCVAP3P) {
//     /* Camera pose ground truth */
//     const Eigen::Matrix3d R = base::axisAngle2RotationMatrix(Eigen::Vector3d(1, -1, 0), 2.0 * M_PI / 3.0);
//     const Eigen::Vector3d t(3, 3, 1.2);

//     std::cout << "R: \n[" << R << "]" << std::endl;
//     // std::cout << "R * R^T: \n[" << R * R.transpose() << "]" << std::endl;
//     std::cout << "t: [" << t.transpose() << "]" << std::endl;

//     Eigen::Matrix3d K; 
//     K << 3205.9, 7.79215, 1971.26, 
//          0, 3185.74, 1544.62, 
//          0, 0, 1; 
//     base::RadialPinHoleCamera camera(K);
//     EigenVec<Eigen::Vector3d> pgs;
//     pgs.push_back(Eigen::Vector3d(1, 0, 0));
//     pgs.push_back(Eigen::Vector3d(0, 2, 0));
//     pgs.push_back(Eigen::Vector3d(0, 0, 3));
//     EigenVec<Eigen::Vector2d> pcs;
//     std::vector<cv::Point3d> pgs_cv(3);
//     std::vector<cv::Point2d> pcs_cv(3);
//     for (int i = 0; i < 3; ++i) {
//         Eigen::Vector2d pix = camera.project(R.transpose() * (pgs[i] - t));
//         pgs_cv[i].x = pgs[i].x();
//         pgs_cv[i].y = pgs[i].y();
//         pgs_cv[i].z = pgs[i].z();
//         pcs_cv[i].x = pcs[i].x();
//         pcs_cv[i].y = pcs[i].y();
//     }

//     cv::Mat K_cv = (cv::Mat_<double>(3, 3)  << 3205.9, 7.79215, 1971.26, 
//          0, 3185.74, 1544.62, 
//          0, 0, 1);
//     cv::Mat dist_cv = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    
// }
