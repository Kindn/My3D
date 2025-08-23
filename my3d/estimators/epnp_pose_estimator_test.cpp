#define TEST_NAME "estimators/epnp_pose_estimator_test"
#include "utils/unit_test.h"

#include "estimators/EPnPPoseEstimator.h"
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

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity(); 
    K << 3205.9, 7.79215, 1971.26, 
         0, 3185.74, 1544.62, 
         0, 0, 1; 
    base::RadialPinHoleCamera camera(K);
    EigenVec<Eigen::Vector3d> pgs;
    pgs.emplace_back(1, 0, 0);
    pgs.emplace_back(0, 2, 0);
    pgs.emplace_back(0, 0, 3); 
    pgs.emplace_back(1, 1, 1);
    pgs.emplace_back(0, 1, 1);
    pgs.emplace_back(3, 1.0, 2);
    pgs.emplace_back(3, 1.1, 2);
    pgs.emplace_back(3, 1.2, 2);
    pgs.emplace_back(3, 1.3, 2);
    pgs.emplace_back(3, 1.4, 2);
    pgs.emplace_back(2, 1, 1);
    EigenVec<Eigen::Vector2d> pcs;
    for (size_t i = 0; i < pgs.size(); ++i) {
        pcs.push_back(camera.project(R.transpose() * (pgs[i] - t)));
    }
    
    estimator::EPnPPoseEstimator estimator(K);
    EigenVec<Eigen::Matrix3x4d> poses;
    estimator.estimate(pgs, pcs, poses);

    std::cout << "R_est: \n[" << poses[0].leftCols<3>() << "]" << std::endl;
    // std::cout << "R * R^T: \n[" << R * R.transpose() << "]" << std::endl;
    std::cout << "t_est: [" << poses[0].rightCols<1>().transpose() << "]" << std::endl;

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

// int main() {
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
//     pgs.emplace_back(1, 0, 0);
//     pgs.emplace_back(0, 2, 0);
//     pgs.emplace_back(0, 0, 3); 
//     pgs.emplace_back(1, 1, 1);
//     pgs.emplace_back(0, 1, 1);
//     pgs.emplace_back(3, 1.0, 4);
//     pgs.emplace_back(3, 1.1, 4);
//     pgs.emplace_back(3, 1.2, 4);
//     pgs.emplace_back(3, 1.3, 4);
//     pgs.emplace_back(3, 1.4, 4);
//     pgs.emplace_back(2, 1, 7);
//     EigenVec<Eigen::Vector2d> pcs;
//     for (size_t i = 0; i < pgs.size(); ++i) {
//         pcs.push_back(camera.project(R.transpose() * (pgs[i] - t)));
//     }
    
//     estimator::EPnPPoseEstimator estimator(K);
//     EigenVec<Eigen::Matrix3x4d> poses;
//     estimator.estimate(pgs, pcs, poses);

//     bool exists = false;
//     for (size_t i = 0; i < poses.size(); ++i) {
//         if ((poses[i].block<3, 3>(0, 0) * R.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-6 && 
//             (poses[i].col(3) - t).norm() < 1e-6) {
//             exists = true;
//             break;
//         }
//     }
//     assert(exists);

//     return 0; 
// }
