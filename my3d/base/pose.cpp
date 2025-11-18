/*
 * filename: pose.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about pose.
 */

#include "base/pose.h"

namespace my3d {
namespace base {

Eigen::Matrix3d axisAngle2RotationMatrix(const Eigen::Vector3d &axis, 
                                         double angle) {
    const Eigen::Vector3d axis_n = axis.normalized();

    return std::cos(angle) * Eigen::Matrix3d::Identity() + 
           std::sin(angle) * util::getSkewSymmetric(axis_n) + 
           (1.0 - std::cos(angle)) * axis_n * axis_n.transpose();
}

Eigen::Matrix3d rpy2RotationMatrix(const Eigen::Vector3d &rpy) {
    const double roll = rpy.x();
    const double pitch = rpy.y();
    const double yaw = rpy.z();
    Eigen::Quaterniond qx(std::cos(roll / 2.0), std::sin(roll / 2.0), 0, 0);
    Eigen::Quaterniond qy(std::cos(pitch / 2.0), 0, std::sin(pitch / 2.0), 0);
    Eigen::Quaterniond qz(std::cos(yaw / 2.0), 0, 0, std::sin(yaw / 2.0));

    return (qx * qy * qz).toRotationMatrix();
}

Eigen::Matrix3d rpy2RotationMatrix(double roll, double pitch, double yaw) {
    return rpy2RotationMatrix(Eigen::Vector3d(roll, pitch, yaw));
}

Eigen::Matrix3x4d composePose(const Eigen::Matrix3d &rotation, 
                             const Eigen::Vector3d &translation) {
    Eigen::Matrix3x4d composed_pose;
    composed_pose.leftCols<3>() = rotation;
    composed_pose.rightCols<1>() = translation;

    return composed_pose;
}

Eigen::Matrix3d quaternion2RotationMatrix(const Eigen::Quaterniond &quaternion) {
    const Eigen::Quaterniond normalized_quaternion = quaternion.normalized(); 
    return normalized_quaternion.toRotationMatrix(); 
}

Eigen::Quaterniond rotationMatrix2Quaternion(const Eigen::Matrix3d &rotation) {
    return Eigen::Quaterniond(rotation);
}

Eigen::Matrix3x4d computeProjectionMatrix(const Eigen::Matrix3d &K, 
                                          const Eigen::Matrix3d &R, 
                                          const Eigen::Vector3d &t) {
    return K * composePose(R, t);
}

bool checkCheirality(const Eigen::Matrix3d &rotation, 
                     const Eigen::Vector3d &translation, 
                     const Eigen::Matrix3d &K1, 
                     const Eigen::Matrix3d &K2, 
                     const EigenVec<Eigen::Vector2d> &points1, 
                     const EigenVec<Eigen::Vector2d> &points2, 
                     EigenVec<Eigen::Vector3d> *points_3D) {
    assert(points1.size() == points2.size());
    
    Eigen::Matrix3x4d proj1 = K1 * Eigen::Matrix3x4d::Identity();
    Eigen::Matrix3x4d proj2 = K2 * composePose(rotation, translation);
    
    const size_t num_pnts = points1.size();
    const double kMinDepth = std::numeric_limits<double>::epsilon();
    const double max_depth = 1000.0 * (rotation.transpose() * translation).norm(); 
    if (points_3D) {
        points_3D->clear();
    }
    bool ret = false;
    for (size_t i = 0; i < num_pnts; ++i) {
        const Eigen::Vector3d point_3D = 
            triangulatePoint(proj1, proj2, points1[i], points2[i]);
        // std::cout << point_3D.z() << " " << (rotation * point_3D + translation).z() << std::endl;
        const double depth1 = point_3D.z(); 
        const double depth2 = (rotation * point_3D + translation).z(); 
        if (depth1 > kMinDepth && depth1 < max_depth && 
            depth2 > kMinDepth && depth2 < max_depth) {
            ret = true; 
            if (points_3D) {
                points_3D->push_back(point_3D); 
            }
        }
    }

    return ret;
}

}
}
