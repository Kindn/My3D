/*
 * filename: pose.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about pose.
 */

#ifndef _MY3D_BASE_POSE_H_
#define _MY3D_BASE_POSE_H_

#include "base/triangulation.h"
#include "utils/math.h"

namespace my3d {
namespace base {

/**
 * @brief Convert an axis-angle to rotation matrix using Rodrigues formula.
 *
 * @param axis   the axis which only represents the direction
 * @param angle  the angle of rotation around axis (in rad, sign is determined
 * by the right hand rule)
 * @retval       the rotation matrix
 */
Eigen::Matrix3d axisAngle2RotationMatrix(const Eigen::Vector3d &axis,
                                         double angle);

/**
 * @brief Get the corresponding rotation matrix of an intrinsic x-y-z
 * (roll-pitch-yaw) Euler angle. Unit: rad
 */
Eigen::Matrix3d rpy2RotationMatrix(const Eigen::Vector3d &rpy);

/**
 * @brief Get the corresponding rotation matrix of an intrinsic x-y-z
 * (roll-pitch-yaw) Euler angle. Unit: rad
 */
Eigen::Matrix3d rpy2RotationMatrix(double roll, double pitch, double yaw);

/**
 * @brief Compose rotation and translation to form a 3x4 matrix.
 */
Eigen::Matrix3x4d composePose(const Eigen::Matrix3d &rotation,
                              const Eigen::Vector3d &translation);

/**
 * @brief Convert quaternion representation to rotation matrix
 */
Eigen::Matrix3d quaternion2RotationMatrix(const Eigen::Quaterniond &quaternion);

/**
 * @brief Convert a rotation matrix to Hamilton quaternion
 */
Eigen::Quaterniond rotationMatrix2Quaternion(const Eigen::Matrix3d &rotation);

/**
 * @brief Compute the projection matrix corresponding to the given camera matrix
 * K and relative pose [R t]
 */
Eigen::Matrix3x4d computeProjectionMatrix(const Eigen::Matrix3d &K,
                                          const Eigen::Matrix3d &R,
                                          const Eigen::Vector3d &t);

/**
 * @brief Perform cheirality constraint test. The extrinsics of camera 1 is [I |
 * 0] and the extrinsics of camera 2 is [R | t]
 */
bool checkCheirality(const Eigen::Matrix3d &rotation,
                     const Eigen::Vector3d &translation,
                     const Eigen::Matrix3d &K1, const Eigen::Matrix3d &K2,
                     const EigenVec<Eigen::Vector2d> &points1,
                     const EigenVec<Eigen::Vector2d> &points2,
                     EigenVec<Eigen::Vector3d> *points_3D);
} // namespace base
} // namespace my3d

#endif // _MY3D_BASE_POSE_H_
