/*
 * filename: projection.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about camera projection.
 */

#ifndef _MY3D_BASE_PROJECTION_H_
#define _MY3D_BASE_PROJECTION_H_

#include "base/camera/CameraBase.h"
#include "base/pose.h"

namespace my3d {
namespace base {

/**
 * @brief Compute the projection matrix corresponding to the given camera
 * intrinsics and relative pose [R t]
 */
Eigen::Matrix3x4d computeProjectionMatrix(double fx, double fy, double cx,
                                          double cy, double alpha,
                                          const Eigen::Matrix3d &R,
                                          const Eigen::Vector3d &t);

/**
 * @brief Compute reprojection residual of a point
 *
 * @param point_3D     point's coordinate in world frame
 * @param point_2D     pixel coordinate of point_3D
 * @param proj_matrix  camera extrinsics [R t]
 * @param camera       the camera model
 */
Eigen::Vector2d computeReprojectionResidual(
    const Eigen::Vector3d &point_3D, const Eigen::Vector2d &point_2D,
    const Eigen::Matrix3x4d &proj_matrix, const base::CameraBase *camera);

/**
 * @brief Compute reprojection residual of a point. This function only suits for
 * cameras whose projection could be represented as a 3x4 matrix
 *
 * @param point_3D           point's coordinate in camera frame
 * @param point_2D           pixel coordinate of point_3D
 * @param projection_matrix  the camera model
 */
Eigen::Vector2d
computeReprojectionResidual(const Eigen::Vector3d &point_3D,
                            const Eigen::Vector2d &point_2D,
                            const Eigen::Matrix3x4d &projection_matrix);

/**
 * @brief Compute reprojection error of a point
 *
 * @param point_3D     point's coordinate in world frame s.t. pixel =
 * camera->projection(point_3D)
 * @param point_2D     pixel coordinate of point_3D
 * @param proj_matrix  camera extrinsics [R t]
 * @param camera       the camera model
 */
double computeReprojectionError(const Eigen::Vector3d &point_3D,
                                const Eigen::Vector2d &point_2D,
                                const Eigen::Matrix3x4d &proj_matrix,
                                const base::CameraBase *camera);

/**
 * @brief Compute reprojection error of a point. This function only suits for
 * cameras whose projection could be represented as a 3x4 matrix
 *
 * @param point_3D     point's coordinate in camera frame s.t. pixel =
 * camera->projection(point_3D)
 * @param point_2D     pixel coordinate of point_3D
 * @param projection_matrix  the camera model
 */
double computeReprojectionError(const Eigen::Vector3d &point_3D,
                                const Eigen::Vector2d &point_2D,
                                const Eigen::Matrix3x4d &projection_matrix);

/**
 * @brief Compute normalized angular error
 *
 * @param point_2D     point's coordinate in normalized camera frame s.t. pixel
 * = camera->projection(point_3D)
 * @param point_3D     object point in world frame
 * @param proj_matrix  projection matrix
 */
double computeNormalizedAngularError(const Eigen::Vector2d &point_2D,
                                     const Eigen::Vector3d &point_3D,
                                     const Eigen::Matrix3x4d &proj_matrix);

/**
 * @brief Check if the 3D point passes cheirality constraint
 */
bool hasPointPositiveDepth(const Eigen::Matrix3x4d &proj_matrix,
                           const Eigen::Vector3d &point_3D);
} // namespace base
} // namespace my3d

#endif // _MY3D_BASE_PROJECTION_H_
