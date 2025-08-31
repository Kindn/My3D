/*
 * filename: projection.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about camera projection.
 */

#include "base/camera/projection.h"

namespace my3d {
namespace base {

Eigen::Matrix3x4d computeProjectionMatrix(double fx, double fy, double cx,
                                          double cy, double alpha,
                                          const Eigen::Matrix3d &R,
                                          const Eigen::Vector3d &t) {
  Eigen::Matrix3d K;
  K << fx, alpha, cx, 0, fy, cy, 0, 0, 1;

  return computeProjectionMatrix(K, R, t);
}

Eigen::Vector2d computeReprojectionResidual(
    const Eigen::Vector3d &point_3D, const Eigen::Vector2d &point_2D,
    const Eigen::Matrix3x4d &proj_matrix, const base::CameraBase *camera) {
  const Eigen::Vector3d pc = proj_matrix * point_3D.homogeneous();
  const Eigen::Vector2d proj = camera->project(pc);
  return proj - point_2D;
}

Eigen::Vector2d
computeReprojectionResidual(const Eigen::Vector3d &point_3D,
                            const Eigen::Vector2d &point_2D,
                            const Eigen::Matrix3x4d &projection_matrix) {
  const Eigen::Vector2d proj =
      (projection_matrix * point_3D.homogeneous()).hnormalized();
  return proj - point_2D;
}

double computeReprojectionError(const Eigen::Vector3d &point_3D,
                                const Eigen::Vector2d &point_2D,
                                const Eigen::Matrix3x4d &proj_matrix,
                                const base::CameraBase *camera) {
  return computeReprojectionResidual(point_3D, point_2D, proj_matrix, camera)
      .norm();
}

double computeReprojectionError(const Eigen::Vector3d &point_3D,
                                const Eigen::Vector2d &point_2D,
                                const Eigen::Matrix3x4d &projection_matrix) {
  return computeReprojectionResidual(point_3D, point_2D, projection_matrix)
      .norm();
}

bool hasPointPositiveDepth(const Eigen::Matrix3x4d &proj_matrix,
                           const Eigen::Vector3d &point_3D) {
  const double depth = proj_matrix.row(2).dot(point_3D.homogeneous());
  return depth >= std::numeric_limits<double>::epsilon();
}

double computeNormalizedAngularError(const Eigen::Vector2d &point_2D,
                                     const Eigen::Vector3d &point_3D,
                                     const Eigen::Matrix3x4d &proj_matrix) {
  const Eigen::Vector3d ray1 = point_2D.homogeneous();
  const Eigen::Vector3d ray2 = proj_matrix * point_3D.homogeneous();
  return std::acos(ray1.normalized().dot(ray2.normalized()));
}

} // namespace base
} // namespace my3d