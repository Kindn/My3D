/*
 * filename: points.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about points.
 */

#ifndef _MY3D_BASE_POINTS_H_
#define _MY3D_BASE_POINTS_H_

#include "utils/eigen_types.h"

namespace my3d {
namespace base {

/**
 * @brief Normalize a 2D point-set using translation and scale such that
 * the center of the point-set is the origin and the standard variants of
 * the distances of the points from the origin is sqrt(2).
 *
 * @param src[in]    the source point-set
 * @param dst[out]   the normalized point-set which satisfies dst[i] = scale[i]
 * * (src + t)
 * @param t[out]     the translation vector
 * @param scale[out] the scale factor
 */
void normalize2DPointSet(const EigenVec<Eigen::Vector2d> &src,
                         EigenVec<Eigen::Vector2d> &dst, Eigen::Vector2d &t,
                         double &scale);

/**
 * @brief Normalize a 2D point-set using translation and scale such that
 * the center of the point-set is the origin and the standard variants of
 * the distances of the points from the origin is sqrt(2).
 *
 * @param src[in]    the source point-set
 * @param dst[out]   the normalized point-set which satisfies dst[i] = scale[i]
 * * (src + t)
 * @param transform_matrix[out]     the transform matrix
 */
void normalize2DPointSet(const EigenVec<Eigen::Vector2d> &src,
                         EigenVec<Eigen::Vector2d> &dst,
                         Eigen::Matrix3d &transform_matrix);

} // namespace base
} // namespace my3d

#endif // _MY3D_BASE_POINTS_H_
