/*
 * filename: estimator_utils.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Utilities for estimators.
 */

#ifndef _MY3D_ESTIMATOR_ESTIMATOR_UTILS_H_
#define _MY3D_ESTIMATOR_ESTIMATOR_UTILS_H_

#include "utils/eigen_types.h"

namespace my3d {
namespace estimator {

/**
 * @brief Compute squraed Sampson error for a fundamatal/essential matrix.
 * The model is point2.tranpose() * E * point1 == 0
*/
double computeSquaredSampsonError(const Eigen::Vector2d &point1, 
                                  const Eigen::Vector2d &point2, 
                                  const Eigen::Matrix3d &E);

void computeSquaredSampsonError(const EigenVec<Eigen::Vector2d> &points1, 
                                 const EigenVec<Eigen::Vector2d> &points2, 
                                 const Eigen::Matrix3d &E, 
                                 std::vector<double> &errors);

}
}

#endif // _MY3D_ESTIMATOR_ESTIMATOR_UTILS_H_
