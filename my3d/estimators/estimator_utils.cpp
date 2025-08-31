/*
 * filename: estimator_utils.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Utilities for estimators.
 */

#include "estimators/estimator_utils.h"

namespace my3d {
namespace estimator {

double computeSquaredSampsonError(const Eigen::Vector2d &point1,
                                  const Eigen::Vector2d &point2,
                                  const Eigen::Matrix3d &E) {
  const double e00 = E(0, 0);
  const double e01 = E(0, 1);
  const double e02 = E(0, 2);
  const double e10 = E(1, 0);
  const double e11 = E(1, 1);
  const double e12 = E(1, 2);
  const double e20 = E(2, 0);
  const double e21 = E(2, 1);
  const double e22 = E(2, 2);
  const double p1_0 = point1.x();
  const double p1_1 = point1.y();
  const double p2_0 = point2.x();
  const double p2_1 = point2.y();

  const double Ep1_0 = e00 * p1_0 + e01 * p1_1 + e02;
  const double Ep1_1 = e10 * p1_0 + e11 * p1_1 + e12;
  const double Ep1_2 = e20 * p1_0 + e21 * p1_1 + e22;
  const double Etp2_0 = e00 * p2_0 + e10 * p2_1 + e20;
  const double Etp2_1 = e01 * p2_0 + e11 * p2_1 + e21;
  const double p2tEp1 = p2_0 * Ep1_0 + p2_1 * Ep1_1 + Ep1_2;

  return (p2tEp1 * p2tEp1) /
         (Ep1_0 * Ep1_0 + Ep1_1 * Ep1_1 + Etp2_0 * Etp2_0 + Etp2_1 * Etp2_1);

  // const double error_alg = point2.homogeneous().dot(E *
  // point1.homogeneous()); const double den = (E *
  // point1.homogeneous()).hnormalized().squaredNorm() +
  //                    (E.transpose() *
  //                    point2.homogeneous()).hnormalized().squaredNorm();

  // return error_alg * error_alg / den;
}

void computeSquaredSampsonError(const EigenVec<Eigen::Vector2d> &points1,
                                const EigenVec<Eigen::Vector2d> &points2,
                                const Eigen::Matrix3d &E,
                                std::vector<double> &error2s) {
  assert(points1.size() == points2.size());

  const size_t num_pnts = points1.size();
  error2s.resize(num_pnts);
  for (size_t i = 0; i < num_pnts; ++i) {
    error2s[i] = computeSquaredSampsonError(points1[i], points2[i], E);
  }
}

} // namespace estimator
} // namespace my3d
