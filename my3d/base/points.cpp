/*
 * filename: points.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about points.
 */

#include "base/points.h"

namespace my3d {
namespace base {

void normalize2DPointSet(const EigenVec<Eigen::Vector2d> &src,
                         EigenVec<Eigen::Vector2d> &dst, Eigen::Vector2d &t,
                         double &scale) {
  assert(!src.empty());

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      tmp_src = src;
  dst.clear();
  dst.reserve(tmp_src.size());

  t.setZero();
  for (auto &pnt : tmp_src) {
    t += pnt;
  }
  t /= tmp_src.size();
  t = -t;

  double sum_sqr_norm = 0.0;
  for (auto &pnt : tmp_src) {
    dst.push_back(pnt + t);
    sum_sqr_norm += dst.back().squaredNorm();
  }
  sum_sqr_norm /= src.size();
  scale = std::sqrt(2.0 / sum_sqr_norm);
  for (auto &pnt : dst) {
    pnt *= scale;
  }
}

void normalize2DPointSet(const EigenVec<Eigen::Vector2d> &src,
                         EigenVec<Eigen::Vector2d> &dst,
                         Eigen::Matrix3d &transform_matrix) {
  Eigen::Vector2d t;
  double scale;
  normalize2DPointSet(src, dst, t, scale);

  Eigen::Matrix3d t_mat, scale_mat;
  t_mat.setIdentity();
  t_mat.block<2, 1>(0, 2) = t;
  scale_mat.setIdentity();
  scale_mat.block<2, 2>(0, 0) *= scale;

  transform_matrix = scale_mat * t_mat;
}

} // namespace base
} // namespace my3d
