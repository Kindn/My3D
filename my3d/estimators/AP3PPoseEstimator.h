/*
 * filename: AP3PPoseEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_ESTIMATOR_AP3P_POSE_ESTIMATOR_H_
#define _MY3D_ESTIMATOR_AP3P_POSE_ESTIMATOR_H_

#include <iostream>

#include "base/polynomial.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace estimator {

/**
 * @brief 3D-2D pose estimator using AP3P algorithm.
 * Reference:
 *      T. Ke and S. I. Roumeliotis, "An Efficient Algebraic Solution to the
 * Perspective-Three-Point Problem," 2017 IEEE Conference on Computer Vision and
 * Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 4618-4626,
 * doi: 10.1109/CVPR.2017.491.
 */
class AP3PPoseEstimator {
public:
  const static size_t kMinNumSamples = 3;

  typedef Eigen::Vector3d XDataType;
  typedef EigenVec<Eigen::Vector3d> VecXData;
  typedef Eigen::Vector2d YDataType;
  typedef EigenVec<Eigen::Vector2d> VecYData;
  typedef Eigen::Matrix3x4d ModelType;

  explicit AP3PPoseEstimator(
      const Eigen::Matrix3d &K = Eigen::Matrix3d::Identity())
      : K_{K} {}

  ~AP3PPoseEstimator() {}

  void setK(const Eigen::Matrix3d &K) { K_ = K; }

  /**
   * @brief Compute poses from given 3D-2D correspondences using AP3P algorithm.
   *
   * @param x_data      the 3D coordinates of points in reference frame
   * @param y_data      the corresponding pixel coordinates of x_data
   * @param Rs[out]     the solutions of camera frame's orientation w.r.t
   * reference frame
   * @param ts[out]     the solutions of camera frame's translation w.r.t
   * reference frame
   */
  int computePose(const VecXData &x_data, const VecYData &y_data,
                  EigenVec<Eigen::Matrix3d> &Rs, EigenVec<Eigen::Vector3d> &ts);

  /**
   * @brief Estimate the camera pose from given 3D-2D correspondences using AP3P
   * algorithm.
   *
   * @param x_data      the 3D coordinates of points in reference frame
   * @param y_data      the corresponding pixel coordinates of x_data
   * @retval            the estimated camera pose [R | t]
   */
  void estimate(const VecXData &x_data, const VecYData &y_data,
                EigenVec<ModelType> &models);

  /**
   * @brief Total RMSE.
   */
  double computeRMSE(const VecXData &x_data, const VecYData &y_data,
                     const ModelType &model);

  /**
   * @brief Euclidean norm of residual.
   */
  double computeError(const XDataType &x_data, const YDataType &y_data,
                      const ModelType &model);

  /**
   * @brief Euclidean norms of residuals.
   */
  void computeError(const VecXData &x_data, const VecYData &y_data,
                    const ModelType &model, std::vector<double> &errors);

  /**
   * @brief Reprojection residual
   */
  Eigen::Vector2d computeResidual(const XDataType &x_data,
                                  const YDataType &y_data,
                                  const ModelType &model);

  void setEpsilon(const double &epsilon) { epsilon_ = epsilon; }

private:
  Eigen::Matrix3d K_;
  double epsilon_{1e-3};
};

} // namespace estimator
} // namespace my3d

#endif // _MY3D_ESTIMATOR_AP3P_POSE_ESTIMATOR_H_
