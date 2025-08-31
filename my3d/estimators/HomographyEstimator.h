/*
 * filename: HomographyEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_ESTIMATOR_HOMOGRAPHY_ESTIMATOR_
#define _MY3D_ESTIMATOR_HOMOGRAPHY_ESTIMATOR_

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <eigen3/Eigen/Eigen>

#include "base/points.h"
#include "base/pose.h"

namespace my3d {
namespace estimator {

class HomographyEstimator {
public:
  const static size_t kMinNumSamples = 4;

  typedef Eigen::Vector2d XDataType;
  typedef Eigen::aligned_allocator<Eigen::Vector2d> XAllocType;
  typedef std::vector<XDataType, XAllocType> VecXData;
  typedef Eigen::Vector2d YDataType;
  typedef Eigen::aligned_allocator<Eigen::Vector2d> YAllocType;
  typedef std::vector<YDataType, YAllocType> VecYData;
  typedef Eigen::Matrix3d ModelType;

  explicit HomographyEstimator();
  ~HomographyEstimator();

public:
  /**
   * @brief Estimate the homography H which satisfies y = Hx
   *
   * @param x_data points in plane 1
   * @param y_data points in plane 2
   */
  static void estimate(const VecXData &x_data, const VecYData &y_data,
                       EigenVec<ModelType> &models);

  /**
   * @brief Total RMSE.
   */
  static double computeRMSE(const VecXData &x_data, const VecYData &y_data,
                            const ModelType &model);

  /**
   * @brief Euclidean norm of residual.
   */
  static double computeError(const XDataType &x_data, const YDataType &y_data,
                             const ModelType &model);

  /**
   * @brief Euclidean norm of residual.
   */
  static void computeError(const VecXData &x_data, const VecYData &y_data,
                           const ModelType &model, std::vector<double> &errors);

  /**
   * @brief residual = y_data - model * x_data
   */
  static Eigen::Vector2d computeResidual(const XDataType &x_data,
                                         const YDataType &y_data,
                                         const ModelType &model);
};

double computeOppositeOfMinor(const Eigen::Matrix3d &matrix, const size_t row,
                              const size_t col);

/**
 * @brief Decompose a homography using method in
 *      Malis, Ezio, and Manuel Vargas. "Deeper understanding of the homography
 *      decomposition for vision-based control." (2007): 90.
 *
 * The homography H is expressed as: H = K2 * (R - t.transpose() * n) *
 * K1.inverse() / scale The first pose is assumed to be P = [I | 0]. Note that
 * the homography is plane-induced if `R.size() == t.size() == n.size() == 4`.
 * If `R.size() == t.size() == n.size() == 1` the homography is pure-rotational.
 */
void decomposeHomography(const Eigen::Matrix3d &H, const Eigen::Matrix3d &K1,
                         const Eigen::Matrix3d &K2,
                         EigenVec<Eigen::Matrix3d> &R,
                         EigenVec<Eigen::Vector3d> &t,
                         EigenVec<Eigen::Vector3d> &n);

Eigen::Matrix3d homographyFromPose(const Eigen::Matrix3d &K1,
                                   const Eigen::Matrix3d &K2,
                                   const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &t,
                                   const Eigen::Vector3d &n, double d);

/**
 * @brief Recover camera pose from the given homography H mapping points
 * in camera 1 to camera 2. The pose of camera 1 is set to [I | 0] and the
 * pose of camera 2 is [R | t].
 *
 * @param H               the given homography
 * @param K1              calibration matrix of camera 1
 * @param K2              calibration matrix of
 * @param points1         pixel coordinates of points in camera 1
 * @param points2         pixel coordinates of points in camera 2
 * @param R[out]          the result rotation
 * @param t[out]          the result up-to-scale translation
 * @param points_3d[out]  the 3D coordinates of points that satisfy the
 * cheirality constraint
 */
void poseFromHomography(const Eigen::Matrix3d &H, const Eigen::Matrix3d &K1,
                        const Eigen::Matrix3d &K2,
                        const EigenVec<Eigen::Vector2d> &points1,
                        const EigenVec<Eigen::Vector2d> &points2,
                        Eigen::Matrix3d &R, Eigen::Vector3d &t,
                        Eigen::Vector3d &n,
                        EigenVec<Eigen::Vector3d> &points_3d);

} // namespace estimator

} // namespace my3d

#endif // _MY3D_ESTIMATOR_HOMOGRAPHY_ESTIMATOR_
