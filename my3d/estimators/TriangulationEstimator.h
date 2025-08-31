/*
 * filename: TriangulationEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_ESTIMATOR_TRIANGULATION_ESTIMATOR_H_
#define _MY3D_ESTIMATOR_TRIANGULATION_ESTIMATOR_H_

#include "base/camera/CameraBase.h"
#include "base/camera/projection.h"
#include "base/triangulation.h"
#include "estimators/LORANSAC.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace estimator {

class TriangulationEstimator;

typedef LORANSAC<TriangulationEstimator, TriangulationEstimator>
    RobustTriangulationEstimator;

class TriangulationEstimator {
public:
  const static size_t kMinNumSamples = 2;

  enum EstimationErrorType { ANGULAR_ERROR = 0, REPROJECTION_ERROR = 1 };

  struct PointData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointData(){};
    PointData(const Eigen::Vector2d &_point,
              const Eigen::Vector2d &_point_normalized)
        : point{_point}, point_normalized{_point_normalized} {}

    Eigen::Vector2d point;
    Eigen::Vector2d point_normalized;
  };

  struct PoseData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PoseData() : camera{nullptr} {};
    PoseData(const Eigen::Matrix3x4d &_proj_matrix,
             const Eigen::Vector3d &_proj_center,
             const base::CameraBase *_camera)
        : proj_matrix{_proj_matrix}, proj_center{_proj_center}, camera{
                                                                    _camera} {}

    Eigen::Matrix3x4d proj_matrix;
    Eigen::Vector3d proj_center;
    const base::CameraBase *camera;
  };

  typedef PointData XDataType;
  typedef EigenVec<PointData> VecXData;
  typedef PoseData YDataType;
  typedef EigenVec<PoseData> VecYData;
  typedef Eigen::Vector3d ModelType;

  explicit TriangulationEstimator(const double &min_tri_angle_deg = 5.0,
                                  const EstimationErrorType &error_type =
                                      EstimationErrorType::ANGULAR_ERROR)
      : min_tri_angle_deg_(min_tri_angle_deg){};

  ~TriangulationEstimator(){};

public:
  void setMinTriAngleDeg(const double &min_tri_angle_deg) {
    min_tri_angle_deg_ = min_tri_angle_deg;
  }

  void setEstimationErrorType(const EstimationErrorType &error_type) {
    error_type_ = error_type;
  }

  /**
   * @brief Estimate the homography H which satisfies y = Hx
   *
   * @param x_data points in plane 1
   * @param y_data points in plane 2
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
   * @brief Euclidean norm of residual.
   */
  void computeError(const VecXData &x_data, const VecYData &y_data,
                    const ModelType &model, std::vector<double> &errors);

  /**
   * @brief residual = y_data - model * x_data
   */
  Eigen::Vector2d computeResidual(const XDataType &x_data,
                                  const YDataType &y_data,
                                  const ModelType &model);

protected:
  /* Minimum triangulation angle in degree that any two views should have */
  double min_tri_angle_deg_{5.0};
  /* Estimation error type */
  EstimationErrorType error_type_{EstimationErrorType::ANGULAR_ERROR};
};

RobustTriangulationEstimator::Report estimateRobustTriangulation(
    const typename estimator::RANSACConfig &config,
    const typename TriangulationEstimator::VecXData &x_data,
    const typename TriangulationEstimator::VecYData &y_data);

} // namespace estimator
} // namespace my3d

#endif // _MY3D_ESTIMATOR_TRIANGULATION_ESTIMATOR_H_
