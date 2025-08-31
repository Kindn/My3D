/*
 * filename: TriangulationEstimator.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "estimators/TriangulationEstimator.h"

namespace my3d {
namespace estimator {

void TriangulationEstimator::estimate(const VecXData &x_data,
                                      const VecYData &y_data,
                                      EigenVec<ModelType> &models) {
  assert(!x_data.empty());
  assert(!y_data.empty());
  assert(x_data.size() == y_data.size());
  assert(x_data.size() >= kMinNumSamples &&
         "More than 2 points are needed to estimate triangulation. ");

  const size_t num_points = x_data.size();

  models.clear();
  Eigen::Vector3d triangulated_point;
  if (x_data.size() == 2) {
    triangulated_point = base::triangulatePoint(
        y_data[0].proj_matrix, y_data[1].proj_matrix,
        x_data[0].point_normalized, x_data[1].point_normalized);
  } else {
    EigenVec<Eigen::Vector2d> points_2D;
    EigenVec<Eigen::Matrix3x4d> proj_matrices;
    points_2D.reserve(num_points);
    proj_matrices.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      points_2D.push_back(x_data[i].point_normalized);
      proj_matrices.push_back(y_data[i].proj_matrix);
    }
    triangulated_point =
        base::triangulatePointMultiView(proj_matrices, points_2D);
  }
  for (const auto &pose : y_data) {
    if (!base::hasPointPositiveDepth(pose.proj_matrix, triangulated_point)) {
      return;
    }
  }
  for (int64_t i = 0; i < static_cast<int64_t>(y_data.size()) - 1; ++i) {
    for (int64_t j = i + 1; j < static_cast<int64_t>(y_data.size()); ++j) {
      const Eigen::Vector3d ti = y_data[i].proj_center;
      const Eigen::Vector3d tj = y_data[j].proj_center;
      const double tri_angle_rad =
          base::computeTriangulationAngle(ti, tj, triangulated_point);
      if (std::abs(tri_angle_rad) >=
          std::abs(util::deg2Rad(min_tri_angle_deg_))) {
        // std::cout << "small tri angle" << std::endl;
        models.push_back(triangulated_point);
        return;
      }
    }
  }
}

double TriangulationEstimator::computeRMSE(const VecXData &x_data,
                                           const VecYData &y_data,
                                           const ModelType &model) {
  assert(x_data.size() == y_data.size());
  if (x_data.empty()) {
    return 0.0;
  }

  double error2_sum = 0;
  for (size_t i = 0; i < x_data.size(); ++i) {
    double error = computeError(x_data[i], y_data[i], model);
    error2_sum += error * error;
  }

  return std::sqrt(error2_sum / x_data.size());
}

double TriangulationEstimator::computeError(const XDataType &x_data,
                                            const YDataType &y_data,
                                            const ModelType &model) {
  if (error_type_ == EstimationErrorType::ANGULAR_ERROR) {
    return base::computeNormalizedAngularError(x_data.point_normalized, model,
                                               y_data.proj_matrix);
  } else {
    return computeResidual(x_data, y_data, model).norm();
  }
}

void TriangulationEstimator::computeError(const VecXData &x_data,
                                          const VecYData &y_data,
                                          const ModelType &model,
                                          std::vector<double> &errors) {
  assert(x_data.size() == y_data.size());

  const size_t num_pnts = x_data.size();
  errors.resize(num_pnts);
  for (size_t i = 0; i < num_pnts; ++i) {
    errors[i] = computeError(x_data[i], y_data[i], model);
  }
}

Eigen::Vector2d TriangulationEstimator::computeResidual(
    const XDataType &x_data, const YDataType &y_data, const ModelType &model) {
  return (y_data.proj_matrix * model.homogeneous()).hnormalized() -
         x_data.point_normalized;
}

RobustTriangulationEstimator::Report estimateRobustTriangulation(
    const typename estimator::RANSACConfig &config,
    const typename TriangulationEstimator::VecXData &x_data,
    const typename TriangulationEstimator::VecYData &y_data) {
  assert(!x_data.empty());
  assert(!y_data.empty());
  assert(x_data.size() == y_data.size());
  assert(x_data.size() >= TriangulationEstimator::kMinNumSamples &&
         "More than 2 points are needed to estimate triangulation. ");

  RobustTriangulationEstimator estimator(config);
  return estimator.estimate(x_data, y_data);
}

} // namespace estimator
} // namespace my3d
