/*
 * filename: RANSAC.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_ESTIMATOR_LORANSAC_H_
#define _MY3D_ESTIMATOR_LORANSAC_H_

#include "estimators/RANSAC.h"

namespace my3d {
namespace estimator {

template <typename EstimatorType, typename LocalEstimatorType>
class LORANSAC : public RANSAC<EstimatorType> {
public:
  using typename RANSAC<EstimatorType>::Report;

  explicit LORANSAC(const RANSACConfig &config)
      : RANSAC<EstimatorType>(config) {}

  ~LORANSAC() {}

  LocalEstimatorType &getLocalEstimator() { return local_estimator_; }

  Report estimate(const typename EstimatorType::VecXData &x_data,
                  const typename EstimatorType::VecYData &y_data,
                  const bool &exhaust = false);

protected:
  LocalEstimatorType local_estimator_;
};

template <typename EstimatorType, typename LocalEstimatorType>
typename LORANSAC<EstimatorType, LocalEstimatorType>::Report
LORANSAC<EstimatorType, LocalEstimatorType>::estimate(
    const typename EstimatorType::VecXData &x_data,
    const typename EstimatorType::VecYData &y_data, const bool &exhaust) {
  assert(x_data.size() == y_data.size());
  assert(x_data.size() >= EstimatorType::kMinNumSamples &&
         x_data.size() >= LocalEstimatorType::kMinNumSamples);

  const size_t num_points = x_data.size();

  Report report;
  size_t dyn_max_iter_num = this->config_.max_iter_num;
  // double best_inlier_error = std::numeric_limits<double>::infinity();
  double best_inlier_error_sum = std::numeric_limits<double>::infinity();
  size_t max_num_inliers = 0;

  report.num_iter = 0;
  report.inlier_mask.clear();
  bool found = false;
  // std::cout << "---------------------" << std::endl;
  bool abort = false;

  while (report.num_iter < this->config_.max_iter_num) {
    std::vector<size_t> sampled_idx;
    RANSAC<EstimatorType>::sampleMinimumPointSet(num_points, sampled_idx);
    typename EstimatorType::VecXData sampled_x_data;
    typename EstimatorType::VecYData sampled_y_data;
    for (const auto &sampled_id : sampled_idx) {
      sampled_x_data.emplace_back(x_data[sampled_id]);
      sampled_y_data.emplace_back(y_data[sampled_id]);
    }

    EigenVec<typename EstimatorType::ModelType> models;
    this->estimator_.estimate(sampled_x_data, sampled_y_data, models);

    for (const auto &model : models) {
      std::vector<bool> inlier_mask;
      std::vector<double> errors;
      const size_t num_inliers =
          this->countInliers(x_data, y_data, model, inlier_mask, errors);
      const double inlier_ratio =
          static_cast<double>(num_inliers) / static_cast<double>(num_points);
      const double inlier_error =
          this->computeInlierRMSE(x_data, y_data, inlier_mask, model);
      const double inlier_error_sum =
          this->computeInlierErrorSum(x_data, y_data, inlier_mask, model);
      dyn_max_iter_num = this->computeDynamicMaxIterNum(inlier_ratio);
      dyn_max_iter_num = std::max(dyn_max_iter_num, this->config_.min_iter_num);

      // Do local optimization if better than all previous subsets
      if (num_inliers > max_num_inliers ||
          (num_inliers == max_num_inliers &&
           inlier_error_sum < best_inlier_error_sum)) {
        best_inlier_error_sum = inlier_error_sum;
        // best_inlier_error = inlier_error;
        max_num_inliers = num_inliers;
        report.num_inliers = max_num_inliers;
        report.inlier_error = inlier_error;
        report.inlier_error_sum = inlier_error_sum;
        report.model = model;
        report.inlier_mask = inlier_mask;
        report.errors = errors;
        found = true;

        if (num_inliers > EstimatorType::kMinNumSamples &&
            num_inliers >= LocalEstimatorType::kMinNumSamples) {
          const size_t kMaxNumLocalIters = 10;
          std::vector<bool> local_inlier_mask = inlier_mask;
          typename LocalEstimatorType::VecXData local_x_data;
          typename LocalEstimatorType::VecYData local_y_data;
          for (size_t iter = 0; iter < kMaxNumLocalIters; ++iter) {
            // Prepare data
            local_x_data.clear();
            local_y_data.clear();
            for (size_t i = 0; i < num_points; ++i) {
              if (report.inlier_mask[i]) {
                local_x_data.emplace_back(x_data[i]);
                local_y_data.emplace_back(y_data[i]);
              }
            }

            // Local estimation
            const size_t prev_max_num_inliers = max_num_inliers;
            EigenVec<typename LocalEstimatorType::ModelType> local_models;
            local_estimator_.estimate(local_x_data, local_y_data, local_models);
            for (const auto &local_model : local_models) {
              std::vector<bool> local_inlier_mask;
              std::vector<double> local_errors;
              const size_t num_local_inliers = this->countInliers(
                  x_data, y_data, local_model, local_inlier_mask, local_errors);
              const double local_inlier_error = this->computeInlierRMSE(
                  x_data, y_data, local_inlier_mask, local_model);
              const double local_inlier_error_sum = this->computeInlierErrorSum(
                  x_data, y_data, local_inlier_mask, local_model);
              if (num_local_inliers > max_num_inliers ||
                  (num_local_inliers == max_num_inliers &&
                   local_inlier_error_sum < best_inlier_error_sum)) {
                best_inlier_error_sum = inlier_error_sum;
                max_num_inliers = num_local_inliers;
                report.num_inliers = max_num_inliers;
                report.inlier_error = local_inlier_error;
                report.inlier_error_sum = local_inlier_error_sum;
                report.model = local_model;
                report.inlier_mask = local_inlier_mask;
                report.errors = local_errors;
              }
            }

            if (max_num_inliers <= prev_max_num_inliers) {
              break;
            }
          }
        }
      }
    }

    if (abort) {
      break;
    }

    report.num_iter++;
    if (this->config_.verbose) {
      std::cout << "\r[RANSAC] iterations: " << report.num_iter << "/"
                << this->config_.max_iter_num;
    }
    if (report.num_iter > dyn_max_iter_num &&
        report.num_iter > this->config_.min_iter_num) {
      abort = true;
      break;
    }
  }

  if (this->config_.verbose) {
    std::cout << std::endl;
  }

  if (!found) {
    report.success = false;
  } else {
    report.success = true;
  }

  return report;
}

} // namespace estimator
} // namespace my3d

#endif // _MY3D_ESTIMATOR_LORANSAC_H_
