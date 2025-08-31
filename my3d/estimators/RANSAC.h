/*
 * filename: RANSAC.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MY3D_ESTIMATOR_RANSAC_H_
#define _MY3D_ESTIMATOR_RANSAC_H_

#include <iostream>
#include <numeric>
#include <vector>

#include <eigen3/Eigen/Eigen>

#include "utils/RandomNumberGenerator.h"
#include "utils/eigen_types.h"

namespace my3d {
namespace estimator {

struct RANSACConfig {
  double max_inlier_error{0.1};
  double confidence{0.99};
  size_t min_iter_num{0};
  size_t max_iter_num{10000000};
  double min_inlier_ratio{0.1};
  bool verbose{false};
};

template <typename EstimatorType> class RANSAC {
public:
  struct Report {
    bool success = false;
    /* RMSE of inliers */
    double inlier_error = -1.0;
    double inlier_error_sum = -1.0;
    size_t num_iter = 0;
    size_t num_inliers = 0;
    std::vector<bool> inlier_mask;
    std::vector<double> errors;
    typename EstimatorType::ModelType model;
  };

public:
  explicit RANSAC(const RANSACConfig &config);
  ~RANSAC() {}

  size_t countInliers(const typename EstimatorType::VecXData &x_data,
                      const typename EstimatorType::VecYData &y_data,
                      const typename EstimatorType::ModelType &model,
                      std::vector<bool> &inlier_mask,
                      std::vector<double> &errors);

  double computeInlierRMSE(const typename EstimatorType::VecXData &x_data,
                           const typename EstimatorType::VecYData &y_data,
                           const std::vector<bool> &inlier_mask,
                           const typename EstimatorType::ModelType &model);

  double computeInlierErrorSum(const typename EstimatorType::VecXData &x_data,
                               const typename EstimatorType::VecYData &y_data,
                               const std::vector<bool> &inlier_mask,
                               const typename EstimatorType::ModelType &model);

  void sampleMinimumPointSet(size_t num_points,
                             std::vector<size_t> &sampled_idx);

  size_t computeDynamicMaxIterNum(const double &inlier_ratio);

  Report estimate(const typename EstimatorType::VecXData &x_data,
                  const typename EstimatorType::VecYData &y_data,
                  const bool &exhaust = false);

  Report estimateBySampling(const typename EstimatorType::VecXData &x_data,
                            const typename EstimatorType::VecYData &y_data);

  Report estimateByExhaustion(const typename EstimatorType::VecXData &x_data,
                              const typename EstimatorType::VecYData &y_data);

  EstimatorType &getEstimator() { return estimator_; }

protected:
  RANSACConfig config_;
  EstimatorType estimator_;
  util::RandomNumberGenerator rng_;
};

template <typename EstimatorType>
RANSAC<EstimatorType>::RANSAC(const RANSACConfig &config) {
  assert(config.max_inlier_error >= 0);
  assert(config.min_inlier_ratio >= 0.0 && config.min_inlier_ratio <= 1.0);
  assert(config.max_iter_num >= config.min_iter_num);

  config_ = config;
  const size_t dyn_max_iter_num =
      computeDynamicMaxIterNum(config_.min_inlier_ratio);
  config_.max_iter_num =
      std::min<size_t>(dyn_max_iter_num, config_.max_iter_num);
}

template <typename EstimatorType>
size_t RANSAC<EstimatorType>::countInliers(
    const typename EstimatorType::VecXData &x_data,
    const typename EstimatorType::VecYData &y_data,
    const typename EstimatorType::ModelType &model,
    std::vector<bool> &inlier_mask, std::vector<double> &errors) {
  assert(x_data.size() == y_data.size());

  inlier_mask.resize(x_data.size());
  errors.resize(x_data.size());
  size_t count_inliers = 0;
  for (size_t i = 0; i < x_data.size(); ++i) {
    const double error = estimator_.computeError(x_data[i], y_data[i], model);
    if (error <= config_.max_inlier_error) {
      inlier_mask[i] = true;
      ++count_inliers;
    } else {
      inlier_mask[i] = false;
    }
    errors[i] = error;
  }

  return count_inliers;
}

template <typename EstimatorType>
double RANSAC<EstimatorType>::computeInlierRMSE(
    const typename EstimatorType::VecXData &x_data,
    const typename EstimatorType::VecYData &y_data,
    const std::vector<bool> &inlier_mask,
    const typename EstimatorType::ModelType &model) {
  typename EstimatorType::VecXData inliers_x_data;
  typename EstimatorType::VecYData inliers_y_data;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inliers_x_data.push_back(x_data[i]);
      inliers_y_data.push_back(y_data[i]);
    }
  }

  if (!inliers_x_data.empty()) {
    return estimator_.computeRMSE(inliers_x_data, inliers_y_data, model);
  }

  return std::numeric_limits<double>::infinity();
}

template <typename EstimatorType>
double RANSAC<EstimatorType>::computeInlierErrorSum(
    const typename EstimatorType::VecXData &x_data,
    const typename EstimatorType::VecYData &y_data,
    const std::vector<bool> &inlier_mask,
    const typename EstimatorType::ModelType &model) {
  double inlier_error_sum = 0.0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_error_sum +=
          std::pow(estimator_.computeError(x_data[i], y_data[i], model), 2.0);
    }
  }

  if (x_data.empty()) {
    return std::numeric_limits<double>::infinity();
  } else {
    return inlier_error_sum;
  }
}

template <typename EstimatorType>
void RANSAC<EstimatorType>::sampleMinimumPointSet(
    size_t num_points, std::vector<size_t> &sampled_idx) {
  assert(num_points >= EstimatorType::kMinNumSamples);
  assert(num_points > 0);

  sampled_idx.clear();
  sampled_idx.reserve(EstimatorType::kMinNumSamples);
  std::vector<size_t> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);
  const size_t last_idx = num_points - 1UL;
  for (size_t i = 0; i < EstimatorType::kMinNumSamples; ++i) {
    size_t j = rng_.uniformIndex(i, last_idx);
    // std::cout << indices[j] << " ";
    sampled_idx.emplace_back(indices[j]);
    std::swap(indices[i], indices[j]);
  }
  // std::cout << std::endl;
}

template <typename EstimatorType>
size_t
RANSAC<EstimatorType>::computeDynamicMaxIterNum(const double &inlier_ratio) {
  const double nom = 1.0 - config_.confidence;
  if (nom <= 0) {
    return std::numeric_limits<size_t>::max();
  }
  const double denom =
      1.0 - std::pow(inlier_ratio,
                     static_cast<double>(EstimatorType::kMinNumSamples));
  if (denom <= 0) {
    return 1;
  } else if (denom == 1.0) {
    return std::numeric_limits<size_t>::max();
  }

  return static_cast<size_t>(std::ceil(std::log(nom) / std::log(denom)));
}

template <typename EstimatorType>
typename RANSAC<EstimatorType>::Report
RANSAC<EstimatorType>::estimate(const typename EstimatorType::VecXData &x_data,
                                const typename EstimatorType::VecYData &y_data,
                                const bool &exhaust) {
  assert(x_data.size() == y_data.size());
  assert(x_data.size() >= EstimatorType::kMinNumSamples);

  if (exhaust) {
    return estimateByExhaustion(x_data, y_data);
  } else {
    return estimateBySampling(x_data, y_data);
  }
}

template <typename EstimatorType>
typename RANSAC<EstimatorType>::Report
RANSAC<EstimatorType>::estimateBySampling(
    const typename EstimatorType::VecXData &x_data,
    const typename EstimatorType::VecYData &y_data) {
  assert(x_data.size() == y_data.size());
  assert(x_data.size() >= EstimatorType::kMinNumSamples);

  const size_t num_points = x_data.size();

  Report report;
  size_t dyn_max_iter_num = config_.max_iter_num;
  // double best_inlier_error = std::numeric_limits<double>::infinity();
  double best_inlier_error_sum = std::numeric_limits<double>::infinity();
  size_t max_num_inliers = 0;

  report.num_iter = 0;
  report.inlier_mask.clear();
  bool found = false;
  // std::cout << "---------------------" << std::endl;
  bool abort = false;
  while (report.num_iter < config_.max_iter_num) {
    std::vector<size_t> sampled_idx;
    RANSAC<EstimatorType>::sampleMinimumPointSet(num_points, sampled_idx);
    typename EstimatorType::VecXData sampled_x_data;
    typename EstimatorType::VecYData sampled_y_data;
    for (const auto &sampled_id : sampled_idx) {
      sampled_x_data.emplace_back(x_data[sampled_id]);
      sampled_y_data.emplace_back(y_data[sampled_id]);
    }

    EigenVec<typename EstimatorType::ModelType> models;
    estimator_.estimate(sampled_x_data, sampled_y_data, models);

    for (const auto &model : models) {
      std::vector<bool> inlier_mask;
      std::vector<double> errors;
      const size_t num_inliers =
          countInliers(x_data, y_data, model, inlier_mask, errors);
      const double inlier_ratio =
          static_cast<double>(num_inliers) / static_cast<double>(num_points);
      const double inlier_error =
          computeInlierRMSE(x_data, y_data, inlier_mask, model);
      const double inlier_error_sum =
          computeInlierErrorSum(x_data, y_data, inlier_mask, model);
      dyn_max_iter_num = computeDynamicMaxIterNum(inlier_ratio);
      // static_cast<size_t>(std::log(1.0 - config_.confidence) /
      // std::log(1.0 - std::pow(inlier_ratio,
      // static_cast<double>(EstimatorType::kMinNumSamples))));
      dyn_max_iter_num = std::max(dyn_max_iter_num, config_.min_iter_num);
      // std::cout << "inlier number: " << num_inliers << ", inlier ratio: " <<
      // inlier_ratio << ", "
      //           << ", error: " << inlier_error << ", min error: " <<
      //           best_inlier_error << ", "
      //           << report.num_iter << ", " << dyn_max_iter_num << std::endl;
      // std::cout << model << std::endl;
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
      }
    }

    if (abort) {
      break;
    }
    // std::cout << "ransac" << std::endl;
    report.num_iter++;
    if (config_.verbose) {
      std::cout << "\r[RANSAC] iterations: " << report.num_iter << "/"
                << config_.max_iter_num;
    }
    if (report.num_iter > dyn_max_iter_num &&
        report.num_iter > config_.min_iter_num) {
      abort = true;
      break;
    }
  }
  if (config_.verbose) {
    std::cout << std::endl;
  }

  if (!found) { // No valid model was found
    report.success = false;
  } else {
    report.success = true;
  }

  return report;
}

template <typename EstimatorType>
typename RANSAC<EstimatorType>::Report
RANSAC<EstimatorType>::estimateByExhaustion(
    const typename EstimatorType::VecXData &x_data,
    const typename EstimatorType::VecYData &y_data) {
  Report report;

  return report;
}

} // namespace estimator
} // namespace my3d

#endif // _MY3D_ESTIMATOR_RANSAC_H_
