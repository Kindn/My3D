#include "utils/math.h"

namespace my3d {
namespace util {

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &a) {
  Eigen::Matrix3d a_hat;
  a_hat << 0, -a(2), a(1), a(2), 0, -a(0), -a(1), a(0), 0;

  return a_hat;
}

double rad2Deg(const double &val_rad) { return val_rad * 180.0 / M_PI; }

double deg2Rad(const double &val_deg) { return val_deg * M_PI / 180.0; }

size_t computeCombinationNumber(const size_t &n, const size_t &k) {
  if (k == 0 || k == n) {
    return 1;
  }
  if (k > n) {
    return 0;
  }
  const size_t m = std::min(k, n - k);
  long int num = 1, den = 1;
  for (size_t i = 0; i < m; ++i) {
    num *= static_cast<long int>(n - i);
    den *= static_cast<long int>(i + 1);
  }

  // std::cout << "comb: " << num / den << std::endl;
  return static_cast<size_t>(num / den);
}

size_t computeCombinations(const size_t &n, const size_t &k,
                           std::vector<std::vector<size_t>> *combinations) {
  if (combinations) {
    combinations->clear();
  }

  if (k > n || n == 0) {
    return 0;
  } else if (k == n || k == 0) {
    if (combinations) {
      combinations->resize(1);
      std::iota(combinations->at(0).begin(), combinations->at(0).end(),
                static_cast<size_t>(0));
    }
    return 1;
  }

  return 0;
  // TODO
}

double computeGaussian(const double &x, const double &sigma,
                       const double &mean) {
  const double a = 1.0 / (sigma * std::sqrt(2.0 * M_PI));

  return a * std::exp(-0.5 * std::pow((x - mean) / sigma, 2.0));
}

double computeGaussian(const Eigen::VectorXd &x, const Eigen::VectorXd &mean,
                       const Eigen::MatrixXd &covariance) {
  assert(x.size() > 0);
  assert(x.size() == mean.size());
  assert(x.size() == covariance.rows());
  assert(covariance.rows() == covariance.cols());

  const Eigen::Index num_dims = x.size();
  const Eigen::MatrixXd information = covariance.inverse();
  const double a = 1.0 / std::sqrt(std::pow(2.0 * M_PI, num_dims) *
                                   covariance.determinant());

  return a * std::exp(-0.5 * x.dot(information * x));
}

} // namespace util
} // namespace my3d