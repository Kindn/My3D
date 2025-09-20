/*
 * filename: feature_utils.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include <fstream>

#include "base/image/image_processing.h"
#include "estimators/FundamentalMatrixEstimator.h"
#include "estimators/HomographyEstimator.h"
#include "estimators/LORANSAC.h"
#include "utils/io.h"
#include "utils/math.h"

#ifndef _MY3D_FEATURE_FEATURE_UTILS_H_
#define _MY3D_FEATURE_FEATURE_UTILS_H_

namespace my3d {
namespace feature {

/**
 * @brief Class of Gaussian space
 */
class GaussianSpace {
public:
  GaussianSpace();
  explicit GaussianSpace(const base::Image &base_image,
                         const size_t &num_levels, const double &scale_factor,
                         const double &sigma);
  ~GaussianSpace();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
  /**
   * @brief Build Gaussian space from a base image
   */
  bool build(const base::Image &base_image, const size_t &num_levels,
             const double &scale_factor, const double &sigma);

  /**
   * @brief Get the reference of level with given index (const version)
   */
  const Eigen::MatrixXf &operator[](const size_t &level_idx) const {
    return levels_[level_idx];
  }
  // const base::Image &operator [] (const size_t &level_idx) const { return
  // levels_[level_idx]; }

  /**
   * @brief Get the reference of level with given index
   */
  Eigen::MatrixXf &operator[](const size_t &level_idx) {
    return levels_[level_idx];
  }
  // base::Image &operator [] (const size_t &level_idx) { return
  // levels_[level_idx]; }

  /**
   * @brief Get size of the Gaussian space, i.e., number of levels
   */
  size_t getNumLevels() const { return levels_.size(); }

  /**
   * @brief Check whether the Gaussian space is empty
   */
  bool isEmpty() const { return levels_.empty(); }

  /**
   * @brief Get standard deviation of Gauss kernal applied to the
   * base image
   */
  double getSigma(const double level_idx = 0) const {
    return sigma_ * std::pow(scale_factor_, level_idx);
  }

  /**
   * @brief Compute gradient of image
   */
  Eigen::Vector2d computeGradient(const size_t &x, const size_t &y,
                                  const size_t &level_idx) const;

  /**
   * @brief Get width (i.e. number of columns) of the Gaussian space
   */
  size_t getWidth() const { return width_; }

  /**
   * @brief Get height (i.e. number of rows) of the Gaussian space
   */
  size_t getHeight() const { return height_; }

  /**
   * @brief Get the scale factor of sigma
   */
  double getScaleFactor() const { return scale_factor_; }

protected:
  // EigenVec<base::Image> levels_;
  EigenVec<Eigen::MatrixXf> levels_;
  size_t width_{0};
  size_t height_{0};
  double sigma_{1.0};
  double scale_factor_{1.0};
};

/**
 * @brief Class of difference-of-Gaussian space
 */
class DoGSpace {
public:
  DoGSpace();
  explicit DoGSpace(const GaussianSpace &gaussian_space);
  explicit DoGSpace(const base::Image &base_image,
                    const size_t &num_valid_dog_levels,
                    const double &scale_factor, const double &sigma);
  ~DoGSpace();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
  /**
   * @brief Build the DoG space from a Gaussian space
   */
  bool build(const GaussianSpace &gaussian_space);

  /**
   * @brief Build the DoG space from a base image and given number of valid
   * DoG levels
   */
  bool build(const base::Image &base_image, const size_t &num_valid_dog_levels,
             const double &scale_factor, const double &sigma);

  /**
   * @brief Get the reference of level with given index (const version)
   */
  const Eigen::MatrixXf &operator[](const size_t &level_idx) const {
    return levels_[level_idx];
  }

  /**
   * @brief Get the reference of level with given index
   */
  Eigen::MatrixXf &operator[](const size_t &level_idx) {
    return levels_[level_idx];
  }

  /**
   * @brief Get number of levels of DoG space, i.e., number of levels
   */
  size_t getNumLevels() const { return levels_.size(); }

  /**
   * @brief Get number of valid DoG levels, i.e., max(number_of_levels - 2, 0)
   */
  size_t getNumValidDoGLevels() const;

  /**
   * @brief Check whether the DoG space is empty
   */
  bool isEmpty() const { return levels_.empty(); }

  /**
   * @brief Get standard deviation of Gauss kernal applied to the
   * level_idx level
   */
  double getSigma(const double level_idx = 0) const {
    return sigma_ * std::pow(scale_factor_, level_idx);
  }

  /**
   * @brief Get width (i.e. number of columns) of the Gaussian space
   */
  size_t getWidth() const { return width_; }

  /**
   * @brief Get height (i.e. number of rows) of the Gaussian space
   */
  size_t getHeight() const { return height_; }

  /**
   * @brief Get the scale factor of sigma
   */
  double getScaleFactor() const { return scale_factor_; }

  /**
   * @brief Check whether the given position is an extremum point
   */
  bool isExtremumPoint(const size_t &valid_level_idx, const size_t &x,
                       const size_t &y, const double &thresh = 0.0) const;

  /**
   * @brief Detect extrema in the given valid DoG level
   *
   * @param valid_level    index of valid DoG level (0 <= valid_level <
   * levels_.size() - 2)
   * @param xs[out]        x indices of detected extrema
   * @param ys[out]        y indices of detected extrema
   * @param vals[out]      DoG values of detected extrema
   *
   * @retval number of detected extrema
   */
  size_t detectExtrema(const size_t &valid_level_idx, std::vector<size_t> &xs,
                       std::vector<size_t> &ys, std::vector<double> &vals,
                       const double &thresh = 0.0) const;

  /**
   * @brief Compute the sub-pixel position of extremum point
   */
  bool computeSubPixelExtremumPosition(const size_t &x, const size_t &y,
                                       const size_t &valid_level_idx,
                                       Eigen::Vector3d &position,
                                       double *response,
                                       double *edge_score) const;
  /**
   * @brief Compute gradient at (x, y, valid_level_idx)
   */
  Eigen::Vector3d computeGradient(const size_t &x, const size_t &y,
                                  const size_t &valid_level_idx) const;

  /**
   * @brief Compute Hessian at (x, y, valid_level_idx)
   */
  Eigen::Matrix3d computeHessian(const size_t &x, const size_t &y,
                                 const size_t &valid_level_idx) const;

protected:
  EigenVec<Eigen::MatrixXf> levels_;
  size_t width_{0};
  size_t height_{0};
  double sigma_{1.0};
  double scale_factor_{1.0};
};

// /**
//  * @brief Class of DoG octave
// */
// class DoGOctave {
// public:
//     DoGOctave() = delete;
//     DoGOctave(const base::Image &base_image,
//               const size_t &num_valid_dog_levels,
//               const double &sigma,
//               const double &scale_factor);

// public:
//     /**
//      * @brief Build the DoG space from a base image and given number of valid
//      * DoG levels
//     */
//     bool build(const base::Image &base_image,
//                const size_t &num_valid_dog_levels,
//                const double &scale_factor,
//                const double &sigma);

// private:
//     GaussianSpace gaussian_space_;
//     DoGSpace dog_space_;
//     base::Image base_image_;
//     double sigma_;
//     double scale_factor_;
// };

template <typename KeyPointListType, typename DescriptorListType>
struct FeatureImage {
  size_t image_idx;
  std::string name;
  size_t num_pnts;
  KeyPointListType key_points;
  DescriptorListType descriptors;
  EigenVec<Eigen::Vector2d> points_2D;
  EigenVec<Eigen::Matrix<uint8_t, 3, 1>> colors;
};

// template <typename KeyPointListType, typename DescriptorListType>
// BOOST_CLASS_VERSION(FeatureImage<KeyPointListType, DescriptorListType>, 0)

template <typename KeyPointListType, typename DescriptorListType>
using FeatureImageList =
    EigenVec<FeatureImage<KeyPointListType, DescriptorListType>>;

struct Match {
  Match(int qindex = -1, int tindex = -1, double d = -1.0)
      : query_index(qindex), train_index(tindex), distance(d) {}

  Match(const Match &other) {
    query_index = other.query_index;
    train_index = other.train_index;
    distance = other.distance;
  }

  Match &operator=(const Match &other) {
    query_index = other.query_index;
    train_index = other.train_index;
    distance = other.distance;

    return *this;
  }

  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  size_t query_index;
  size_t train_index;
  double distance{-1.0};
};

typedef std::vector<Match> Matches;

struct PairWiseMatchingInfo {
  size_t query_image_idx;
  size_t train_image_idx;
  std::string query_image_name;
  std::string train_image_name;
  Matches matches;

  bool serialize(std::ofstream &ofs) const;
  bool deserialize(std::ifstream &ifs);
};

feature::Matches getInlierMatches(const EigenVec<Eigen::Vector2d> &key_points1,
                                  const EigenVec<Eigen::Vector2d> &key_points2,
                                  const feature::Matches &matches,
                                  const estimator::RANSACConfig &ransac_config);

} // namespace feature
} // namespace my3d

#endif // _MY3D_FEATURE_FEATURE_UTILS_H_
