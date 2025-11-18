/*
 * filename: math.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Some useful mathematical algorithms 
 */

#ifndef _MY3D_UTIL_MATH_H_
#define _MY3D_UTIL_MATH_H_

#include <iostream>
#include <numeric>
#include <algorithm>

#include "utils/eigen_types.h"

namespace my3d {
namespace util {

/**
 * @brief Get the corresponding skew-symmetric of a.
*/
Eigen::Matrix3d getSkewSymmetric(const Eigen::Vector3d &a);

/**
 * @brief Converts a value in radians to a value in degrees 
 * 
 * @param val_rad  value in radians
 * @return         value in degrees
*/
double rad2Deg(const double &val_rad); 

/**
 * @brief Converts a value in degrees to a value in radians 
 * 
 * @param val_deg  value in degrees
 * @return         value in radians
*/
double deg2Rad(const double &val_deg); 

/**
 * @brief Compute combination number
*/
size_t computeCombinationNumber(const size_t &n,    
                                const size_t &k); 

/**
 * @brief Compute combinations. 
*/
size_t computeCombinations(const size_t &n, 
                           const size_t &k, 
                           std::vector<std::vector<size_t>> *combinations); 

/**
 * @brief Compute linear interpolation 
 * 
 * @retval (1 - a) * x1 + a * x2
*/
template <typename T> 
inline T computeLinearInterpolation(const T &x1, const T &x2, 
                                    const double &a) {
    return (1.0 - a) * x1 + a * x2; 
}

/**
 * @brief Compute linear interpolation
 * 
 * @retval x1 + (s - s1) / (s2 - s1) * (x2 - x1)
*/
template <typename T> 
inline T computeLinearInterpolation(const T &x1, const T &x2, 
                                    const double &s1, const double &s2, 
                                    const double &s) {
    return computeLinearInterpolation<T>(x1, x2, (s - s1) / (s2 - s1));
}

/**
 * @brief Compute bilinear interpolation 
 * 
 * @retval (1 - b) * ((1 - a) * x11 + a * x12) + b * ((1 - a) * x21 + a * x22)
*/
template <typename T>
inline T computeBilinearInterpolation(const T &x11, const T &x12, 
                                      const T &x21, const T &x22, 
                                      const double &a, const double &b) {
    const T x1 = computeLinearInterpolation(x11, x12, a); 
    const T x2 = computeLinearInterpolation(x21, x22, a); 

    return computeLinearInterpolation<T>(x1, x2, b); 
}

/**
 * @brief Compute bilinear interpolation 
 * 
 * @retval (x11 + (s - s1) / (s2 - s1) * x12) + (t - t1) / (t2 - t1) * (x21 + (s - s1) / (s2 - s1) * x22)
*/
template <typename T> 
inline T computeBilinearInterpolation(const T &x11, const T &x12, 
                                      const T &x21, const T &x22, 
                                      const double &s1, const double &s2, 
                                      const double &t1, const double &t2, 
                                      const double &s, const double &t) {
    const double a = (s - s1) / (s2 - s1); 
    const double b = (t - t1) / (t2 - t1); 

    return computeBilinearInterpolation<T>(x11, x12, x21, x22, a, b); 
}

/**
 * @brief Clamp a quantity between lower and upper
*/
template <typename T>
inline T clamp(const T &quantity, 
               const T &lower, const T &upper) {
    return (quantity < lower ? lower : (quantity > upper ? upper : quantity)); 
}

/**
 * @brief Compute value of Gaussian function of scalar variable
 * 
*/ 
double computeGaussian(const double &x, const double &sigma, const double &mean = 0); 

/**
 * @brief Compute value of Gaussian function of vector variable
 * 
*/ 
double computeGaussian(const Eigen::VectorXd &x, 
                       const Eigen::VectorXd &mean, 
                       const Eigen::MatrixXd &covariance);

}

}


#endif
