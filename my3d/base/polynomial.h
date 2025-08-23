/*
 * filename: polynomial.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_POLYNOMIAL_H_
#define _MY3D_BASE_POLYNOMIAL_H_

#include "utils/eigen_types.h"

namespace my3d {
namespace base {

void removeLeadingZeros(const Eigen::VectorXd &all_coeffs, 
                        Eigen::VectorXd &coeffs);

void removeTrailingZeros(const Eigen::VectorXd &all_coeffs, 
                         Eigen::VectorXd &coeffs);

bool solveLinearPolynomial(const Eigen::VectorXd &coeffs, 
                           Eigen::VectorXd &roots_real, 
                           Eigen::VectorXd &roots_imag);

bool solveQuadraticPolynomial(const Eigen::VectorXd &coeffs, 
                              Eigen::VectorXd &roots_real, 
                              Eigen::VectorXd &roots_imag);

/**
 * @brief Find the n roots of coeffs(0) + coeffs(1) * x + ... + coeffs(n) * x^n = 0 
 * with n the index of the last non-zero entry of coeffs. 
 * 
 * @param coeffs             the coefficients of the polynomial to be solved
 * @param roots_real[out]    the real parts of the n roots
 * @param roots_image[out]    the imaginary parts of the n roots
 * @retval false if there is no root or there are infs/nans in roots, true otherwise
*/
bool solvePolynomialCompanionMatrix(const Eigen::VectorXd &coeffs, 
                                    Eigen::VectorXd &roots_real, 
                                    Eigen::VectorXd &roots_imag);

/**
 * @brief Refine the real roots of of coeffs(0) + coeffs(1) * x + ... + coeffs(n) * x^n = 0 
 * with n == coeffs.size() using Newton method.
*/
bool refineRoots(const Eigen::VectorXd &coeffs, 
                 Eigen::VectorXd &real_roots, 
                 uint32_t iter_num = 2U);

/**
 * @brief Compute value of the real polynomial coeffs(0) + coeffs(1) * x + ... + coeffs(n) * x^n = 0 
 * with n == coeffs.size() using Newton method.
*/
double computePolynomial(const Eigen::VectorXd &coeffs, 
                         double x);

/**
 * @brief Compute coefficients of s-th derivative of the real polynomial coeffs(0) + coeffs(1) * x 
 * + ... + coeffs(n) * x^n = 0
*/
void derivativeCoefficients(const Eigen::VectorXd &coeffs, 
                            Eigen::VectorXd &deriv_coeffs, 
                            uint32_t s);

/**
 * @brief Compute s-th derivative of the real polynomial coeffs(0) + coeffs(1) * x + ... + coeffs(n) * x^n = 0 
 * with n == coeffs.size() using Newton method.
*/
double computePolynomialDerivative(const Eigen::VectorXd &coeffs, 
                                   double x, 
                                   uint32_t s);

}
}

#endif // _MY3D_BASE_POLYNOMIAL_H_
