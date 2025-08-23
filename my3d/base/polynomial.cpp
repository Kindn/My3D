/*
 * filename: polynomial.cpp
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include <iostream>

#include "base/polynomial.h"

namespace my3d {
namespace base {

void removeLeadingZeros(const Eigen::VectorXd &all_coeffs, 
                        Eigen::VectorXd &coeffs) {
    if (all_coeffs.size() == 0) {
        return;
    }

    const Eigen::VectorXd original = all_coeffs;
    Eigen::Index num_leading_zeros = 0;
    for (Eigen::Index i = (long)coeffs.size() - 1L; i >= 0; --i) {
        if (coeffs(i) == 0) {
            ++num_leading_zeros;
        } else {
            break;
        }
    }

    coeffs = original.head(original.size() - num_leading_zeros);
}

void removeTrailingZeros(const Eigen::VectorXd &all_coeffs, 
                         Eigen::VectorXd &coeffs) {
    if (all_coeffs.size() == 0) {
        return;
    }

    const Eigen::VectorXd original = all_coeffs;
    Eigen::Index num_trailing_zeros = 0;
    for (Eigen::Index i = 0; i < (long)coeffs.size() - 1L; ++i) {
        if (coeffs(i) == 0) {
            ++num_trailing_zeros;
        } else {
            break;
        }
    }

    coeffs = original.tail(original.size() - num_trailing_zeros);
}

bool solveLinearPolynomial(const Eigen::VectorXd &coeffs, 
                           Eigen::VectorXd &roots_real, 
                           Eigen::VectorXd &roots_imag) {
    Eigen::VectorXd effective_coeffs;
    removeLeadingZeros(coeffs, effective_coeffs);

    if ((long)effective_coeffs.size() < 2L) {
        return false; 
    }

    roots_real.resize(1);
    roots_imag.resize(1);
    roots_real(0) = -effective_coeffs(0) / effective_coeffs(1);
    roots_imag(0) = 0;

    return true;
}

bool solveQuadraticPolynomial(const Eigen::VectorXd &coeffs, 
                              Eigen::VectorXd &roots_real, 
                              Eigen::VectorXd &roots_imag) {
    Eigen::VectorXd effective_coeffs;
    removeLeadingZeros(coeffs, effective_coeffs);

    if ((long)effective_coeffs.size() < 3L) {
        return false; 
    }

    roots_real.resize(2);
    roots_imag.resize(2);
    const double a = effective_coeffs(2);
    const double b = effective_coeffs(1);
    const double c = effective_coeffs(0);
    const double delta = b * b - 4.0 * a * c;

    if (delta >= 0) {
        roots_real(0) = (-b - std::sqrt(delta)) / (2.0 * a);
        roots_real(1) = (-b + std::sqrt(delta)) / (2.0 * a);
        roots_imag(0) = 0.0;
        roots_imag(1) = 0.0;
    } else {
        roots_real(0) = -b / (2.0 * a);
        roots_real(1) = -b / (2.0 * a);
        roots_imag(0) = -std::sqrt(-delta) / (2.0 * a);
        roots_imag(1) = std::sqrt(-delta) / (2.0 * a);
    }   

    return true;
}

bool solvePolynomialCompanionMatrix(const Eigen::VectorXd &coeffs, 
                                    Eigen::VectorXd &roots_real, 
                                    Eigen::VectorXd &roots_imag) {
    Eigen::VectorXd effective_coeffs;
    removeLeadingZeros(coeffs, effective_coeffs);
    const int degree = static_cast<int>(effective_coeffs.size()) - 1;

    if (degree <= 0) {
        std::cout << __PRETTY_FUNCTION__ << "Error: Degree is not positive. " << std::endl;
        return false;
    } else if (degree == 1) {
        std::cout << __PRETTY_FUNCTION__ << "Waring: Directly solving linear polynomial. " << std::endl;
        return solveLinearPolynomial(effective_coeffs, roots_real, roots_imag);
    } else if (degree == 2) {
        std::cout << __PRETTY_FUNCTION__ << "Waring: Directly solving quadratic polynomial. " << std::endl;
        return solveQuadraticPolynomial(effective_coeffs, roots_real, roots_imag);
    }

    roots_real = Eigen::VectorXd::Zero(degree);
    roots_imag = Eigen::VectorXd::Zero(degree);
    removeTrailingZeros(effective_coeffs, effective_coeffs);
    const int effective_degree = static_cast<int>(effective_coeffs.size()) - 1;
    if (effective_degree == 0) { // Only zero is a root
        return true;
    } else if (effective_degree == 1) {
        Eigen::VectorXd reals, imags;
        solveLinearPolynomial(effective_coeffs, reals, imags);
        roots_real.head(1) = reals;
        roots_imag.head(1) = imags;
        
        return true;
    } else if (effective_degree == 2) {
        Eigen::VectorXd reals, imags;
        solveQuadraticPolynomial(effective_coeffs, reals, imags);
        roots_real.head(2) = reals;
        roots_imag.head(2) = imags;

        return true;
    }

    /* Construct companion matrix */
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(effective_degree, effective_degree);
    C.bottomLeftCorner(effective_degree - 1, effective_degree - 1).setIdentity();
    C.rightCols<1>() = -effective_coeffs.head(effective_degree) / effective_coeffs(effective_degree);
    /* Solve for the eigenvalues of companion matrix */
    Eigen::EigenSolver<Eigen::MatrixXd> solver(C, false);
    if (solver.info() != Eigen::Success) {
        // std::cout << __PRETTY_FUNCTION__ << "Warning: EigenSolver failed" << std::endl;
        return false;
    }
    roots_real.head(effective_degree) = solver.eigenvalues().real();
    roots_imag.head(effective_degree) = solver.eigenvalues().imag();

    return true;
}

bool refineRoots(const Eigen::VectorXd &coeffs, 
                 Eigen::VectorXd &real_roots, 
                 uint32_t iter_num) {
    for (uint32_t iter = 0; iter < iter_num; ++iter) {
        for (Eigen::Index root_idx = 0; root_idx < (long)real_roots.size(); ++root_idx) {
            double val = base::computePolynomial(coeffs, real_roots(root_idx));
            double deriv = base::computePolynomialDerivative(coeffs, real_roots(root_idx), 1);
            if (std::abs(deriv) > 1e-5) {
                real_roots(root_idx) -= val / deriv;
            }
        }
    }

    return true;
}

double computePolynomial(const Eigen::VectorXd &coeffs, 
                         double x) {
    double val = 0;
    double xi = 1;
    for (int i = 0; i < coeffs.size(); ++i) {
        val += coeffs(i) * xi;
        xi *= x;
    }

    return val;
}

void derivativeCoefficients(const Eigen::VectorXd &coeffs, 
                            Eigen::VectorXd &deriv_coeffs, 
                            uint32_t s) {
    const Eigen::Index degree = (long)coeffs.size() - 1L;
    if (degree <= 0 && s > degree) {
        deriv_coeffs = Eigen::VectorXd::Zero(1);
        return;
    }

    deriv_coeffs.resize(coeffs.size() - s);
    for (int i = static_cast<int>(s); i < coeffs.size(); ++i) {
        double coeff = 1.0;
        for (int j = 0; j < static_cast<int>(s); ++j) {
            coeff *= static_cast<double>(i - j);
        }
        deriv_coeffs(i - s) = coeff * coeffs(i);
    }
}

double computePolynomialDerivative(const Eigen::VectorXd &coeffs, 
                                   double x, 
                                   uint32_t s) {
    Eigen::VectorXd deriv_coeffs;
    derivativeCoefficients(coeffs, deriv_coeffs, s);
    return computePolynomial(deriv_coeffs, x);
}

}
}
