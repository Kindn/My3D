#define TEST_NAME "base/image/polynomial_test"
#include "utils/unit_test.h"

#include "base/polynomial.h"

using namespace my3d;

BOOST_AUTO_TEST_CASE(TestComputePolynomial) {
    {   
        Eigen::VectorXd coeffs(6);
        coeffs << 0, -2, 1, 0, 0, 0; // x * (x - 2)
        BOOST_CHECK_EQUAL(base::computePolynomial(coeffs, 1), -1);
        BOOST_CHECK_EQUAL(base::computePolynomial(coeffs, 5), 15);
    }

    {   
        Eigen::VectorXd coeffs(6);
        coeffs << 3, -2, 1, 7, 6, 1; // x * (x - 2)
        BOOST_CHECK_EQUAL(base::computePolynomial(coeffs, 1), 16);
        BOOST_CHECK_EQUAL(base::computePolynomial(coeffs, 2), 187);
    }
}

BOOST_AUTO_TEST_CASE(TestComputePolynomialDerivative) {
    {   
        Eigen::VectorXd coeffs(6);
        coeffs << 0, -2, 1, 0, 0, 0; // x * (x - 2)
        BOOST_CHECK_EQUAL(base::computePolynomialDerivative(coeffs, 1, 0), -1);
        BOOST_CHECK_EQUAL(base::computePolynomialDerivative(coeffs, 1, 1), 0);
        BOOST_CHECK_EQUAL(base::computePolynomialDerivative(coeffs, 1, 2), 2);
    }

    {   
        Eigen::VectorXd coeffs(6);
        coeffs << 3, -2, 1, 7, 6, 1; // x * (x - 2)
        BOOST_CHECK_EQUAL(base::computePolynomialDerivative(coeffs, 1, 0), 16);
        BOOST_CHECK_EQUAL(base::computePolynomialDerivative(coeffs, 1, 1), 50);
    }
}

BOOST_AUTO_TEST_CASE(TestSolvePolynomialCompanionMatrix) {
    {
        Eigen::VectorXd coeffs(1);
        Eigen::VectorXd roots_real, roots_imag;
        bool ret = base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag);
        BOOST_CHECK(!ret);
    }
    {
        // x - 2
        Eigen::VectorXd coeffs(2); 
        coeffs << -2, 1;
        Eigen::VectorXd roots_real, roots_imag;
        bool ret = base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag);
        BOOST_CHECK(ret);
        BOOST_CHECK(roots_real.size() == roots_imag.size());
        BOOST_CHECK(roots_real.size() == 1);
        BOOST_CHECK_EQUAL(roots_real(0), 2);
        BOOST_CHECK_EQUAL(roots_imag(0), 0);
    }
    {
        // 2 * (x - 2) * (x - 5)
        Eigen::VectorXd coeffs(3);
        coeffs << 20, -14, 2;
        Eigen::VectorXd roots_real, roots_imag;
        bool ret = base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag);
        BOOST_CHECK(ret);
        BOOST_CHECK(roots_real.size() == roots_imag.size());
        BOOST_CHECK(roots_real.size() == 2);
        BOOST_CHECK_EQUAL(roots_real(0), 2);
        BOOST_CHECK_EQUAL(roots_imag(0), 0);
        BOOST_CHECK_EQUAL(roots_real(1), 5);
        BOOST_CHECK_EQUAL(roots_imag(1), 0);
    }
    {
        // x * x * (x - 2) * (x - 5)
        Eigen::VectorXd coeffs(5);
        coeffs << 0, 0, 10, -7, 1;
        Eigen::VectorXd roots_real, roots_imag;
        bool ret = base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag);
        BOOST_CHECK(ret);
        BOOST_CHECK(roots_real.size() == roots_imag.size());
        BOOST_CHECK(roots_real.size() == 4);
        BOOST_CHECK_EQUAL(roots_real(0), 2);
        BOOST_CHECK_EQUAL(roots_imag(0), 0);
        BOOST_CHECK_EQUAL(roots_real(1), 5);
        BOOST_CHECK_EQUAL(roots_imag(1), 0);
        BOOST_CHECK_EQUAL(roots_real(2), 0);
        BOOST_CHECK_EQUAL(roots_imag(2), 0);
        BOOST_CHECK_EQUAL(roots_real(3), 0);
        BOOST_CHECK_EQUAL(roots_imag(3), 0);
    }
    {
        // (x + 1) * (x - 1) * (x - 2) * (x - 3)
        Eigen::VectorXd coeffs(5);
        coeffs << -6, 5, 5, -5, 1;
        Eigen::VectorXd roots_real, roots_imag;
        bool ret = base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag);
        BOOST_CHECK(ret);
        BOOST_CHECK(roots_real.size() == roots_imag.size());
        BOOST_CHECK(roots_real.size() == 4);
        // Check if -1 is in the solution set
        bool exists = false;
        for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
            if (std::abs(roots_real(i) - -1) < 1e-6 && std::abs(roots_imag(i)) < 1e-6) {
                exists = true;
                break;
            }
        }
        BOOST_CHECK(exists);
        // Check if 1 is in the solution set
        exists = false;
        for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
            if (std::abs(roots_real(i) - 1) < 1e-6 && std::abs(roots_imag(i)) < 1e-6) {
                exists = true;
                break;
            }
        }
        BOOST_CHECK(exists);
        // Check if 2 is in the solution set
        exists = false;
        for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
            if (std::abs(roots_real(i) - 2) < 1e-6 && std::abs(roots_imag(i)) < 1e-6) {
                exists = true;
                break;
            }
        }
        BOOST_CHECK(exists);
        // Check if 3 is in the solution set
        exists = false;
        for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
            if (std::abs(roots_real(i) - 3) < 1e-6 && std::abs(roots_imag(i)) < 1e-6) {
                exists = true;
                break;
            }
        }
        BOOST_CHECK(exists);
    }
}
