/*
 * filename: FundamentalMatrixEstimator.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "estimators/FundamentalMatrixEstimator.h"

namespace my3d {
namespace estimator {

FundamentalMatrixSevenPointEstimator::FundamentalMatrixSevenPointEstimator() {

}

FundamentalMatrixSevenPointEstimator::~FundamentalMatrixSevenPointEstimator() {

}

void 
FundamentalMatrixSevenPointEstimator::estimate(const VecXData &x_data, 
                                               const VecYData &y_data, 
                                               EigenVec<ModelType> &models) {
    assert(x_data.size() >= kMinNumSamples && "Data size should be greater than minimum sample number. ");
    assert(x_data.size() == y_data.size());

    /* Solve for null space */ 
    Eigen::Matrix<double, 7, 9> A; 
    for (size_t i = 0; i < 7; ++i) {
        const double x1 = x_data[i](0); 
        const double x2 = x_data[i](1); 
        const double y1 = y_data[i](0); 
        const double y2 = y_data[i](1); 
        A(i, 0) = y1 * x1; 
        A(i, 1) = y1 * x2; 
        A(i, 2) = y1; 
        A(i, 3) = y2 * x1; 
        A(i, 4) = y2 * x2; 
        A(i, 5) = y2; 
        A(i, 6) = x1; 
        A(i, 7) = x2; 
        A(i, 8) = 1.0; 
    }
    Eigen::JacobiSVD<Eigen::Matrix<double, 7, 9>> svd(A, Eigen::ComputeFullV); 
    const Eigen::Matrix<double, 9, 9> V = svd.matrixV(); 
    Eigen::Matrix<double, 9, 1> f1 = V.col(7), f2 = V.col(8); 
    f1 -= f2; 

    /* Solve for lambda */ 
    const double s1 = f1(4) * f1(8) - f1(5) * f1(7); 
    const double s2 = f1(3) * f1(8) - f1(5) * f1(6); 
    const double s3 = f1(3) * f1(7) - f1(4) * f1(6); 
    const double s4 = f2(4) * f2(8) - f2(5) * f2(7); 
    const double s5 = f2(3) * f2(8) - f2(5) * f2(6); 
    const double s6 = f2(3) * f2(7) - f2(4) * f2(6);
    const double s7 = f1(4) * f2(8) + f1(8) * f2(4) - f1(5) * f2(7) - f1(7) * f2(5); 
    const double s8 = f1(3) * f2(8) + f1(8) * f2(3) - f1(5) * f2(6) - f1(6) * f2(5); 
    const double s9 = f1(3) * f2(7) + f1(7) * f2(3) - f1(4) * f2(6) - f1(6) * f2(4); 

    Eigen::Vector4d coeffs; 
    coeffs(3) = f1(0) * s1 - f1(1) * s2 + f1(2) * s3; 
    coeffs(2) = f1(0) * s7 + f2(0) * s1 - 
                f1(1) * s8 - f2(1) * s2 + 
                f1(2) * s9 + f2(2) * s3; 
    coeffs(1) = f1(0) * s4 + f2(0) * s7 - 
                f1(1) * s5 - f2(1) * s8 + 
                f1(2) * s6 + f2(2) * s9; 
    coeffs(0) = f2(0) * s4 - f2(1) * s5 + f2(2) * s6; 
    
    models.clear(); 
    Eigen::VectorXd roots_real, roots_imag; 
    if (!base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag)) {
        return; 
    }

    /* Extract models */ 
    for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
        const double kMaxRootImag = 1e-10; 
        if (std::abs(roots_imag(i)) > kMaxRootImag) {
            continue; 
        }

        const double lambda = roots_real(i); 
        const Eigen::Matrix<double, 9, 1> f = lambda * f1 + f2; 
        ModelType model; 
        model << f(0), f(1), f(2), 
                 f(3), f(4), f(5), 
                 f(6), f(7), f(8); 
        
        const double kEps = 1e-10; 
        if (std::abs(model(2, 2)) < kEps) {
            continue; 
        }
        model /= model(2, 2); 
        models.push_back(model); 
    }
}

double 
FundamentalMatrixSevenPointEstimator::computeRMSE(const VecXData &x_data, 
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

double 
FundamentalMatrixSevenPointEstimator::computeError(const XDataType &x_data, 
                                                   const YDataType &y_data, 
                                                   const ModelType &model) {
    return std::sqrt(computeSquaredSampsonError(x_data, y_data, model));                                                 
}

void 
FundamentalMatrixSevenPointEstimator::computeError(const VecXData &x_data, 
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

FundamentalMatrixEightPointEstimator::FundamentalMatrixEightPointEstimator() {

}

FundamentalMatrixEightPointEstimator::~FundamentalMatrixEightPointEstimator() {

}

void 
FundamentalMatrixEightPointEstimator::estimate(const VecXData &x_data, 
                                               const VecYData &y_data, 
                                               EigenVec<ModelType> &models) {
    assert(x_data.size() >= kMinNumSamples && "Data size should be greater than minimum sample number. ");
    assert(x_data.size() == y_data.size());

    const size_t num_pnts = x_data.size();

    /* Normalize the xy points. */
    Eigen::Matrix3d transform_x, transform_y;
    VecXData x_data_normalized, y_data_normalized;
    base::normalize2DPointSet(x_data, x_data_normalized, transform_x);
    base::normalize2DPointSet(y_data, y_data_normalized, transform_y);

    /* Solve for the initial E. */
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(num_pnts, 9);
    for (size_t i = 0; i < num_pnts; ++i) {
        A(i, 0) = y_data_normalized[i](0) * x_data_normalized[i](0);
        A(i, 1) = y_data_normalized[i](0) * x_data_normalized[i](1);
        A(i, 2) = y_data_normalized[i](0);
        A(i, 3) = y_data_normalized[i](1) * x_data_normalized[i](0);
        A(i, 4) = y_data_normalized[i](1) * x_data_normalized[i](1);
        A(i, 5) = y_data_normalized[i](1);
        A(i, 6) = x_data_normalized[i](0);
        A(i, 7) = x_data_normalized[i](1);
        A(i, 8) = 1.0;
    }
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd_A(A, Eigen::ComputeFullV);
    Eigen::VectorXd nullspace_A = svd_A.matrixV().col(8);
    Eigen::Map<Eigen::Matrix3d> F_normt(nullspace_A.data());
    
    /* Enforce the singularity constraint. */
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_F_norm(F_normt.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd_F_norm.singularValues();
    singular_values(2) = 0.0;
    const Eigen::Matrix3d F_norm = svd_F_norm.matrixU() * singular_values.asDiagonal() * svd_F_norm.matrixV().transpose(); 
    const Eigen::Matrix3d F = transform_y.transpose() * F_norm * transform_x; 
    
    models.clear();
    models.push_back(F); 
}

double 
FundamentalMatrixEightPointEstimator::computeRMSE(const VecXData &x_data, 
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

double 
FundamentalMatrixEightPointEstimator::computeError(const XDataType &x_data, 
                                                   const YDataType &y_data, 
                                                   const ModelType &model) {
    return std::sqrt(computeSquaredSampsonError(x_data, y_data, model));                                                 
}

void 
FundamentalMatrixEightPointEstimator::computeError(const VecXData &x_data, 
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

}
}
