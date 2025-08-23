/*
 * filename: EssentialMatrixEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "estimators/EssentialMatrixEstimator.h"

namespace my3d {
namespace estimator {

EssentialMatrixEightPointEstimator::EssentialMatrixEightPointEstimator() {

}

EssentialMatrixEightPointEstimator::~EssentialMatrixEightPointEstimator() {

}

void
EssentialMatrixEightPointEstimator::estimate(const VecXData &x_data, 
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
    Eigen::Map<Eigen::Matrix3d> E_normt(nullspace_A.data());
    Eigen::Matrix3d E = transform_y.transpose() * E_normt.transpose() * transform_x;

    /* Enforce the singularity constraint. */
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd_E.singularValues();
    singular_values(0) = 0.5 * (singular_values(0) + singular_values(1));
    singular_values(1) = singular_values(0);
    singular_values(2) = 0.0;
    E = svd_E.matrixU() * singular_values.asDiagonal() * svd_E.matrixV().transpose();

    models.clear();
    models.push_back(E);
}

double 
EssentialMatrixEightPointEstimator::computeRMSE(const VecXData &x_data, 
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
EssentialMatrixEightPointEstimator::computeError(const XDataType &x_data, 
                                                 const YDataType &y_data, 
                                                 const ModelType &model) {
    return std::sqrt(computeSquaredSampsonError(x_data, y_data, model));                                                 
}

void 
EssentialMatrixEightPointEstimator::computeError(const VecXData &x_data, 
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

void
EssentialMatrixFivePointEstimator::estimate(const VecXData &x_data, 
                                             const VecYData &y_data, 
                                             EigenVec<ModelType> &models) {
    assert(x_data.size() >= kMinNumSamples && "Data size should be greater than minimum sample number. ");
    assert(x_data.size() == y_data.size());

    const size_t num_pnts = x_data.size();

    // Step 1: Extraction of the nullspace x, y, z, w.

    Eigen::Matrix<double, Eigen::Dynamic, 9> Q(num_pnts, 9);
    for (size_t i = 0; i < num_pnts; ++i) {
        const double x1_0 = x_data[i](0);
        const double x1_1 = x_data[i](1);
        const double x2_0 = y_data[i](0);
        const double x2_1 = y_data[i](1);
        Q(i, 0) = x1_0 * x2_0;
        Q(i, 1) = x1_1 * x2_0;
        Q(i, 2) = x2_0;
        Q(i, 3) = x1_0 * x2_1;
        Q(i, 4) = x1_1 * x2_1;
        Q(i, 5) = x2_1;
        Q(i, 6) = x1_0;
        Q(i, 7) = x1_1;
        Q(i, 8) = 1;
    }

    // Extract the 4 Eigen vectors corresponding to the smallest singular values.
    const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        Q, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 9, 4> E = svd.matrixV().block<9, 4>(0, 5);

    // Step 3: Gauss-Jordan elimination with partial pivoting on A.

    Eigen::Matrix<double, 10, 20> A;
    #include "estimators/essential_matrix_poly.h"
    Eigen::Matrix<double, 10, 10> AA =
        A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

    // Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
    //         matrix B to obtain the tenth degree polynomial.

    Eigen::Matrix<double, 13, 3> B;
    for (size_t i = 0; i < 3; ++i) {
        B(0, i) = 0;
        B(4, i) = 0;
        B(8, i) = 0;
        B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
        B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
        B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
        B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
        B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
        B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
    }

    // Step 5: Extraction of roots from the degree 10 polynomial.
    Eigen::VectorXd coeffs(11);
    #include "estimators/essential_matrix_coeffs.h"

    models.clear(); 

    Eigen::VectorXd roots_real;
    Eigen::VectorXd roots_imag;
    if (!base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag)) {
        return; 
    }

    models.reserve(roots_real.size());

    for (Eigen::VectorXd::Index i = 0; i < roots_imag.size(); ++i) {
        const double kMaxRootImag = 1e-10;
        if (std::abs(roots_imag(i)) > kMaxRootImag) {
        continue;
        }

        const double z1 = roots_real(i);
        const double z2 = z1 * z1;
        const double z3 = z2 * z1;
        const double z4 = z3 * z1;

        Eigen::Matrix3d Bz;
        for (size_t j = 0; j < 3; ++j) {
        Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
        Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
        Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 +
                    B(12, j);
        }

        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
        const Eigen::Vector3d X = svd.matrixV().block<3, 1>(0, 2);

        const double kMaxX3 = 1e-10;
        if (std::abs(X(2)) < kMaxX3) {
        continue;
        }

        Eigen::MatrixXd essential_vec = E.col(0) * (X(0) / X(2)) +
                                        E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
                                        E.col(3);
        essential_vec /= essential_vec.norm();

        const Eigen::Matrix3d essential_matrix =
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
                essential_vec.data());
        models.push_back(essential_matrix);
    }
}

double 
EssentialMatrixFivePointEstimator::computeRMSE(const VecXData &x_data, 
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
EssentialMatrixFivePointEstimator::computeError(const XDataType &x_data, 
                                                 const YDataType &y_data, 
                                                 const ModelType &model) {
    return std::sqrt(computeSquaredSampsonError(x_data, y_data, model));                                                 
}

void 
EssentialMatrixFivePointEstimator::computeError(const VecXData &x_data, 
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

void decomposeEssentialMatrix(const Eigen::Matrix3d &E, 
                              EigenVec<Eigen::Matrix3d> &Rs, 
                              EigenVec<Eigen::Vector3d> &ts) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd_E.matrixU();
    Eigen::Matrix3d Vt = svd_E.matrixV().transpose();
    if (U.determinant() < 0) {
        U *= -1; 
    }
    if (Vt.determinant() < 0) {
        Vt *= -1; 
    }
    
    Eigen::Matrix3d W;
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    const Eigen::Matrix3d R1 = U * W * Vt;
    const Eigen::Matrix3d R2 = U * W.transpose() * Vt;
    const Eigen::Vector3d u3 = U.col(2).normalized();
    Rs = {R1, R2, R1, R2};
    ts = {u3, u3, -u3, -u3};
}

Eigen::Matrix3d essentialMatrixFromPose(const Eigen::Matrix3d &R, 
                                        const Eigen::Vector3d &t) {
    return util::skewSymmetric(t.normalized()) * R; 
}

void poseFromEssentialMatrix(const Eigen::Matrix3d &E, 
                             const EigenVec<Eigen::Vector2d> &points1, 
                             const EigenVec<Eigen::Vector2d> &points2, 
                             Eigen::Matrix3d &R, 
                             Eigen::Vector3d &t, 
                             EigenVec<Eigen::Vector3d> &points_3D) {
    assert(points1.size() == points2.size());

    EigenVec<Eigen::Matrix3d> candidates_R;
    EigenVec<Eigen::Vector3d> candidates_t;
    decomposeEssentialMatrix(E, candidates_R, candidates_t);
    for (size_t i = 0; i < candidates_R.size(); ++i) {
        EigenVec<Eigen::Vector3d> tmp_points_3D;
        base::checkCheirality(candidates_R[i], candidates_t[i], 
                              Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(), 
                              points1, points2, &tmp_points_3D);
        if (tmp_points_3D.size() > points_3D.size()) {
            R = candidates_R[i];
            t = candidates_t[i];
            points_3D = tmp_points_3D;
        }
    }
}

}
}
