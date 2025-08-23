/*
 * filename: HomographyEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "estimators/HomographyEstimator.h"

namespace my3d {
namespace estimator{
HomographyEstimator::HomographyEstimator() {
}

HomographyEstimator::~HomographyEstimator() {
}

void
HomographyEstimator::estimate(const VecXData &x_data, 
                              const VecYData &y_data, 
                              EigenVec<ModelType> &models) {
    assert(!x_data.empty());
    assert(!y_data.empty());
    assert(x_data.size() == y_data.size());      
    assert(x_data.size() >= kMinNumSamples && "More than 4 points are needed to estimate homography. ");

    Eigen::Vector2d t_x, t_y;
    double scale_x, scale_y;
    VecXData x_data_normalized;
    VecYData y_data_normalized;
    base::normalize2DPointSet(x_data, x_data_normalized, t_x, scale_x);
    base::normalize2DPointSet(y_data, y_data_normalized, t_y, scale_y);
    Eigen::Matrix3d T_x, T_y;
    T_x.setIdentity(); T_x.block<2, 1>(0, 2) = t_x;
    T_y.setIdentity(); T_y.block<2, 1>(0, 2) = t_y;

    /* Build and solve the linear system */
    // Eigen::MatrixXd L = Eigen::MatrixXd::Zero(2 * x_data_normalized.size(), 8);
    // Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * x_data_normalized.size());
    // for (size_t i = 0; i < x_data_normalized.size(); ++i) {
    //     Eigen::Vector3d x(x_data_normalized[i].x(), x_data_normalized[i].y(), 1.0);
    //     Eigen::Vector3d y(y_data_normalized[i].x(), y_data_normalized[i].y(), 1.0);
    //     L.block<1, 3>(2 * i, 0) = x.transpose();
    //     L.block<1, 2>(2 * i, 6) = -y.x() * x.transpose().head<2>();
    //     L.block<1, 3>(2 * i + 1, 3) = x.transpose();
    //     L.block<1, 2>(2 * i + 1, 6) = -y.y() * x.transpose().head<2>();

    //     b.segment<2>(2 * i) = y.head<2>();
    // }
    // Eigen::MatrixXd h = (L.transpose() * L).ldlt().solve(L.transpose() * b);
    Eigen::Matrix<double, Eigen::Dynamic, 9> A = Eigen::Matrix<double, Eigen::Dynamic, 9>::Zero(2 * x_data_normalized.size(), 9); 
    for (size_t i = 0; i < x_data_normalized.size(); ++i) {
        Eigen::Vector3d x(x_data_normalized[i].x(), x_data_normalized[i].y(), 1.0);
        Eigen::Vector3d y(y_data_normalized[i].x(), y_data_normalized[i].y(), 1.0);
        A.block<1, 3>(2 * i, 0) = x.transpose(); 
        A.block<1, 3>(2 * i, 6) = -y.x() * x.transpose(); 
        A.block<1, 3>(2 * i + 1, 3) = x.transpose(); 
        A.block<1, 3>(2 * i + 1, 6) = -y.y() * x.transpose(); 
    }
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd_A(A, Eigen::ComputeFullV); 
    const Eigen::VectorXd h = svd_A.matrixV().col(8); 

    /* Get the original H */
    Eigen::Matrix3d H_norm, H;
    H_norm << h(0), h(1), h(2), 
              h(3), h(4), h(5), 
              h(6), h(7), h(8); 
    Eigen::Matrix3d scale_mat_x = Eigen::Matrix3d::Identity(), 
                    scale_mat_y = Eigen::Matrix3d::Identity();
    scale_mat_x.block<2, 2>(0, 0) *= scale_x;
    scale_mat_y.block<2, 2>(0, 0) *= scale_y;
    H = T_y.inverse() * scale_mat_y.inverse() * H_norm * scale_mat_x * T_x;
    H /= H(2, 2);

    models.clear();
    if (std::isnan(H(0, 0)) || std::isinf(H(0, 0))) {
        return; 
    }

    models.push_back(H);
}

double 
HomographyEstimator::computeRMSE(const VecXData &x_data, 
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
HomographyEstimator::computeError(const XDataType &x_data, 
                                  const YDataType &y_data, 
                                  const ModelType &model) {
    return computeResidual(x_data, y_data, model).norm();
}

void 
HomographyEstimator::computeError(const VecXData &x_data, 
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

Eigen::Vector2d
HomographyEstimator::computeResidual(const XDataType &x_data, 
                                     const YDataType &y_data, 
                                     const ModelType &model) {
    return y_data - (model * x_data.homogeneous()).hnormalized();                                    
}

double computeOppositeOfMinor(const Eigen::Matrix3d& matrix, 
                              const size_t row,
                              const size_t col) {
  const size_t col1 = col == 0 ? 1 : 0;
  const size_t col2 = col == 2 ? 1 : 2;
  const size_t row1 = row == 0 ? 1 : 0;
  const size_t row2 = row == 2 ? 1 : 2;
  return (matrix(row1, col2) * matrix(row2, col1) -
          matrix(row1, col1) * matrix(row2, col2));
}

void decomposeHomography(const Eigen::Matrix3d &H0, 
                         const Eigen::Matrix3d &K1, 
                         const Eigen::Matrix3d &K2, 
                         EigenVec<Eigen::Matrix3d> &R, 
                         EigenVec<Eigen::Vector3d> &t, 
                         EigenVec<Eigen::Vector3d> &n) {
    /* Remove intrinsics and scale from homography */
    Eigen::Matrix3d H = K2.inverse() * H0 * K1;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_H(H);
    H /= svd_H.singularValues()(1);

    /* Ensure the determinant of H is positive */
    if (H.determinant() < 0) {
        H *= -1.0;
    }
    const Eigen::Matrix3d S = H.transpose() * H - Eigen::Matrix3d::Identity();

    /* If pure rotation occurs. */
    const double kMinInfinityNorm = 1e-3;
    if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
        R = {H};
        t = {Eigen::Vector3d::Zero()};
        n = {Eigen::Vector3d::Zero()};
        return;
    }

    /* Extract entries of S */
    const double s11 = S(0, 0);
    const double s22 = S(1, 1);
    const double s33 = S(2, 2);
    const double s12 = S(0, 1);
    const double s13 = S(0, 2);
    const double s23 = S(1, 2);

    /* Compute opposite minor of S(i, i) */
    const double Ms11 = computeOppositeOfMinor(S, 0, 0);
    const double Ms22 = computeOppositeOfMinor(S, 1, 1);
    const double Ms33 = computeOppositeOfMinor(S, 2, 2);
    const double Ms12 = computeOppositeOfMinor(S, 0, 1);
    const double Ms13 = computeOppositeOfMinor(S, 0, 2);
    const double Ms23 = computeOppositeOfMinor(S, 1, 2);

    /* Compute the normal vector and the translation in frame 1 (reference frame) */
    Eigen::Vector3d na, nb;
    Eigen::Vector3d ta, tb;
    const double max_abs_sii = std::max(std::abs(s11), std::max(std::abs(s22), std::abs(s33)));
    const double nu = 2.0 * std::sqrt(1.0 + S.trace() - Ms11 - Ms22 - Ms33);
    const double rho = std::sqrt(2.0 + S.trace() + nu);
    const double norm_t = std::sqrt(2.0 + S.trace() - nu);

    if (max_abs_sii == std::abs(s11)) {
        const double eps_23 = Ms23 >= 0 ? 1 : -1;
        const double eps_s11 = s11 >= 0 ? 1 : -1;

        na(0) = s11;
        na(1) = s12 + std::sqrt(Ms33);
        na(2) = s13 + eps_23 * std::sqrt(Ms22);
        nb(0) = s11;
        nb(1) = s12 - std::sqrt(Ms33);
        nb(2) = s13 - eps_23 * std::sqrt(Ms22);

        na.normalize();
        nb.normalize();

        ta = norm_t * (eps_s11 *rho * nb - norm_t * na) / 2.0;
        tb = norm_t * (eps_s11 *rho * na - norm_t * nb) / 2.0;
    } else if (max_abs_sii == std::abs(s22)) {
        const double eps_13 = Ms13 >= 0 ? 1 : -1;
        const double eps_s22 = s11 >= 0 ? 1 : -1;

        na(0) = s12 + std::sqrt(Ms33);
        na(1) = s22;
        na(2) = s23 - eps_13 * std::sqrt(Ms11);
        nb(0) = s12 - std::sqrt(Ms33);
        nb(1) = s22;
        nb(2) = s23 + eps_13 * std::sqrt(Ms11);

        na.normalize();
        nb.normalize();

        ta = norm_t * (eps_s22 *rho * nb - norm_t * na) / 2.0;
        tb = norm_t * (eps_s22 *rho * na - norm_t * nb) / 2.0;
    } else if (max_abs_sii == std::abs(s33)) {
        const double eps_12 = Ms12 >= 0 ? 1 : -1;
        const double eps_s33 = s11 >= 0 ? 1 : -1;

        na(0) = s13 + eps_12 * std::sqrt(Ms22);
        na(1) = s23 + std::sqrt(Ms11);
        na(2) = s33;
        nb(0) = s13 - eps_12 * std::sqrt(Ms22);
        nb(1) = s23 - std::sqrt(Ms11);
        nb(2) = s33;

        na.normalize();
        nb.normalize();

        ta = norm_t * (eps_s33 * rho * nb - norm_t * na) / 2.0;
        tb = norm_t * (eps_s33 * rho * na - norm_t * nb) / 2.0;
    }

    /* Recover the rotation and the translation */
    Eigen::Matrix3d Ra = H * (Eigen::Matrix3d::Identity() - 2.0 / nu * ta * na.transpose());
    Eigen::Matrix3d Rb = H * (Eigen::Matrix3d::Identity() - 2.0 / nu * tb * nb.transpose());
    ta = Ra * ta;
    tb = Rb * tb;

    R = {Ra, Rb, Ra, Rb};
    t = {ta, tb, -ta, -tb};
    n = {-na, -nb, na, nb};
}

Eigen::Matrix3d homographyFromPose(const Eigen::Matrix3d &K1, 
                                   const Eigen::Matrix3d &K2, 
                                   const Eigen::Matrix3d &R, 
                                   const Eigen::Vector3d &t, 
                                   const Eigen::Vector3d &n, 
                                   double d) {
    assert(d > 0); 

    return K2 * (R - t * n.normalized().transpose() / d);
}

void poseFromHomography(const Eigen::Matrix3d &H, 
                        const Eigen::Matrix3d &K1, 
                        const Eigen::Matrix3d &K2, 
                        const EigenVec<Eigen::Vector2d> &points1, 
                        const EigenVec<Eigen::Vector2d> &points2, 
                        Eigen::Matrix3d &R, 
                        Eigen::Vector3d &t, 
                        Eigen::Vector3d &n,
                        EigenVec<Eigen::Vector3d> &points_3D) {
    assert(points1.size() == points2.size());

    EigenVec<Eigen::Matrix3d> candidates_R;
    EigenVec<Eigen::Vector3d> candidates_t, candidates_n;
    decomposeHomography(H, K1, K2, candidates_R, candidates_t, candidates_n);
    EigenVec<Eigen::Vector3d> tmp_points_3D;
    for (size_t i = 0; i < candidates_R.size(); ++i) {
        base::checkCheirality(candidates_R[i], candidates_t[i], K1, K2, 
                              points1, points2, &tmp_points_3D);
        if (tmp_points_3D.size() > points_3D.size()) {
            R = candidates_R[i];
            t = candidates_t[i];
            n = candidates_n[i];
            points_3D = tmp_points_3D;
        }
    }
}

}
}
