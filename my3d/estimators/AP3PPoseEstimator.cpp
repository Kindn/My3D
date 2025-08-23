/*
 * filename: AP3PPoseEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "estimators/AP3PPoseEstimator.h"

namespace my3d {
namespace estimator {

int 
AP3PPoseEstimator::computePose(const VecXData &x_data, 
                               const VecYData &y_data, 
                               EigenVec<Eigen::Matrix3d> &Rs, 
                               EigenVec<Eigen::Vector3d> &ts) {
    // const double epsilon = std::numeric_limits<double>::epsilon();
    Rs.clear();
    ts.clear();

    const Eigen::Matrix3d inv_K = K_.inverse();
    const Eigen::Vector3d bc1 = (inv_K * y_data[0].homogeneous()).normalized();
    const Eigen::Vector3d bc2 = (inv_K * y_data[1].homogeneous()).normalized();
    const Eigen::Vector3d bc3 = (inv_K * y_data[2].homogeneous()).normalized();
    const Eigen::Vector3d pG1 = x_data[0];
    const Eigen::Vector3d pG2 = x_data[1];
    const Eigen::Vector3d pG3 = x_data[2];

    // assert((pG1 - pG2).norm() >= epsilon);
    // assert((bc1.cross(bc2)).norm() >= epsilon);
    if ((pG1 - pG2).norm() < epsilon_ || (bc1.cross(bc2)).norm() < epsilon_ || 
        (pG1 - pG3).norm() < epsilon_ || (bc1.cross(bc3)).norm() < epsilon_ || 
        (pG2 - pG3).norm() < epsilon_ || (bc2.cross(bc3)).norm() < epsilon_) {
        // std::cout << "[" << __PRETTY_FUNCTION__ << "] " << "Warning: Singularity detected. " << std::endl;
        return 0;
    }
    const Eigen::Vector3d k1 = (pG1 - pG2).normalized();
    const Eigen::Vector3d k3 = (bc1.cross(bc2)).normalized();
    // const Eigen::Vector3d k2 = k1.cross(k3);

    const Eigen::Vector3d u1 = pG1 - pG3;
    const Eigen::Vector3d u2 = pG2 - pG3;
    const Eigen::Vector3d v1 = bc1.cross(bc3);
    const Eigen::Vector3d v2 = bc2.cross(bc3);

    // double theta2 = std::acos(k1.dot(k3)) - M_PI_2;
    // const Eigen::Vector3d k3p = k2.cross(k1);
    const double delta = (u1.cross(k1)).norm();
    const Eigen::Vector3d k3pp = (u1.cross(k1)).normalized();

    const double f11 = delta * (k3.dot(bc3));
    const double f21 = delta * (bc1.dot(bc2)) * (k3.dot(bc3));
    const double f22 = delta * (k3.dot(bc3)) * ((bc1.cross(bc2)).norm());
    const double f13 = delta * (v1.dot(k3));
    const double f23 = delta * (v2.dot(k3));
    const double f24 = (u2.dot(k1)) * (k3.dot(bc3)) * ((bc1.cross(bc2)).norm());
    const double f15 = -(u1.dot(k1)) * (k3.dot(bc3));
    const double f25 = -(u2.dot(k1)) * (bc1.dot(bc2)) * (k3.dot(bc3));
    const double g1 = f13 * f22;
    const double g2 = f13 * f25 - f15 * f23;
    const double g3 = f11 * f23 - f13 * f21;
    const double g4 = - f13 * f24;
    const double g5 = f11 * f22;
    const double g6 = f11 * f25 - f15 * f21;
    const double g7 = -f15 * f24;

    const double a0 = g7 * g7 - g2 * g2 - g4 * g4;
    const double a1 = 2.0 * (g6 * g7 - g1 * g2 - g3 * g4);
    const double a2 = g6 * g6 + 2.0 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3;
    const double a3 = 2.0 * (g5 * g6 + g1 * g2 + g3 * g4);
    const double a4 = g5 * g5 + g1 * g1 + g3 * g3;
    Eigen::VectorXd coeffs(5);
    coeffs << a0, a1, a2, a3, a4;
    // std::cout << "coeffs: " << coeffs.transpose() << std::endl;
    Eigen::VectorXd roots_real, roots_imag;
    bool success_solving = base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag);
    if (!success_solving) {
        std::cout << "[" << __PRETTY_FUNCTION__ << "] " << "Warning: Solving polynomial failed. " << std::endl;
        return 0;
    } else {
        bool success_refinement = base::refineRoots(coeffs, roots_real, 2u);
        if (!success_refinement) {
            std::cout << "[" << __PRETTY_FUNCTION__ << "] " << "Warning: Root refinement failed. " << std::endl;
            return 0;
        }
    }

    const int sign = (k3.dot(bc3) >= 0) ? 1 : -1;
    for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
        if (std::abs(roots_real(i)) > 1.0 - std::numeric_limits<double>::epsilon()) { // || std::abs(roots_imag(i)) > epsilon) {
            continue;
        }

        const double cos_theta1p = roots_real(i);
        const double cos2_theta1p = cos_theta1p * cos_theta1p;
        const double sin_theta1p = sign * std::sqrt(1.0 - cos2_theta1p);
        if (std::isnan(cos_theta1p) || std::isnan(sin_theta1p)) {
            std::cout << "[" << __PRETTY_FUNCTION__ << "] " << "Warning: NaN occurs. " << std::endl;
            return 0;
        }
        // Compute sin(theta_1') and cos(theta_1')
        const double factor = sin_theta1p / (g5 * cos2_theta1p + g6 * cos_theta1p + g7);
        const double cos_theta3p = factor * (g1 * cos_theta1p + g2);
        const double sin_theta3p = factor * (g3 * cos_theta1p + g4);

        // Compute rotation
        Eigen::Matrix3d R1, R2T;
        R1.col(0) = k1;
        R1.col(1) = k3pp;
        R1.col(2) = k1.cross(k3pp);
        R2T.col(0) = bc1;
        R2T.col(1) = k3;
        R2T.col(2) = bc1.cross(k3);
        Eigen::Matrix3d R_x_theta1p, R_y_theta3p;
        //* Using left-hand rule
        R_x_theta1p << 1, 0, 0, 
                       0, cos_theta1p, sin_theta1p, 
                       0, -sin_theta1p, cos_theta1p;
        R_y_theta3p << cos_theta3p, 0, -sin_theta3p, 
                       0, 1, 0, 
                       sin_theta3p, 0, cos_theta3p;
        if (std::abs(R_x_theta1p.determinant() - 1.0) > 1e-6 || 
            std::abs(R_y_theta3p.determinant() - 1.0) > 1e-6) {
            return 0; 
        }

        Eigen::Matrix3d R13;
        Eigen::Matrix3d R = R1 * R_x_theta1p * R_y_theta3p * R2T.transpose();

        // Compute translation
        Eigen::Vector3d t = pG3 - (delta * sin_theta1p / k3.dot(bc3)) * R * bc3;

        Rs.push_back(R);
        ts.push_back(t);
    }

    return static_cast<int>(Rs.size());
}

void
AP3PPoseEstimator::estimate(const VecXData &x_data, 
                            const VecYData &y_data, 
                            EigenVec<ModelType> &models) {
    
    assert(x_data.size() >= kMinNumSamples);
    assert(y_data.size() >= kMinNumSamples);

    EigenVec<Eigen::Matrix3d> Rs;
    EigenVec<Eigen::Vector3d> ts;
    models.clear();
    int num_solutions = computePose(x_data, y_data, Rs, ts);
    if (num_solutions <= 0) {
        return;
    }
    for (int i = 0; i < num_solutions; ++i) {
        ModelType pose;
        pose.block<3, 3>(0, 0) = Rs[i];
        pose.col(3) = ts[i];
        models.push_back(pose);
    }
}

double 
AP3PPoseEstimator::computeRMSE(const VecXData &x_data, 
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
AP3PPoseEstimator::computeError(const XDataType &x_data, 
                                const YDataType &y_data, 
                                const ModelType &model) {
    return computeResidual(x_data, y_data, model).norm();
}

void 
AP3PPoseEstimator::computeError(const VecXData &x_data, 
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
AP3PPoseEstimator::computeResidual(const XDataType &x_data, 
                                   const YDataType &y_data, 
                                   const ModelType &model) {
    const Eigen::Matrix3d R = model.block<3, 3>(0, 0);
    const Eigen::Vector3d t = model.col(3);
    const Eigen::Vector3d pc = R.transpose() * (x_data - t);
    if (std::abs(pc.z()) < std::numeric_limits<double>::epsilon()) {
        return Eigen::Vector2d(std::numeric_limits<double>::infinity(), 0.0); 
    }

    const Eigen::Vector3d pc_norm = pc / pc.z();
    const Eigen::Vector2d ppix = (K_ * pc_norm).hnormalized();
    
    return ppix - y_data;
}

}
}