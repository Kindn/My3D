/*
 * filename: EPnPPoseEstimator.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "estimators/EPnPPoseEstimator.h"

namespace my3d {
namespace estimator {

bool 
EPnPPoseEstimator::computePose(const VecXData &x_data, 
                               const VecYData &y_data, 
                               Eigen::Matrix3x4d &pose) {
    assert(x_data.size() >= kMinNumSamples);
    assert(y_data.size() >= kMinNumSamples); 
    assert(x_data.size() == y_data.size()); 

    pws_ = x_data; 
    us_ = y_data; 

    selectControlPoints(); 
    if (!computeBarycentricCoordinates()) {
        return false; 
    }

    computeM(); 
    computeVs(); 
    computeL6x10(); 
    computeRho(); 

    EigenArray<Eigen::Matrix3x4d, 3> poses; 
    std::array<double, 3> reproj_errors; 
    Eigen::Vector4d betas; 

    solveForBetasApprox1(betas); 
    optimizeBetas(betas); 
    computeCcs(betas); 
    computePcs(); 
    solveForSign(); 
    estimatePose(poses[0]); 
    reproj_errors[0] = computeRMSE(x_data, y_data, poses[0]); 
    // std::cout << "pose 0: \n[" << poses[0] << "]" << std::endl; 
    // std::cout << "reproj_error 0: " << reproj_errors[0] << std::endl; 

    solveForBetasApprox2(betas); 
    optimizeBetas(betas); 
    computeCcs(betas); 
    computePcs(); 
    solveForSign(); 
    estimatePose(poses[1]); 
    reproj_errors[1] = computeRMSE(x_data, y_data, poses[1]); 
    // std::cout << "pose 1: \n[" << poses[1] << "]" << std::endl; 
    // std::cout << "reproj_error 1: " << reproj_errors[1] << std::endl;

    solveForBetasApprox3(betas); 
    optimizeBetas(betas); 
    computeCcs(betas); 
    computePcs(); 
    solveForSign(); 
    estimatePose(poses[2]); 
    reproj_errors[2] = computeRMSE(x_data, y_data, poses[2]); 
    // std::cout << "pose 2: \n[" << poses[2] << "]" << std::endl; 
    // std::cout << "reproj_error 2: " << reproj_errors[2] << std::endl;

    size_t best_idx = 0; 
    if (reproj_errors[1] < reproj_errors[best_idx]) {
        best_idx = 1; 
    } 
    if (reproj_errors[2] < reproj_errors[best_idx]) {
        best_idx = 2; 
    }

    pose = poses[best_idx]; 

    return true; 
}

void 
EPnPPoseEstimator::estimate(const VecXData &x_data, 
                            const VecYData &y_data, 
                            EigenVec<ModelType> &models) {
    assert(x_data.size() >= kMinNumSamples);
    assert(y_data.size() >= kMinNumSamples); 
    assert(x_data.size() == y_data.size()); 

    models.clear(); 
    ModelType model; 
    if (computePose(x_data, y_data, model)) {
        models.push_back(model); 
    }
}

double 
EPnPPoseEstimator::computeRMSE(const VecXData &x_data, 
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
EPnPPoseEstimator::computeError(const XDataType &x_data, 
                                const YDataType &y_data, 
                                const ModelType &model) {
    return computeResidual(x_data, y_data, model).norm();
}

void 
EPnPPoseEstimator::computeError(const VecXData &x_data, 
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
EPnPPoseEstimator::computeResidual(const XDataType &x_data, 
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

void  
EPnPPoseEstimator::selectControlPoints() {
    const size_t num_points = pws_.size(); 

    cws_[0].setZero(); 
    for (size_t i = 0; i < num_points; ++i) {
        cws_[0] += pws_[i]; 
    }
    cws_[0] /= static_cast<double>(num_points); 

    Eigen::Matrix<double, Eigen::Dynamic, 3> A(num_points, 3);  
    for (size_t i = 0; i < num_points; ++i) {
        A.row(i) = (pws_[i] - cws_[0]).transpose(); 
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_AtA(A.transpose() * A, Eigen::ComputeFullU | 
                                                                 Eigen::ComputeFullV); 
    const Eigen::Vector3d evalues = svd_AtA.singularValues(); 
    const Eigen::Matrix3d U = svd_AtA.matrixU(); 
    for (size_t i = 1; i < 4; ++i) {
        cws_[i] = cws_[0] + 
        std::sqrt(evalues(i - 1) / static_cast<double>(num_points)) * U.col(i - 1); 
    }
} 

bool 
EPnPPoseEstimator::computeBarycentricCoordinates() {
    const size_t num_points = pws_.size(); 

    Eigen::Matrix3d C; 
    for (size_t i = 0; i < 3; ++i) {
        C.col(i) = cws_[i + 1] - cws_[0]; 
    }
    if (C.colPivHouseholderQr().rank() < 3) {
        return false; 
    }

    const Eigen::Matrix3d C_inv = C.inverse(); 
    bcs_.resize(num_points); 
    for (size_t i = 0; i < num_points; ++i) {
        bcs_[i].tail(3) = C_inv * (pws_[i] - cws_[0]); 
        bcs_[i](0) = 1.0 - bcs_[i].tail(3).sum(); 
    }

    return true; 
} 

void 
EPnPPoseEstimator::computeM() {
    const size_t num_points = pws_.size(); 
    const double fx = K_(0, 0); 
    const double fy = K_(1, 1); 
    const double cx = K_(0, 2); 
    const double cy = K_(1, 2); 

    M_ = Eigen::Matrix<double, Eigen::Dynamic, 12>::Zero(2 * num_points, 12); 
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            M_(2 * i, 3 * j) = bcs_[i](j) * fx; 
            M_(2 * i, 3 * j + 2) = bcs_[i](j) * (cx - us_[i].x()); 
            M_(2 * i + 1, 3 * j + 1) = bcs_[i](j) * fy; 
            M_(2 * i + 1, 3 * j + 2) = bcs_[i](j) * (cy - us_[i].y()); 
        }
    }
}

void 
EPnPPoseEstimator::computeVs() {
    const Eigen::Matrix<double, 12, 12> MtM = M_.transpose() * M_; 
    Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd_MtM(MtM, 
        Eigen::ComputeFullU | Eigen::ComputeFullV); 
    const Eigen::Matrix<double, 12, 12> U = svd_MtM.matrixU(); 
    for (size_t i = 0; i < 4; ++i) {
        vs_[i] = U.col(8 + i); 
    }
} 

void 
EPnPPoseEstimator::computeL6x10() {
    size_t row_idx = 0; 
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            const Eigen::Vector3d s1 = vs_[0].segment(i * 3, 3) - vs_[0].segment(j * 3, 3); 
            const Eigen::Vector3d s2 = vs_[1].segment(i * 3, 3) - vs_[1].segment(j * 3, 3); 
            const Eigen::Vector3d s3 = vs_[2].segment(i * 3, 3) - vs_[2].segment(j * 3, 3); 
            const Eigen::Vector3d s4 = vs_[3].segment(i * 3, 3) - vs_[3].segment(j * 3, 3); 

            L6x10_(row_idx, 0) = s1.dot(s1); 
            L6x10_(row_idx, 1) = 2.0 * s1.dot(s2); 
            L6x10_(row_idx, 2) = 2.0 * s1.dot(s3); 
            L6x10_(row_idx, 3) = 2.0 * s1.dot(s4); 
            L6x10_(row_idx, 4) = s2.dot(s2); 
            L6x10_(row_idx, 5) = 2.0 * s2.dot(s3); 
            L6x10_(row_idx, 6) = 2.0 * s2.dot(s4); 
            L6x10_(row_idx, 7) = s3.dot(s3); 
            L6x10_(row_idx, 8) = 2.0 * s3.dot(s4); 
            L6x10_(row_idx, 9) = s4.dot(s4); 

            ++row_idx;
        }   
    }
}

void 
EPnPPoseEstimator::computeRho() {
    size_t row_idx = 0; 
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            rho_(row_idx) = (cws_[i] - cws_[j]).squaredNorm(); 

            ++row_idx;
        }   
    }
} 

// [B11 B12 B13 B14 B22 B23 B24 B33 B34 B44] 
// [B11 B12 B13 B14]
void 
EPnPPoseEstimator::solveForBetasApprox1(Eigen::Vector4d &betas) {
    Eigen::Matrix<double, 6, 4> L6x4; 
    for (int i = 0; i < 6; ++i) {
        L6x4(i, 0) = L6x10_(i, 0); 
        L6x4(i, 1) = L6x10_(i, 1); 
        L6x4(i, 2) = L6x10_(i, 2); 
        L6x4(i, 3) = L6x10_(i, 3); 
    } 

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd_L6x4(L6x4, 
        Eigen::ComputeFullU | Eigen::ComputeFullV); 
    Eigen::Vector6d tmp_rho = rho_; 
    const Eigen::Vector4d b = svd_L6x4.solve(tmp_rho); 

    if (b(0) < 0) {
        betas(0) = std::sqrt(-b(0)); 
        betas(1) = -b(1) / betas(0); 
        betas(2) = -b(2) / betas(0); 
        betas(3) = -b(3) / betas(0); 
    } else {
        betas(0) = std::sqrt(b(0)); 
        betas(1) = b(1) / betas(0); 
        betas(2) = b(2) / betas(0); 
        betas(3) = b(3) / betas(0); 
    }
}

// [B11 B12 B13 B14 B22 B23 B24 B33 B34 B44] 
// [B11 B12         B22]
void 
EPnPPoseEstimator::solveForBetasApprox2(Eigen::Vector4d &betas) {
    Eigen::Matrix<double, 6, 3> L6x3; 
    for (int i = 0; i < 6; ++i) {
        L6x3(i, 0) = L6x10_(i, 0); 
        L6x3(i, 1) = L6x10_(i, 1); 
        L6x3(i, 2) = L6x10_(i, 4); 
    } 

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3>> svd_L6x3(L6x3, 
        Eigen::ComputeFullU | Eigen::ComputeFullV); 
    Eigen::Vector6d tmp_rho = rho_; 
    const Eigen::Vector3d b = svd_L6x3.solve(tmp_rho); 

    betas.setZero(); 
    if (b(0) < 0) {
        betas(0) = std::sqrt(-b(0)); 
        betas(1) = (b(2) < 0) ? std::sqrt(-b(2)) : 0.0; 
    } else {
        betas(0) = std::sqrt(b(0)); 
        betas(1) = (b(2) > 0) ? std::sqrt(b(2)) : 0.0; 
    }

    if (b(1) < 0) {
        betas(0) = -betas(0); 
    }
}

// [B11 B12 B13 B14 B22 B23 B24 B33 B34 B44] 
// [B11 B12 B13     B22 B23]
void 
EPnPPoseEstimator::solveForBetasApprox3(Eigen::Vector4d &betas) {
    Eigen::Matrix<double, 6, 5> L6x5; 
    for (int i = 0; i < 6; ++i) {
        L6x5(i, 0) = L6x10_(i, 0); 
        L6x5(i, 1) = L6x10_(i, 1); 
        L6x5(i, 2) = L6x10_(i, 2); 
        L6x5(i, 3) = L6x10_(i, 4); 
        L6x5(i, 4) = L6x10_(i, 5); 
    } 

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd_L6x5(L6x5, 
        Eigen::ComputeFullU | Eigen::ComputeFullV); 
    Eigen::Vector6d tmp_rho = rho_; 
    const Eigen::Vector5d b = svd_L6x5.solve(tmp_rho); 

    betas.setZero(); 
    if (b(0) < 0) {
        betas(0) = std::sqrt(-b(0)); 
        betas(1) = (b(2) < 0) ? std::sqrt(-b(2)) : 0.0; 
    } else {
        betas(0) = std::sqrt(b(0)); 
        betas(1) = (b(2) > 0) ? std::sqrt(b(2)) : 0.0; 
    }

    if (b(1) < 0) {
        betas(0) = -betas(0); 
    }

    betas(2) = b(2) / betas(0); 
}

void 
EPnPPoseEstimator::optimizeBetas(Eigen::Vector4d &betas) {
    const int kNumIterations = 10; 
    double last_error = INFINITY; 
    for (int iter = 0; iter < kNumIterations; ++iter) {
        // Compute residual
        Eigen::Matrix<double, 10, 1> b; 
        b << betas(0) * betas(0), betas(0) * betas(1), betas(0) * betas(2), betas(0) * betas(3), 
            betas(1) * betas(1), betas(1) * betas(2), betas(1) * betas(3), 
            betas(2) * betas(2), betas(2) * betas(3), 
            betas(3) * betas(3); 
        const Eigen::Vector6d residual = L6x10_ * b - rho_; 
        const double error = residual.norm(); 
        // std::cout << error << std::endl;
        if (error > last_error) {
            std::cout << "[EPnPPoseEstimator] Invalid Gauss-Newton step. " << std::endl; 
            break; 
        }
        last_error = error; 

        // Compute Jacobian 
        Eigen::Matrix<double, 6, 4> jacobian; 
        for (int i = 0; i < 6; ++i) {
            jacobian(i, 0) = betas(0) * 2.0 * L6x10_(i, 0) + betas(1) * L6x10_(i, 1) + 
                             betas(2) * L6x10_(i, 2) + betas(3) * L6x10_(i, 3); 
            jacobian(i, 1) = betas(0) * L6x10_(i, 1) + betas(1) * 2.0 * L6x10_(i, 4) + 
                             betas(2) * L6x10_(i, 5) + betas(3) * L6x10_(i, 6); 
            jacobian(i, 2) = betas(0) * L6x10_(i, 2) + betas(1) * L6x10_(i, 5) + 
                             betas(2) * 2.0 * L6x10_(i, 7) + betas(3) * L6x10_(i, 8); 
            jacobian(i, 3) = betas(0) * L6x10_(i, 3) + betas(1) * L6x10_(i, 6) + 
                             betas(2) * L6x10_(i, 8) + betas(3) * 2.0 * L6x10_(i, 9); 
        }

        // Update
        Eigen::Matrix4d A = jacobian.transpose() * jacobian; 
        const Eigen::Vector4d c = -jacobian.transpose() * residual; 
        const Eigen::Vector4d delta = A.ldlt().solve(c); 
        // const Eigen::Vector4d delta = jacobian.colPivHouseholderQr().solve(-residual); 
        betas += delta; 
    }
}

void 
EPnPPoseEstimator::computeCcs(const Eigen::Vector4d &betas) {
    for (int i = 0; i < 4; ++i) {
        ccs_[i] = betas(0) * vs_[0].segment(i * 3, 3) + 
                  betas(1) * vs_[1].segment(i * 3, 3) + 
                  betas(2) * vs_[2].segment(i * 3, 3) + 
                  betas(3) * vs_[3].segment(i * 3, 3); 
    }
}

void 
EPnPPoseEstimator::computePcs() {
    const size_t num_points = pws_.size(); 
    pcs_.resize(num_points); 
    for (size_t i = 0; i < num_points; ++i) {
        pcs_[i] = bcs_[i](0) * ccs_[0] + 
                  bcs_[i](1) * ccs_[1] + 
                  bcs_[i](2) * ccs_[2] + 
                  bcs_[i](3) * ccs_[3];     
    } 
} 

void 
EPnPPoseEstimator::solveForSign() {
    if (pcs_.front().z() < 0.0) {
        for (int i = 0; i < 4; ++i) {
            ccs_[i] = -ccs_[i]; 
        }
        for (size_t i = 0; i < pws_.size(); ++i) {
            pcs_[i] = -pcs_[i]; 
        }
    }
}

void 
EPnPPoseEstimator::estimatePose(Eigen::Matrix3x4d &pose) {
    const size_t num_points = pws_.size(); 

    Eigen::Vector3d center_pcs = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d center_pws = Eigen::Vector3d::Zero(); 
    for (size_t i = 0; i < num_points; ++i) {
        center_pcs += pcs_[i]; 
        center_pws += pws_[i]; 
    }
    center_pcs /= static_cast<double>(num_points); 
    center_pws /= static_cast<double>(num_points); 

    Eigen::Matrix<double, Eigen::Dynamic, 3> Pc(num_points, 3), Pw(num_points, 3); 
    for (size_t i = 0; i < num_points; ++i) {
        Pc.row(i) = (pcs_[i] - center_pcs).transpose(); 
        Pw.row(i) = (pws_[i] - center_pws).transpose(); 
    } 
    
    const Eigen::Matrix3d W = Pc.transpose() * Pw; 
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_W(W, Eigen::ComputeFullU | Eigen::ComputeFullV); 
    const Eigen::Matrix3d U = svd_W.matrixU(); 
    Eigen::Matrix3d V = svd_W.matrixV(); 

    Eigen::Matrix3d R = U * V.transpose(); 
    if (R.determinant() < 0) {
        V.col(2) *= -1.0; 
        R = U * V.transpose(); 
    }
    const Eigen::Vector3d t = center_pcs - R * center_pws; 

    pose.leftCols<3>() = R.transpose(); 
    pose.rightCols<1>() = -R.transpose() * t; 
}

}
}