/*
 * filename: EPnPPoseEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_ESTIMATOR_EPNP_POSE_ESTIMATOR_H_
#define _MY3D_ESTIMATOR_EPNP_POSE_ESTIMATOR_H_

#include <iostream>

#include "utils/eigen_types.h"
#include "base/polynomial.h"

namespace my3d {
namespace estimator {

/**
 * @brief 3D-2D pose estimator using EPnP algorithm. 
 * Reference: 
 *      F. Moreno-Noguer, V. Lepetit and P. Fua, "Accurate Non-Iterative 𝑂(𝑛) Solution 
 *      to the P𝑛P Problem," 2007 IEEE 11th International Conference on Computer Vision, 
 *      Rio de Janeiro, Brazil, 2007, pp. 1-8, doi: 10.1109/ICCV.2007.4409116.
*/
class EPnPPoseEstimator {
public: 
    const static size_t kMinNumSamples = 4;

    typedef Eigen::Vector3d XDataType;
    typedef EigenVec<Eigen::Vector3d> VecXData;
    typedef Eigen::Vector2d YDataType;
    typedef EigenVec<Eigen::Vector2d> VecYData;
    typedef Eigen::Matrix3x4d ModelType;

    explicit EPnPPoseEstimator(const Eigen::Matrix3d &K = Eigen::Matrix3d::Identity()): 
    K_{K} {

    }

    ~EPnPPoseEstimator() {

    }

    void setK(const Eigen::Matrix3d &K) {
        K_ = K;
    }

    /**
     * @brief Compute poses from given 3D-2D correspondences using AP3P algorithm. 
     * 
     * @param x_data      the 3D coordinates of points in reference frame 
     * @param y_data      the corresponding pixel coordinates of x_data 
     * @param R[out]      the solution of camera frame's orientation w.r.t reference frame 
     * @param t[out]      the solution of camera frame's translation w.r.t reference frame 
    */
    bool computePose(const VecXData &x_data, 
                     const VecYData &y_data, 
                     Eigen::Matrix3x4d &pose);

    /**
     * @brief Estimate the camera pose from given 3D-2D correspondences using AP3P algorithm. 
     * 
     * @param x_data      the 3D coordinates of points in reference frame 
     * @param y_data      the corresponding pixel coordinates of x_data 
     * @retval            the estimated camera pose [R | t]
    */
    void estimate(const VecXData &x_data, 
                  const VecYData &y_data, 
                  EigenVec<ModelType> &models);

    /**
     * @brief Total RMSE.
    */
    double computeRMSE(const VecXData &x_data, 
                             const VecYData &y_data, 
                             const ModelType &model);
    
    /**
     * @brief Euclidean norm of residual.
    */
    double computeError(const XDataType &x_data, 
                        const YDataType &y_data, 
                        const ModelType &model);

    /**
     * @brief Euclidean norms of residuals.
    */
    void computeError(const VecXData &x_data, 
                      const VecYData &y_data, 
                      const ModelType &model, 
                      std::vector<double> &errors);

    /**
     * @brief Reprojection residual
    */
    Eigen::Vector2d computeResidual(const XDataType &x_data, 
                                    const YDataType &y_data, 
                                    const ModelType &model);

private: 
    void selectControlPoints(); 
    bool computeBarycentricCoordinates(); 
    void computeM(); 
    void computeVs(); 
    void computeL6x10(); 
    void computeRho(); 
    void solveForBetasApprox1(Eigen::Vector4d &betas); 
    void solveForBetasApprox2(Eigen::Vector4d &betas); 
    void solveForBetasApprox3(Eigen::Vector4d &betas); 
    void optimizeBetas(Eigen::Vector4d &betas); 
    void computeCcs(const Eigen::Vector4d &betas); 
    void computePcs(); 
    void solveForSign(); 
    void estimatePose(Eigen::Matrix3x4d &pose); 
    
private: 
    Eigen::Matrix3d K_;

    Eigen::Matrix<double, Eigen::Dynamic, 12> M_;
    Eigen::Matrix<double, 6, 10> L6x10_; 
    Eigen::Vector6d rho_; 
    EigenArray<Eigen::Matrix<double, 12, 1>, 4> vs_; 
    EigenArray<Eigen::Vector3d, 4> ccs_; 
    EigenArray<Eigen::Vector3d, 4> cws_;
    EigenVec<Eigen::Vector3d> pcs_; 
    EigenVec<Eigen::Vector3d> pws_; 
    EigenVec<Eigen::Vector4d> bcs_; 
    EigenVec<Eigen::Vector2d> us_; 
};

}
}

#endif // _MY3D_ESTIMATOR_EPNP_POSE_ESTIMATOR_H_