/*
 * filename: EssentialMatrixEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_ESTIMATOR_ESSENTIAL_MATRIX_ESTIMATOR_H_
#define _MY3D_ESTIMATOR_ESSENTIAL_MATRIX_ESTIMATOR_H_

#include <iostream>

#include "base/pose.h"
#include "base/points.h"
#include "base/polynomial.h"
#include "estimators/estimator_utils.h"

namespace my3d {
namespace estimator {

class EssentialMatrixEightPointEstimator {
public: 
    const static size_t kMinNumSamples = 8;

    typedef Eigen::Vector2d XDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> XAllocType;
    typedef std::vector<XDataType, XAllocType> VecXData;
    typedef Eigen::Vector2d YDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> YAllocType;
    typedef std::vector<YDataType, YAllocType> VecYData;
    typedef Eigen::Matrix3d ModelType;

    explicit EssentialMatrixEightPointEstimator();
    ~EssentialMatrixEightPointEstimator();

public: 

    /**
     * @brief Estimate the essential matrix E which satisfies y' E x == 0
     * 
     * @param x_data normalized camera coordinates of points in camera 1
     * @param y_data normalized camera coordinates of points in camera 2
    */
    static void estimate(const VecXData &x_data, 
                              const VecYData &y_data, 
                              EigenVec<ModelType> &models);

    /**
     * @brief Total RMSE.
    */
    static double computeRMSE(const VecXData &x_data, 
                                    const VecYData &y_data, 
                                    const ModelType &model);
    
    /**
     * @brief Euclidean norm of residual.
    */
    static double computeError(const XDataType &x_data, 
                               const YDataType &y_data, 
                               const ModelType &model);
    
    /**
     * @brief Euclidean norm of residual.
    */
    static void computeError(const VecXData &x_data, 
                             const VecYData &y_data, 
                             const ModelType &model, 
                             std::vector<double> &errors);


}; 

class EssentialMatrixFivePointEstimator {
public: 
    const static size_t kMinNumSamples = 5;

    typedef Eigen::Vector2d XDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> XAllocType;
    typedef std::vector<XDataType, XAllocType> VecXData;
    typedef Eigen::Vector2d YDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> YAllocType;
    typedef std::vector<YDataType, YAllocType> VecYData;
    typedef Eigen::Matrix3d ModelType;

    explicit EssentialMatrixFivePointEstimator() {}
    ~EssentialMatrixFivePointEstimator() {}

public: 

    /**
     * @brief Estimate the essential matrix E which satisfies y' E x == 0
     * 
     * @param x_data normalized camera coordinates of points in camera 1
     * @param y_data normalized camera coordinates of points in camera 2
    */
    static void estimate(const VecXData &x_data, 
                              const VecYData &y_data, 
                              EigenVec<ModelType> &models);

    /**
     * @brief Total RMSE.
    */
    static double computeRMSE(const VecXData &x_data, 
                                    const VecYData &y_data, 
                                    const ModelType &model);
    
    /**
     * @brief Euclidean norm of residual.
    */
    static double computeError(const XDataType &x_data, 
                               const YDataType &y_data, 
                               const ModelType &model);
    
    /**
     * @brief Euclidean norm of residual.
    */
    static void computeError(const VecXData &x_data, 
                             const VecYData &y_data, 
                             const ModelType &model, 
                             std::vector<double> &errors);


}; 

void decomposeEssentialMatrix(const Eigen::Matrix3d &E, 
                              EigenVec<Eigen::Matrix3d> &Rs, 
                              EigenVec<Eigen::Vector3d> &ts);

Eigen::Matrix3d essentialMatrixFromPose(const Eigen::Matrix3d &R, 
                                        const Eigen::Vector3d &t);

/**
 * @brief Recover up-to-scale pose from the given essential matrix E.
 * The pose of camera 1 is set to [I | 0] and the pose of camera 2 is [R | t].
 * 
 * @param E          the given essential matrix
 * @param points1    normalized camera coordinates of points in camera 1
 * @param points2    normalized camera coordinates of points in camera 2
 * @param R[out]     the result rotation  
 * @param t[out]     normalized result up-to-scale translation 
 * @param points_3d  the 3D coordinates of points that satisfy the cheirality constraint
*/
void poseFromEssentialMatrix(const Eigen::Matrix3d &E, 
                             const EigenVec<Eigen::Vector2d> &points1, 
                             const EigenVec<Eigen::Vector2d> &points2, 
                             Eigen::Matrix3d &R, 
                             Eigen::Vector3d &t, 
                             EigenVec<Eigen::Vector3d> &points_3d);

}
}

#endif // _MY3D_ESTIMATOR_ESSENTIAL_MATRIX_ESTIMATOR_H_
