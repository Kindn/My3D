/*
 * filename: FundamentalMatrixEstimator.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_ESTIMATOR_FUNDAMENTAL_MATRIX_ESTIMATOR_H_
#define _MY3D_ESTIMATOR_FUNDAMENTAL_MATRIX_ESTIMATOR_H_

#include <iostream>

#include "base/pose.h"
#include "base/points.h"
#include "base/polynomial.h"
#include "estimators/estimator_utils.h"

namespace my3d {
namespace estimator {

class FundamentalMatrixSevenPointEstimator {
public: 
    const static size_t kMinNumSamples = 7;

    typedef Eigen::Vector2d XDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> XAllocType;
    typedef std::vector<XDataType, XAllocType> VecXData;
    typedef Eigen::Vector2d YDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> YAllocType;
    typedef std::vector<YDataType, YAllocType> VecYData;
    typedef Eigen::Matrix3d ModelType;

    explicit FundamentalMatrixSevenPointEstimator();
    ~FundamentalMatrixSevenPointEstimator();

public: 
    /**
     * @brief Estimate the fundamental matrix F which satisfies y' F x == 0
     * 
     * @param x_data pixel coordinates of points in camera 1
     * @param y_data pixel coordinates of points in camera 2
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

class FundamentalMatrixEightPointEstimator {
public: 
    const static size_t kMinNumSamples = 8;

    typedef Eigen::Vector2d XDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> XAllocType;
    typedef std::vector<XDataType, XAllocType> VecXData;
    typedef Eigen::Vector2d YDataType;
    typedef Eigen::aligned_allocator<Eigen::Vector2d> YAllocType;
    typedef std::vector<YDataType, YAllocType> VecYData;
    typedef Eigen::Matrix3d ModelType;

    explicit FundamentalMatrixEightPointEstimator();
    ~FundamentalMatrixEightPointEstimator();

public: 
    /**
     * @brief Estimate the fundamental matrix F which satisfies y' F x == 0
     * 
     * @param x_data pixel coordinates of points in camera 1
     * @param y_data pixel coordinates of points in camera 2
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

}
}

#endif // _MY3D_ESTIMATOR_FUNDAMENTAL_MATRIX_ESTIMATOR_H_
