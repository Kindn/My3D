/*
 * filename: eigen_types.h
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Type defininition of some Eigen types.
 */

#ifndef _MY3D_UTIL_EIGEN_TYPES_H_
#define _MY3D_UTIL_EIGEN_TYPES_H_

#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>

#include <eigen3/Eigen/Eigen>

namespace Eigen {
    typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
    typedef Eigen::Matrix<double, 4, 3> Matrix4x3d;
    typedef Eigen::Matrix<double, 5, 1> Vector5d; 
    typedef Eigen::Matrix<double, 6, 1> Vector6d; 
}

namespace my3d {
    template <typename EigenMatType>
    using EigenVec = std::vector<EigenMatType, Eigen::aligned_allocator<EigenMatType>>; 

    template <typename EigenMatType, size_t Len> 
    using EigenArray = std::array<EigenMatType, Len>; 

    template <typename KeyType, typename ValueType> 
    using EigenUMap = std::unordered_map<KeyType, ValueType, 
                                         std::hash<KeyType>, std::equal_to<KeyType>, 
                                         Eigen::aligned_allocator<std::pair<KeyType const, ValueType>>>;
}

#endif // _MY3D_UTILS_EIGEN_TYPES_H_