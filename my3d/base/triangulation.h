/*
 * filename: triangulation.cpp 
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about triangulation.
 */

#ifndef _MY3D_BASE_TRIANGULATION_H_
#define _MY3D_BASE_TRIANGULATION_H_

#include <iostream>
#include <vector>

#include "utils/eigen_types.h"

namespace my3d {
namespace base {

/**
 * @brief Triangulate a given point. 
*/
Eigen::Vector3d triangulatePoint(const Eigen::Matrix3x4d &proj1, 
                                 const Eigen::Matrix3x4d &proj2, 
                                 const Eigen::Vector2d &point1, 
                                 const Eigen::Vector2d &point2);

void triangulatePoint(const Eigen::Matrix3x4d &proj1, 
                       const Eigen::Matrix3x4d &proj2, 
                       const EigenVec<Eigen::Vector2d> &points1, 
                       const EigenVec<Eigen::Vector2d> &points2, 
                       EigenVec<Eigen::Vector3d> &triangulated_points);

Eigen::Vector3d triangulatePointMultiView(const EigenVec<Eigen::Matrix3x4d> &projs, 
                                          const EigenVec<Eigen::Vector2d> &points); 

double computeTriangulationAngle(const Eigen::Vector3d &t1, 
                                 const Eigen::Vector3d &t2, 
                                 const Eigen::Vector3d &point_3D);

}
}

#endif // _MY3D_BASE_TRIANGULATION_H_
