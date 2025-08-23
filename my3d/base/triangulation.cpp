/*
 * filename: triangulation.cpp 
 * author:   Peiyan Liu, nROS-LAB, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Algorithms about triangulation.
 */

#include "base/triangulation.h"

namespace my3d {
namespace base {

Eigen::Vector3d triangulatePoint(const Eigen::Matrix3x4d &proj1, 
                                 const Eigen::Matrix3x4d &proj2, 
                                 const Eigen::Vector2d &point1, 
                                 const Eigen::Vector2d &point2) {
    Eigen::Matrix4d A;
    
    A.row(0) = point1.x() * proj1.row(2) - proj1.row(0);
    A.row(1) = point1.y() * proj1.row(2) - proj1.row(1);
    A.row(2) = point2.x() * proj2.row(2) - proj2.row(0);
    A.row(3) = point2.y() * proj2.row(2) - proj2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd_A(A, Eigen::ComputeFullV);
    const Eigen::Vector4d triangulated_point_h = svd_A.matrixV().col(3); 
    const double norm_factor = triangulated_point_h(3);

    return Eigen::Vector3d{triangulated_point_h(0) / norm_factor, 
                           triangulated_point_h(1) / norm_factor, 
                           triangulated_point_h(2) / norm_factor};
}

void triangulatePoint(const Eigen::Matrix3x4d &proj1, 
                       const Eigen::Matrix3x4d &proj2, 
                       const EigenVec<Eigen::Vector2d> &points1, 
                       const EigenVec<Eigen::Vector2d> &points2, 
                       EigenVec<Eigen::Vector3d> &triangulated_points) {
    assert(points1.size() == points2.size());

    const size_t num_pnts = points1.size();
    triangulated_points.resize(num_pnts);
    for (size_t i = 0; i < num_pnts; ++i) {
        triangulated_points[i] = triangulatePoint(proj1, proj2, points1[i], points2[i]);
    }
}

Eigen::Vector3d triangulatePointMultiView(const EigenVec<Eigen::Matrix3x4d> &projs, 
                                          const EigenVec<Eigen::Vector2d> &points) {
    assert(!projs.empty() && !points.empty()); 
    assert(projs.size() == points.size()); 

    Eigen::Matrix4d A = Eigen::Matrix4d::Zero(); 
    for (size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d point = points[i].homogeneous().normalized(); 
        const Eigen::Matrix3x4d term = 
            projs[i] - point * point.transpose() * projs[i]; 
        A += term.transpose() * term; 
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

    return eigen_solver.eigenvectors().col(0).hnormalized();

    // const size_t num_views = projs.size();
    // Eigen::MatrixXd A(2 * num_views, 4); 
    // A.setZero(); 
    // for (size_t i = 0; i < num_views; ++i) {
    //     A.row(2 * i) = points[i].x() * projs[i].row(2) - projs[i].row(0); 
    //     A.row(2 * i + 1) = points[i].y() * projs[i].row(2) - projs[i].row(1); 
    // }

    // Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A, Eigen::ComputeFullV);
    // Eigen::Vector4d triangulated_point_h = svd_A.matrixV().col(3);

    // return triangulated_point_h.hnormalized();
}

double computeTriangulationAngle(const Eigen::Vector3d &t1, 
                                 const Eigen::Vector3d &t2, 
                                 const Eigen::Vector3d &point_3D) {
    const Eigen::Vector3d vec1 = t1 - point_3D;
    const Eigen::Vector3d vec2 = t2 - point_3D; 
    const double squared_base_line = (t1 - t2).squaredNorm(); 
    if (vec1.norm() * vec2.norm() <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    
    const double cos_tri_angle = 
        (vec1.squaredNorm() + vec2.squaredNorm() - squared_base_line) / 
        (2.0 * vec1.norm() * vec2.norm()); 
    const double tri_angle = std::abs(std::acos(cos_tri_angle)); 

    return std::min(tri_angle, M_PI - tri_angle);
}

}
}
