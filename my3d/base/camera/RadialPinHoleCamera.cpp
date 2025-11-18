#include "base/camera/RadialPinHoleCamera.h"

namespace my3d {
namespace base {

RadialPinHoleCamera::RadialPinHoleCamera(const Eigen::Matrix3d &K, double k1, double k2) :
fx_(K(0, 0)), fy_(K(1, 1)), 
cx_(K(0, 2)), cy_(K(1, 2)), 
alpha_(K(0, 1)), 
k1_(k1), k2_(k2) {
    assert(fx_ > 0 && fy_ > 0 && 
           cx_ >= 0 && cy_ >= 0);
}

Eigen::Vector2d RadialPinHoleCamera::project(const Eigen::Vector3d &point3d) const {
    // assert(point3d.z > 0 && "Point's Z in camera frame should be > 0. ");

    Eigen::Vector2d pc_norm = point3d.hnormalized();
    const double r2 = pc_norm.x() * pc_norm.x() + pc_norm.y() * pc_norm.y();
    const double distortion = 1. + r2 * k1_ + r2 * r2 * k2_;
    pc_norm *= distortion;

    Eigen::Vector2d proj;
    proj.x() = fx_ * pc_norm.x() + alpha_ * pc_norm.y() + cx_;
    proj.y() = fy_ * pc_norm.y() + cy_;

    return proj;
} 

Eigen::Vector2d RadialPinHoleCamera::pix2Norm(const Eigen::Vector2d &pix) const {
    Eigen::Vector2d pc_norm; 
    pc_norm.y() = (pix.y() - cy_) / fy_; 
    pc_norm.x() = (pix.x() - alpha_ * pc_norm.y() - cx_) / fx_; 
    // return pc_norm; 
    return undistort(pc_norm); 
}

Eigen::Vector3d RadialPinHoleCamera::pix2Sphere(const Eigen::Vector2d &pix) const {
    Eigen::Vector3d normalized_coord{};
    normalized_coord.head<2>() = pix2Norm(pix);
    normalized_coord.z() = 1.0;
    return normalized_coord.normalized();
}

double RadialPinHoleCamera::thresholdPix2Norm(const double &threshold) const {
    const double mean_focal_length = 0.5 * (fx_ + fy_); 
    return threshold / mean_focal_length; 
}

Eigen::Matrix3d RadialPinHoleCamera::getCameraMatrix() const {
    Eigen::Matrix3d K;
    K << fx_, alpha_, cx_, 
         0, fy_, cy_, 
         0, 0, 1;
    
    return K;
}

Eigen::Vector2d RadialPinHoleCamera::pixel2NormalizedIgnoringDistortion(const Eigen::Vector2d &pix) const {
    return Eigen::Vector2d((pix.x() - cx_) / fx_, 
                           (pix.y() - cy_) / fy_); 
} 

Eigen::Vector2d RadialPinHoleCamera::undistort(const Eigen::Vector2d &distorted_point_2D) const {
    const double r_dist = distorted_point_2D.norm(); 
    const Eigen::Vector2d vec = distorted_point_2D.normalized(); 
    if (std::abs(vec.norm() - 1.0) > 1.0e-6) {
        return distorted_point_2D; 
    }

    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(6); 
    coeffs(0) = r_dist; 
    coeffs(1) = -1.0; 
    coeffs(3) = -k1_; 
    coeffs(5) = -k2_; 
    Eigen::VectorXd roots_real, roots_imag; 
    if (!base::solvePolynomialCompanionMatrix(coeffs, roots_real, roots_imag)) {
        return distorted_point_2D; 
    } 

    double r_orig; 
    for (Eigen::Index i = 0; i < roots_real.size(); ++i) {
        if (roots_real(i) >= 0 && 
            std::abs(roots_imag(i)) <= std::numeric_limits<double>::epsilon()) {
                r_orig = roots_real(i); 
                break; 
            }
    } 

    return r_orig * vec; 
}

}
}
