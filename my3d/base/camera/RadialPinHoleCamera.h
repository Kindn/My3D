#ifndef _MY3D_BASE_CAMERA_RADIAL_PIN_HOLE_CAMERA_H_
#define _MY3D_BASE_CAMERA_RADIAL_PIN_HOLE_CAMERA_H_

#include "base/camera/CameraBase.h"
#include "base/polynomial.h" 

namespace my3d {
namespace base {

struct RadialPinHoleCamera : public CameraBase {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RadialPinHoleCamera() {}
    RadialPinHoleCamera(const Eigen::Matrix3d &K, double k1 = 0., double k2 = 0.);
    virtual ~RadialPinHoleCamera() {}

    virtual Eigen::Vector2d project(const Eigen::Vector3d &point3d) const override; 

    /** 
     * @brief Transform a pixel to normalized camera coordinate 
    */
    virtual Eigen::Vector2d pix2Norm(const Eigen::Vector2d &pix) const; 

    /**
     * @brief Convert a threshold in pixel plane to normalized plane 
    */
    virtual double thresholdPix2Norm(const double &threshold) const override; 

    // virtual bool hasBogusParams() const override; 

    Eigen::Matrix3d getCameraMatrix() const;

    /**
     * @brief Convert a pixel coordinate to normalized camera coordinate 
    */
    Eigen::Vector2d pixel2NormalizedIgnoringDistortion(const Eigen::Vector2d &pix) const; 

    /** 
     * @brief Find the original of a distorted normalized point 
     */
    Eigen::Vector2d undistort(const Eigen::Vector2d &distorted_point_2D) const; 

    double fx_;
    double fy_;
    double cx_;
    double cy_;
    double alpha_{0.};
    double k1_{0.};
    double k2_{0.};
    // cv::Point2d pc_norm_;
};

}
}

#endif // _MY3D_BASE_CAMERA_RADIAL_PIN_HOLE_CAMERA_H_
