#ifndef _MY3D_BASE_CAMERA_CAMERA_BASE_H_
#define _MY3D_BASE_CAMERA_CAMERA_BASE_H_

#include <iostream>
#include <map>
#include <vector>
#include <memory>

#include <eigen3/Eigen/Eigen>

namespace my3d {
namespace base {

struct CameraBase {
    CameraBase() {}
    virtual ~CameraBase() {}

    virtual Eigen::Vector2d project(const Eigen::Vector3d &point3d) const = 0;

    /** 
     * @brief Transform a pixel to normalized camera coordinate 
    */
    virtual Eigen::Vector2d pix2Norm(const Eigen::Vector2d &pix) const = 0; 

    /**
     * @brief Convert a threshold in pixel plane to normalized plane 
    */
    virtual double thresholdPix2Norm(const double &threshold) const = 0;
    // virtual bool hasBogusParams() const = 0;
};

} // namespace camera
} // namespace calib

#endif // _MY3D_BASE_CAMERA_CAMERA_BASE_H_
