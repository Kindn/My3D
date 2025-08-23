/*
 * filename: exif.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_IMAGE_CAMERA_EXIF_H_
#define _MY3D_BASE_IMAGE_CAMERA_EXIF_H_

#include "TinyEXIF/TinyEXIF.h"
#include "base/camera/CameraDatabase.h"

namespace my3d {
namespace base {

enum FocalLengthPriorType {
    FROM_EXIF_FOCAL_LENGTH_35MM = 0, 
    FROM_CAMERA_DATABASE = 1, 
    FROM_DEFAULT_VALUE = 2, 
    FROM_EXIF_CALIB_INFO = 3
};

int extractIntrinsicsFromEXIF(const TinyEXIF::EXIFInfo &exif_info, 
                               double &fx, double &fy, double cx, double cy, 
                               const size_t &width, const size_t &height);

}
}

#endif // _MY3D_BASE_IMAGE_CAMERA_EXIF_H_
