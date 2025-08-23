/*
 * filename: OpenCVImageReader.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_IMAGE_OPENCV_IMAGE_READER_H_
#define _MY3D_BASE_IMAGE_OPENCV_IMAGE_READER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "base/image/ImageReaderBase.h"
#include "base/image/exif.h"
#include "base/camera/RadialPinHoleCamera.h"

namespace my3d {
namespace base {

/**
 * @brief Image reader based on OpenCV.
 */
class OpenCVImageReader : public ImageReaderBase {
public: 
    OpenCVImageReader() {}
    virtual ~OpenCVImageReader() {}

public: 
    virtual bool read(const std::string &path, 
                      Image &out, 
                      ImReadMode mode) const override;
    
    virtual bool save(const std::string &path, 
                      const Image &src) const override;
};

}
}

#endif // _MY3D_BASE_IMAGE_OPENCV_IMAGE_READER_H_
