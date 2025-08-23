/*
 * filename: ImageReaderBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _MY3D_BASE_IMAGE_IMAGE_READER_BASE_H_
#define _MY3D_BASE_IMAGE_IMAGE_READER_BASE_H_

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "utils/eigen_types.h"
#include "base/image/Image.h"

namespace my3d {
namespace base {

class ImageReaderBase {
public: 
    enum ImReadMode {
        IMREAD_GRAYSCALE = 0, 
        IMREAD_COLOR = 1
    };

public: 
    ImageReaderBase() {}
    virtual ~ImageReaderBase() {}

public: 
    virtual bool read(const std::string &path, 
                      Image &out, 
                      ImReadMode mode = ImReadMode::IMREAD_COLOR) const = 0;
    
    virtual bool save(const std::string &path, 
                      const Image &src) const = 0;

    void setFocalLengthPrior(const double &fx_prior, 
                             const double &fy_prior) {
        // assert(fx_prior > std::numeric_limits<double>::epsilon()); 
        // assert(fy_prior > std::numeric_limits<double>::epsilon());

        fx_prior_ = fx_prior; 
        fy_prior_ = fy_prior; 
    }

protected: 
    double fx_prior_{500.0};
    double fy_prior_{500.0}; 
};

}
}

#endif // _MY3D_BASE_IMAGE_IMAGE_READER_BASE_H_
