/*
 * filename: image_io.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "utils/io.h"
#include "base/image/Image.h"
#include "base/image/ImageReaderBase.h"

namespace my3d {
namespace base {

extern const std::unordered_set<std::string> 
    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".JPG", 
                                  ".png", ".PNG", 
                                  ".jpeg", ".JPEG"}; 

template <class ImageReaderType> 
size_t readImages(const std::string &directory, 
                  const double &fx_prior, 
                  const double &fy_prior, 
                  std::vector<Image> &images, 
                  const std::unordered_set<std::string> &extensions = SUPPORTED_IMAGE_EXTENSIONS, 
                  const ImageReaderBase::ImReadMode &mode = ImageReaderBase::ImReadMode::IMREAD_COLOR) {
    size_t num_read = 0; 
    ImageReaderType reader;
    reader.setFocalLengthPrior(fx_prior, fy_prior); 

    const std::vector<std::string> files = 
        util::collectFiles(directory, extensions);
    images.clear();  
    for (const std::string &file : files) {
        Image image; 
        if (reader.read(file, image, mode)) {
            image.setId(num_read); 
            images.push_back(image); 
            num_read++; 
        }
    }

    return num_read; 
}

}
}
