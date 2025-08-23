#include <iostream>

#include "base/image/OpenCVImageReader.h"
#include "base/image/image_processing.h"

using namespace my3d; 

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Error! Invalid argument. Usage: \n" << 
                     "\tsparse_reconstruction <image_directory>" << std::endl; 
        return -1;
    }
    const std::string image_path = argv[1]; 

    base::OpenCVImageReader image_reader; 
    base::Image image; 
    image_reader.read(image_path, image, base::ImageReaderBase::IMREAD_COLOR); 

    base::Image blurred, scaled; 

    base::blurGaussian(image, blurred, 1); 
    image_reader.save("blurred_1.jpg", blurred); 
    base::blurGaussian(image, blurred, 8); 
    image_reader.save("blurred_8.jpg", blurred); 

    // base::scaleImage(image, scaled, 0.7); 
    // image_reader.save("scaled_0_7.jpg", scaled); 
    // base::scaleImage(image, scaled, 1.7); 
    // image_reader.save("scaled_1_7.jpg", scaled); 

    // base::Image gray; 
    // base::convertImageColorBGR2GrayScale(image, gray); 
    // image_reader.save("gray.jpg", gray); 

    return 0; 
}